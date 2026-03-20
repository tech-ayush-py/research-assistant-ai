"""
Literature Mining Agent
Crawls ArXiv and Semantic Scholar, builds paper embeddings, stores in ChromaDB.

v2 improvements:
  - Fetches citationCount, influentialCitationCount, publicationTypes, journal
    from Semantic Scholar so credibility signals are available for ranking.
  - Credibility score blends log-scaled citations + recency + abstract quality.
  - Final ranking blends similarity (55%) + credibility (45%) so highly-cited
    papers beat obscure ones even if their abstract is slightly less similar.
  - ArXiv papers get a neutral credibility of 0.3 + recency (no citation data).
"""
import logging
import math
import time
from datetime import datetime
from typing import List, Dict, Any

import arxiv
import requests

from config.settings import ARXIV_MAX
from core.vector_store import ingest_papers, similarity_search

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
_CURRENT_YEAR = datetime.now().year


def _s2_headers() -> dict:
    import os
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    return {"x-api-key": key} if key else {}


def _credibility_score(paper: Dict[str, Any]) -> float:
    """
    Compute a 0-1 credibility score from citation signals and recency.

    Weights:
      45% influential citation count (log-scaled, cap 500)
      25% total citation count      (log-scaled, cap 10 000)
      20% recency                   (linear, year 2000 → now)
      10% has abstract              (data quality flag)

    ArXiv papers have no citation data → neutral 0.3 + recency bonus.
    """
    source = paper.get("source", "")
    year   = paper.get("year", "")

    # Recency component (0-1)
    try:
        yr = int(year) if year else 2000
    except (ValueError, TypeError):
        yr = 2000
    recency = max(0.0, min(1.0, (yr - 2000) / max(1, _CURRENT_YEAR - 2000)))

    if source == "arxiv":
        return round(0.3 + 0.1 * recency, 4)

    cites      = int(paper.get("citation_count", 0) or 0)
    inf_cites  = int(paper.get("influential_citation_count", 0) or 0)
    has_abs    = 1.0 if paper.get("abstract", "").strip() else 0.0

    c_inf    = math.log1p(inf_cites) / math.log1p(500)
    c_total  = math.log1p(cites)     / math.log1p(10_000)

    score = (
        0.45 * min(1.0, c_inf)   +
        0.25 * min(1.0, c_total) +
        0.20 * recency           +
        0.10 * has_abs
    )
    return round(min(1.0, score), 4)


def _blend_score(similarity: float, credibility: float) -> float:
    """Blend similarity (55%) and credibility (45%) into a final ranking score."""
    return round(0.55 * similarity + 0.45 * credibility, 4)


class LiteratureMiningAgent:
    """
    Fetches papers from ArXiv + Semantic Scholar, embeds them,
    and returns a credibility-aware ranked list.
    """

    name = "Literature Mining Agent"
    description = (
        "Crawls ArXiv and Semantic Scholar for papers matching a research topic, "
        "ranks by blended similarity + credibility, and returns top papers."
    )

    def run(self, query: str, max_papers: int = 30) -> Dict[str, Any]:
        logger.info("[LiteratureMining] Query: %s", query)

        arxiv_papers = self._fetch_arxiv(query, max_papers // 2)
        s2_papers    = self._fetch_semantic_scholar(query, max_papers // 2)
        all_papers   = arxiv_papers + s2_papers

        new_count = ingest_papers(all_papers)

        # Get similarity-ranked candidates, then re-rank with credibility
        candidates = similarity_search(query, n_results=min(len(all_papers) or 1, 30))
        ranked     = self._rank_with_credibility(candidates, all_papers)

        return {
            "agent":        self.name,
            "query":        query,
            "fetched":      len(all_papers),
            "new_ingested": new_count,
            "top_papers":   ranked[:15],
        }

    # ── Credibility ranking ────────────────────────────────────────────────────

    def _rank_with_credibility(
        self,
        candidates: List[Dict],
        all_papers: List[Dict],
    ) -> List[Dict]:
        """
        Attach credibility metadata to each similarity-ranked candidate,
        compute the blended score, and re-sort.
        """
        # Build a lookup from paper_id → full paper dict (for citation fields)
        lookup: Dict[str, Dict] = {}
        for p in all_papers:
            pid = p.get("paper_id", "")
            if pid:
                lookup[pid] = p

        enriched = []
        for c in candidates:
            pid     = c.get("paper_id", "")
            full    = lookup.get(pid, c)
            cred    = _credibility_score(full)
            blend   = _blend_score(c["similarity"], cred)
            enriched.append({
                **c,
                "citation_count":              int(full.get("citation_count", 0) or 0),
                "influential_citation_count":  int(full.get("influential_citation_count", 0) or 0),
                "journal":                     full.get("journal", ""),
                "credibility":                 cred,
                "score":                       blend,
            })

        enriched.sort(key=lambda x: x["score"], reverse=True)
        return enriched

    # ── ArXiv ─────────────────────────────────────────────────────────────────

    def _fetch_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=min(limit, ARXIV_MAX),
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in search.results():
                papers.append({
                    "title":                     r.title,
                    "abstract":                  r.summary,
                    "authors":                   [str(a) for a in r.authors],
                    "year":                      r.published.year if r.published else "",
                    "url":                       r.entry_id,
                    "source":                    "arxiv",
                    "paper_id":                  r.entry_id.split("/")[-1],
                    "citation_count":            0,
                    "influential_citation_count": 0,
                    "journal":                   "arXiv preprint",
                })
        except Exception as exc:
            logger.warning("[LiteratureMining] ArXiv error: %s", exc)
        return papers

    # ── Semantic Scholar ───────────────────────────────────────────────────────

    def _fetch_semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        try:
            params = {
                "query":  query,
                "limit":  limit,
                "fields": (
                    "title,abstract,authors,year,externalIds,url,"
                    "citationCount,influentialCitationCount,"
                    "publicationTypes,openAccessPdf,journal"
                ),
            }
            resp = requests.get(
                f"{S2_BASE}/paper/search",
                params=params,
                headers=_s2_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    journal_info = item.get("journal") or {}
                    papers.append({
                        "title":                     item.get("title", ""),
                        "abstract":                  item.get("abstract", "") or "",
                        "authors":                   [a["name"] for a in item.get("authors", [])],
                        "year":                      item.get("year", ""),
                        "url":                       item.get("url", ""),
                        "source":                    "semantic_scholar",
                        "paper_id":                  item.get("paperId", ""),
                        "citation_count":            item.get("citationCount", 0) or 0,
                        "influential_citation_count": item.get("influentialCitationCount", 0) or 0,
                        "journal":                   journal_info.get("name", "") if isinstance(journal_info, dict) else "",
                        "publication_types":         item.get("publicationTypes", []) or [],
                        "open_access":               bool(item.get("openAccessPdf")),
                    })
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("[LiteratureMining] Semantic Scholar error: %s", exc)
        return papers
