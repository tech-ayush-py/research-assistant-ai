"""
Literature Mining Agent
Crawls ArXiv and Semantic Scholar, builds paper embeddings, stores in ChromaDB.

Discipline-aware fetch strategy:
- Technical fields (CS, ML, Physics, etc.) → split evenly between ArXiv and S2.
- Non-technical fields (History, Literature, Social Science, etc.) → skip ArXiv
  entirely (ArXiv is a preprint server dominated by STEM/CS papers and returns
  irrelevant results for humanities/social science queries), rely fully on
  Semantic Scholar which covers all academic disciplines.
- A lightweight CS-jargon filter is also applied to ArXiv results so that even
  for borderline topics, unrelated ML papers are not injected into the corpus.

Credibility-weighted ranking:
- Semantic Scholar papers carry citationCount and influentialCitationCount.
- A credibility score is computed per paper and blended with semantic similarity
  so that highly-cited, peer-reviewed, influential papers rank above obscure ones
  even when their abstract embedding similarity is equal.
"""
import logging
import math
import time
from typing import List, Dict, Any

import arxiv
import requests

from config.settings import ARXIV_MAX
from core.vector_store import ingest_papers, similarity_search

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Keywords that strongly indicate a topic is NOT suited for ArXiv
_NON_ARXIV_SIGNALS = {
    "history", "historical", "historiograph", "archiv", "medieval", "ancient",
    "colonial", "postcolonial", "war", "empire", "revolution", "dynasty",
    "literature", "literary", "novel", "poetry", "narrative", "fiction",
    "philosophy", "philosophical", "ethics", "metaphysics", "epistemolog",
    "anthropology", "anthropological", "ethnograph", "culture", "cultural",
    "sociology", "sociolog", "social movement", "gender studies", "feminism",
    "political science", "governance", "diplomacy", "geopolitics",
    "economics", "econom", "fiscal", "monetary policy", "labour market",
    "psychology", "psycholog", "mental health", "cognitive behaviour",
    "education", "pedagog", "curriculum", "teaching",
    "law", "legal", "jurisprudence", "constitutional",
    "art history", "museum", "archaeology", "archaeolog",
    "theology", "religion", "religious", "spiritual",
    "espionage", "intelligence agenc", "cold war", "propaganda",
    "public health", "epidemiolog", "global health",
}

# CS/ML jargon that has no place in non-technical paper corpora
_CS_JARGON_FILTER = {
    "neural network", "deep learning", "transformer", "large language model",
    "llm", "bert", "gpt", "reinforcement learning", "convolutional",
    "object detection", "image segmentation", "natural language processing",
    "federated learning", "generative adversarial", "diffusion model",
    "knowledge graph", "graph neural", "swarm robotics", "autonomous vehicle",
    "computer vision", "speech recognition", "machine translation",
}

# Credibility blend weights
_W_INFLUENTIAL  = 0.45   # S2 influentialCitationCount (strongest signal)
_W_CITATIONS    = 0.25   # raw citationCount
_W_RECENCY      = 0.20   # publication year (2000-present, scaled 0-1)
_W_ABSTRACT     = 0.10   # has a non-empty abstract

# Final score blend
_W_SIMILARITY   = 0.55   # semantic relevance stays primary
_W_CREDIBILITY  = 0.45   # credibility lifts or demotes


def _s2_headers() -> dict:
    """
    Build Semantic Scholar request headers at call time (not at import time).
    Bug 6 fix: the old module-level S2_HEADERS dict was built once during import,
    so any key injected into os.environ afterward (e.g. via Streamlit secrets) was
    silently ignored for the entire session.
    """
    import os
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    return {"x-api-key": key} if key else {}


def _is_non_technical_topic(query: str) -> bool:
    """Return True if the query belongs to humanities / social sciences / non-STEM fields."""
    q = query.lower()
    return any(signal in q for signal in _NON_ARXIV_SIGNALS)


def _is_cs_jargon_paper(paper: Dict[str, Any]) -> bool:
    """Return True if the paper is clearly a CS/ML paper unrelated to the query field."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    matches = sum(1 for jargon in _CS_JARGON_FILTER if jargon in text)
    return matches >= 2


def _credibility_score(paper: Dict[str, Any]) -> float:
    """
    Return a 0-1 credibility score based on citation signals and recency.

    Uses log-scaling for citation counts so that the difference between
    0 and 10 citations matters as much as the difference between 100 and 1000.
    Papers with no citation data (e.g. ArXiv preprints) get a neutral 0.3.
    """
    citations     = paper.get("citation_count", None)
    influential   = paper.get("influential_citation_count", None)
    year          = paper.get("year", None)
    has_abstract  = 1.0 if paper.get("abstract", "").strip() else 0.0

    # If no citation data at all (ArXiv paper), return neutral score
    if citations is None and influential is None:
        recency = _recency_score(year)
        return round(0.3 + 0.1 * recency + 0.1 * has_abstract, 4)

    # Log-scale normalisation: log1p(x) / log1p(cap)
    # Cap at 10,000 citations and 500 influential — beyond that returns diminish
    cite_norm  = math.log1p(citations or 0)    / math.log1p(10_000)
    inf_norm   = math.log1p(influential or 0)  / math.log1p(500)
    recency    = _recency_score(year)

    score = (
        _W_INFLUENTIAL * inf_norm
        + _W_CITATIONS * cite_norm
        + _W_RECENCY   * recency
        + _W_ABSTRACT  * has_abstract
    )
    return round(min(score, 1.0), 4)


def _recency_score(year) -> float:
    """Scale year linearly from 0 (year 2000 or earlier) to 1 (current year)."""
    if not year:
        return 0.5
    try:
        y = int(year)
    except (ValueError, TypeError):
        return 0.5
    import datetime
    current = datetime.datetime.now().year
    return max(0.0, min(1.0, (y - 2000) / max(current - 2000, 1)))


def _rank_papers(papers: List[Dict[str, Any]], query: str, top_n: int) -> List[Dict[str, Any]]:
    """
    Rank papers by a blend of semantic similarity (from ChromaDB) and
    credibility score (from citation metadata).

    Steps:
    1. Run similarity_search to get similarity scores for all ingested papers.
    2. Build a lookup from paper title → similarity score.
    3. For each paper, compute credibility_score and blend with similarity.
    4. Sort descending by final_score, return top_n.
    """
    # Get similarity scores for all papers in corpus against the query
    similar = similarity_search(query, n_results=min(len(papers) + 50, 100))
    sim_lookup = {p["title"].strip().lower(): p["similarity"] for p in similar}

    ranked = []
    for p in papers:
        title_key  = p.get("title", "").strip().lower()
        similarity = sim_lookup.get(title_key, 0.5)   # neutral if not found
        credibility = _credibility_score(p)
        final_score = _W_SIMILARITY * similarity + _W_CREDIBILITY * credibility

        ranked.append({
            **p,
            "similarity":   round(similarity, 4),
            "credibility":  credibility,
            "final_score":  round(final_score, 4),
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_n]


class LiteratureMiningAgent:
    """
    Fetches papers from ArXiv + Semantic Scholar for a given research query,
    embeds them via ChromaDB, and returns top papers ranked by a blend of
    semantic similarity and credibility (citation count + recency).

    Source allocation is discipline-aware:
    - Non-technical topics skip ArXiv (CS-heavy) and use Semantic Scholar only.
    - Technical topics split the budget evenly between both sources.
    - ArXiv results for borderline topics are filtered to remove CS-jargon papers.
    """

    name = "Literature Mining Agent"
    description = (
        "Crawls ArXiv and Semantic Scholar for papers matching a research topic, "
        "ingests them into the vector store, and returns top papers ranked by "
        "credibility (citations + recency) blended with semantic relevance."
    )

    # ── public API ────────────────────────────────────────────

    def run(self, query: str, max_papers: int = 30) -> Dict[str, Any]:
        logger.info("[LiteratureMining] Query: %s", query)

        non_technical = _is_non_technical_topic(query)

        if non_technical:
            logger.info("[LiteratureMining] Non-technical topic — skipping ArXiv, using Semantic Scholar only.")
            arxiv_papers = []
            s2_papers    = self._fetch_semantic_scholar(query, max_papers)
        else:
            half         = max_papers // 2
            arxiv_papers = self._fetch_arxiv(query, half)
            s2_papers    = self._fetch_semantic_scholar(query, half)

        all_papers = arxiv_papers + s2_papers

        # Drop CS-jargon papers for non-technical topics (safety net)
        if non_technical:
            before = len(all_papers)
            all_papers = [p for p in all_papers if not _is_cs_jargon_paper(p)]
            dropped = before - len(all_papers)
            if dropped:
                logger.info("[LiteratureMining] Filtered out %d CS-jargon papers.", dropped)

        # Ingest into ChromaDB (builds the embedding corpus)
        new_count = ingest_papers(all_papers)

        # Rank by credibility + similarity blend
        top_papers = _rank_papers(all_papers, query, top_n=15)

        logger.info(
            "[LiteratureMining] Fetched %d, ingested %d new. Top paper: '%s' "
            "(similarity=%.3f, credibility=%.3f, final=%.3f)",
            len(all_papers), new_count,
            top_papers[0]["title"][:60] if top_papers else "—",
            top_papers[0].get("similarity", 0) if top_papers else 0,
            top_papers[0].get("credibility", 0) if top_papers else 0,
            top_papers[0].get("final_score", 0) if top_papers else 0,
        )

        return {
            "agent":         self.name,
            "query":         query,
            "fetched":       len(all_papers),
            "new_ingested":  new_count,
            "top_papers":    top_papers,
            "non_technical": non_technical,
        }

    # ── ArXiv ─────────────────────────────────────────────────

    def _fetch_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        ArXiv has no citation API. Papers are returned with citation_count=None
        so the credibility scorer gives them a neutral score based on recency only.
        """
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=min(limit, ARXIV_MAX),
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in search.results():
                papers.append({
                    "title":                    r.title,
                    "abstract":                 r.summary,
                    "authors":                  [str(a) for a in r.authors],
                    "year":                     r.published.year if r.published else "",
                    "url":                      r.entry_id,
                    "source":                   "arxiv",
                    "paper_id":                 r.entry_id.split("/")[-1],
                    "citation_count":           None,   # not available from ArXiv
                    "influential_citation_count": None,
                    "is_open_access":           True,   # ArXiv is always OA
                    "publication_types":        ["Preprint"],
                })
        except Exception as exc:
            logger.warning("[LiteratureMining] ArXiv error: %s", exc)
        return papers

    # ── Semantic Scholar ───────────────────────────────────────

    def _fetch_semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetches citationCount, influentialCitationCount, publicationTypes, and
        openAccessPdf in addition to the standard fields. These are stored on
        each paper dict and used by _credibility_score().
        """
        papers = []
        try:
            params = {
                "query":  query,
                "limit":  limit,
                "fields": (
                    "title,abstract,authors,year,url,externalIds,"
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
                    oa_pdf = item.get("openAccessPdf") or {}
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
                        "is_open_access":            bool(oa_pdf.get("url")),
                        "publication_types":         item.get("publicationTypes") or [],
                        "journal":                   (item.get("journal") or {}).get("name", ""),
                    })
            time.sleep(0.5)  # respect rate limit
        except Exception as exc:
            logger.warning("[LiteratureMining] Semantic Scholar error: %s", exc)
        return papers
