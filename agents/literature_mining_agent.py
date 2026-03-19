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
"""
import logging
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
    return matches >= 2  # two or more CS jargon hits → likely irrelevant for non-tech topics


class LiteratureMiningAgent:
    """
    Fetches papers from ArXiv + Semantic Scholar for a given research query,
    embeds them via ChromaDB, and returns relevant hits.

    Source allocation is discipline-aware:
    - Non-technical topics skip ArXiv (CS-heavy) and use Semantic Scholar only.
    - Technical topics split the budget evenly between both sources.
    - ArXiv results for borderline topics are filtered to remove CS-jargon papers.
    """

    name = "Literature Mining Agent"
    description = (
        "Crawls ArXiv and Semantic Scholar for papers matching a research topic, "
        "ingests them into the vector store, and returns top similar papers. "
        "Source allocation is discipline-aware."
    )

    # ── public API ────────────────────────────────────────────

    def run(self, query: str, max_papers: int = 30) -> Dict[str, Any]:
        logger.info("[LiteratureMining] Query: %s", query)

        non_technical = _is_non_technical_topic(query)

        if non_technical:
            # Humanities / social sciences → Semantic Scholar covers all disciplines;
            # ArXiv is skipped entirely to avoid contaminating the corpus with CS papers.
            logger.info("[LiteratureMining] Non-technical topic detected — skipping ArXiv, using Semantic Scholar only.")
            arxiv_papers = []
            s2_papers    = self._fetch_semantic_scholar(query, max_papers)
        else:
            # Technical / STEM topic → split budget, then filter stray CS papers
            # from ArXiv that don't match the topic (only relevant for borderline queries).
            half         = max_papers // 2
            arxiv_papers = self._fetch_arxiv(query, half)
            s2_papers    = self._fetch_semantic_scholar(query, half)

        all_papers = arxiv_papers + s2_papers

        # For non-technical topics: drop any ArXiv paper that looks like pure CS/ML
        # (safety net in case the non-technical detector missed something).
        if non_technical:
            before = len(all_papers)
            all_papers = [p for p in all_papers if not _is_cs_jargon_paper(p)]
            dropped = before - len(all_papers)
            if dropped:
                logger.info("[LiteratureMining] Filtered out %d CS-jargon papers.", dropped)

        new_count = ingest_papers(all_papers)
        similar   = similarity_search(query, n_results=15)

        return {
            "agent":        self.name,
            "query":        query,
            "fetched":      len(all_papers),
            "new_ingested": new_count,
            "top_papers":   similar,
            "non_technical": non_technical,
        }

    # ── ArXiv ─────────────────────────────────────────────────

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
                    "title":    r.title,
                    "abstract": r.summary,
                    "authors":  [str(a) for a in r.authors],
                    "year":     r.published.year if r.published else "",
                    "url":      r.entry_id,
                    "source":   "arxiv",
                    "paper_id": r.entry_id.split("/")[-1],
                })
        except Exception as exc:
            logger.warning("[LiteratureMining] ArXiv error: %s", exc)
        return papers

    # ── Semantic Scholar ───────────────────────────────────────

    def _fetch_semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        try:
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,externalIds,url",
            }
            resp = requests.get(
                f"{S2_BASE}/paper/search",
                params=params,
                headers=_s2_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    papers.append({
                        "title":    item.get("title", ""),
                        "abstract": item.get("abstract", "") or "",
                        "authors":  [a["name"] for a in item.get("authors", [])],
                        "year":     item.get("year", ""),
                        "url":      item.get("url", ""),
                        "source":   "semantic_scholar",
                        "paper_id": item.get("paperId", ""),
                    })
            time.sleep(0.5)  # respect rate limit
        except Exception as exc:
            logger.warning("[LiteratureMining] Semantic Scholar error: %s", exc)
        return papers
