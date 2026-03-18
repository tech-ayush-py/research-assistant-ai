"""
Literature Mining Agent
Crawls ArXiv and Semantic Scholar, builds paper embeddings, stores in ChromaDB.
"""
import logging
import time
from typing import List, Dict, Any

import arxiv
import requests

from config.settings import ARXIV_MAX, S2_API_KEY
from core.vector_store import ingest_papers, similarity_search

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


class LiteratureMiningAgent:
    """
    Fetches papers from ArXiv + Semantic Scholar for a given research query,
    embeds them via ChromaDB, and returns relevant hits.
    """

    name = "Literature Mining Agent"
    description = (
        "Crawls ArXiv and Semantic Scholar for papers matching a research topic, "
        "ingests them into the vector store, and returns top similar papers."
    )

    # ── public API ────────────────────────────────────────────

    def run(self, query: str, max_papers: int = 30) -> Dict[str, Any]:
        logger.info("[LiteratureMining] Query: %s", query)

        arxiv_papers = self._fetch_arxiv(query, max_papers // 2)
        s2_papers    = self._fetch_semantic_scholar(query, max_papers // 2)
        all_papers   = arxiv_papers + s2_papers

        new_count = ingest_papers(all_papers)
        similar   = similarity_search(query, n_results=15)

        return {
            "agent":       self.name,
            "query":       query,
            "fetched":     len(all_papers),
            "new_ingested": new_count,
            "top_papers":  similar,
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
                headers=S2_HEADERS,
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
