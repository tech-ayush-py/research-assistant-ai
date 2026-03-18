"""
Plagiarism & Novelty Scoring Agent
Computes semantic similarity of a proposed research idea against the paper corpus.
"""
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer
from core.vector_store import similarity_search, collection_stats
from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


class NoveltyScoringAgent:
    """
    Scores a research proposal's novelty by:
    1. Embedding the proposal abstract.
    2. Comparing against top-k corpus papers via cosine similarity.
    3. Returning per-paper overlap + aggregate novelty score (0-1).
    """

    name = "Plagiarism & Novelty Scoring Agent"
    description = (
        "Computes semantic similarity between a research proposal and the existing "
        "literature corpus to produce a novelty score and plagiarism risk flag."
    )

    NOVELTY_THRESHOLDS = {
        "highly_novel":   (0.75, 1.0),
        "novel":          (0.55, 0.75),
        "moderately_novel": (0.35, 0.55),
        "low_novelty":    (0.0,  0.35),
    }

    def run(self, proposal_abstract: str, top_k: int = 10) -> Dict[str, Any]:
        logger.info("[NoveltyScoringAgent] Scoring proposal…")

        stats = collection_stats()
        if stats["total_papers"] == 0:
            return {"agent": self.name, "error": "No corpus. Run Literature Mining first."}

        similar = similarity_search(proposal_abstract, n_results=top_k)
        novelty_score = self._compute_novelty(similar)
        label         = self._label(novelty_score)
        high_risk     = [p for p in similar if p["similarity"] > 0.85]

        return {
            "agent":             self.name,
            "novelty_score":     novelty_score,
            "novelty_label":     label,
            "corpus_size":       stats["total_papers"],
            "papers_compared":   len(similar),
            "high_similarity_papers": [
                {"title": p["title"], "year": p["year"], "similarity": p["similarity"]}
                for p in high_risk
            ],
            "top_similar": [
                {"title": p["title"], "year": p["year"], "similarity": p["similarity"]}
                for p in similar[:5]
            ],
            "recommendation": self._recommend(novelty_score, high_risk),
        }

    # ── Helpers ───────────────────────────────────────────────

    def _compute_novelty(self, similar: List[Dict]) -> float:
        if not similar:
            return 1.0
        avg_sim = sum(p["similarity"] for p in similar[:5]) / min(5, len(similar))
        return round(1.0 - avg_sim, 3)

    def _label(self, score: float) -> str:
        for label, (lo, hi) in self.NOVELTY_THRESHOLDS.items():
            if lo <= score < hi:
                return label
        return "unknown"

    def _recommend(self, score: float, high_risk: list) -> str:
        if high_risk:
            titles = "; ".join(p["title"] for p in high_risk[:2])
            return (
                f"⚠️  High semantic overlap detected with existing work: {titles}. "
                "Strongly differentiate your contribution section."
            )
        if score >= 0.75:
            return "✅ Excellent novelty. Proposal appears highly original."
        if score >= 0.55:
            return "✅ Good novelty. Consider highlighting specific differentiators."
        return "⚠️  Moderate novelty. Review existing work carefully and sharpen your unique contributions."
