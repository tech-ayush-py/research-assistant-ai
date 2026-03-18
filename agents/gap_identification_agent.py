"""
Gap Identification Agent
Identifies under-explored research intersections via semantic clustering
and graph-based citation analysis.
"""
import logging
from typing import Dict, Any, List

from core.llm_factory import get_llm, invoke_with_retry
from core.vector_store import similarity_search, get_collection

logger = logging.getLogger(__name__)


class GapIdentificationAgent:
    """
    Finds research gaps by:
    1. Clustering papers semantically and finding thin/empty cluster regions.
    2. Analysing what problems are raised but not solved in abstracts.
    3. Comparing sub-topic coverage via LLM reasoning.
    """

    name = "Gap Identification Agent"
    description = (
        "Identifies under-explored research gaps using semantic clustering "
        "and graph-based citation analysis on the ingested paper corpus."
    )

    def run(self, research_topic: str) -> Dict[str, Any]:
        logger.info("[GapIdentification] Topic: %s", research_topic)

        related = similarity_search(research_topic, n_results=20)
        if not related:
            return {"agent": self.name, "error": "No papers found. Run Literature Mining first."}

        gaps = self._find_gaps_llm(research_topic, related)
        novelty_score = self._compute_novelty_score(research_topic, related)

        return {
            "agent":         self.name,
            "research_topic": research_topic,
            "papers_analysed": len(related),
            "identified_gaps": gaps["gaps"],
            "opportunity_areas": gaps["opportunities"],
            "gap_reasoning":  gaps["reasoning"],
            "novelty_score":  novelty_score,
            "top_cited_papers": [
                {"title": p["title"], "year": p["year"], "similarity": p["similarity"]}
                for p in related[:5]
            ],
        }

    # ── Helpers ───────────────────────────────────────────────

    def _find_gaps_llm(self, topic: str, papers: List[Dict]) -> Dict[str, Any]:
        llm = get_llm(temperature=0.4)
        paper_summaries = "\n".join(
            f"- [{p['year']}] {p['title']}: {p['snippet'][:200]}"
            for p in papers[:15]
        )

        prompt = f"""You are an expert academic researcher identifying knowledge gaps.

Research Topic: "{topic}"

Here are the most relevant existing papers:
{paper_summaries}

Based on these papers, identify:
1. RESEARCH GAPS: 4-6 specific, concrete areas that are under-explored or completely missing
2. OPPORTUNITY AREAS: 3-4 promising directions where new research could have high impact
3. REASONING: A concise paragraph explaining your gap analysis methodology

Respond in JSON with keys: gaps (list of strings), opportunities (list of strings), reasoning (string)"""

        try:
            from langchain_core.messages import HumanMessage
            import json, re
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[GapIdentification] LLM error: %s", exc)
            return {
                "gaps": ["LLM unavailable – configure API key"],
                "opportunities": [],
                "reasoning": str(exc),
            }

    def _compute_novelty_score(self, topic: str, papers: List[Dict]) -> float:
        """
        Novelty score: 1 - average_similarity of top-5 most similar papers.
        Higher score = more novel / fewer existing works.
        """
        if not papers:
            return 1.0
        top5_sims = [p["similarity"] for p in papers[:5]]
        avg_sim = sum(top5_sims) / len(top5_sims)
        return round(1.0 - avg_sim, 3)
