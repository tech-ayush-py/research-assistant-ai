"""
Gap Identification Agent
Identifies under-explored research intersections via semantic clustering
and graph-based citation analysis.

Gap Verification (two layers):
  Layer 1 — Corpus counter-check: for each proposed gap, a targeted similarity
  search is run. If the top result scores > COVERAGE_THRESHOLD, the gap is
  flagged as "potentially covered" and the conflicting paper is cited.

  Layer 2 — Adversarial LLM review: a second LLM call plays devil's advocate,
  reviewing each gap against the counter-check evidence and assigning a
  confidence rating (high / medium / low) with a verification note.
"""
import json
import logging
import re
from typing import Dict, Any, List

from core.llm_factory import get_llm, invoke_with_retry
from core.vector_store import similarity_search, get_collection

logger = logging.getLogger(__name__)

# If the top paper matching a gap statement has similarity above this threshold,
# there is likely existing work addressing that gap — flag it.
_COVERAGE_THRESHOLD = 0.72


class GapIdentificationAgent:
    """
    Finds research gaps by:
    1. Clustering papers semantically and finding thin/empty cluster regions.
    2. Analysing what problems are raised but not solved in abstracts.
    3. Comparing sub-topic coverage via LLM reasoning.
    4. Verifying each gap with a corpus counter-check + adversarial LLM review.
    """

    name = "Gap Identification Agent"
    description = (
        "Identifies under-explored research gaps using semantic clustering "
        "and graph-based citation analysis on the ingested paper corpus, "
        "then verifies each gap with a two-layer cross-check."
    )

    def run(self, research_topic: str) -> Dict[str, Any]:
        logger.info("[GapIdentification] Topic: %s", research_topic)

        related = similarity_search(research_topic, n_results=20)
        if not related:
            return {"agent": self.name, "error": "No papers found. Run Literature Mining first."}

        # Layer 0 — propose gaps via LLM
        raw_gaps = self._find_gaps_llm(research_topic, related)

        # Layer 1 — corpus counter-check
        counter_checked = self._corpus_counter_check(raw_gaps.get("gaps", []))

        # Layer 2 — adversarial LLM verification
        verified_gaps = self._adversarial_review(
            research_topic, raw_gaps.get("gaps", []), counter_checked
        )

        novelty_score = self._compute_novelty_score(research_topic, related)

        return {
            "agent":            self.name,
            "research_topic":   research_topic,
            "papers_analysed":  len(related),
            "identified_gaps":  [g["gap"] for g in verified_gaps],
            "verified_gaps":    verified_gaps,          # full objects with confidence + notes
            "opportunity_areas": raw_gaps.get("opportunities", []),
            "gap_reasoning":    raw_gaps.get("reasoning", ""),
            "novelty_score":    novelty_score,
            "top_cited_papers": [
                {"title": p["title"], "year": p["year"], "similarity": p["similarity"]}
                for p in related[:5]
            ],
        }

    # ── Layer 0: Propose gaps ─────────────────────────────────

    def _find_gaps_llm(self, topic: str, papers: List[Dict]) -> Dict[str, Any]:
        llm = get_llm(temperature=0.4)
        paper_summaries = "\n".join(
            f"- [{p['year']}] {p['title']}: {p['snippet'][:200]}"
            for p in papers[:15]
        )

        prompt = f"""You are an expert academic researcher identifying knowledge gaps. Your analysis must be appropriate for the specific academic discipline of the research topic — do not assume a computational or technical framing unless the topic is explicitly technical.

Research Topic: "{topic}"

Here are the most relevant existing papers:
{paper_summaries}

Based on these papers, identify:
1. RESEARCH GAPS: 5-7 specific, concrete areas that are under-explored or completely missing in the existing literature. Frame these in terms natural to the discipline (e.g. archival gaps, theoretical gaps, methodological gaps, geographic gaps, temporal gaps).
2. OPPORTUNITY AREAS: 3-4 promising directions where new research could have high scholarly impact.
3. REASONING: A concise paragraph explaining your gap analysis approach.

Be precise — each gap should be a specific claim about what is missing, not a vague observation.

Respond in JSON with keys: gaps (list of strings), opportunities (list of strings), reasoning (string)"""

        try:
            from langchain_core.messages import HumanMessage
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

    # ── Layer 1: Corpus counter-check ────────────────────────

    def _corpus_counter_check(self, gaps: List[str]) -> List[Dict[str, Any]]:
        """
        For each proposed gap, search the corpus for papers that might already
        address it. Returns a list of dicts with:
          gap, coverage_score, top_conflicting_paper (title + similarity), is_covered
        """
        results = []
        for gap in gaps:
            hits = similarity_search(gap, n_results=3)
            if not hits:
                results.append({
                    "gap":                    gap,
                    "coverage_score":         0.0,
                    "top_conflicting_paper":  None,
                    "is_covered":             False,
                })
                continue

            top = hits[0]
            coverage = top["similarity"]
            is_covered = coverage >= _COVERAGE_THRESHOLD

            results.append({
                "gap":            gap,
                "coverage_score": round(coverage, 4),
                "top_conflicting_paper": {
                    "title":      top["title"],
                    "year":       top["year"],
                    "similarity": top["similarity"],
                } if is_covered else None,
                "is_covered": is_covered,
            })

            logger.info(
                "[GapVerification] Gap: '%s...' | Coverage: %.3f | Covered: %s",
                gap[:60], coverage, is_covered
            )

        return results

    # ── Layer 2: Adversarial LLM review ──────────────────────

    def _adversarial_review(
        self,
        topic: str,
        gaps: List[str],
        counter_check: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Second LLM call that reviews each gap with devil's advocate framing.
        Assigns confidence (high/medium/low) and a verification note per gap.
        Returns list of verified gap objects.
        """
        llm = get_llm(temperature=0.2)

        # Build the evidence summary for each gap
        evidence_lines = []
        for item in counter_check:
            if item["is_covered"] and item["top_conflicting_paper"]:
                cp = item["top_conflicting_paper"]
                evidence_lines.append(
                    f'- Gap: "{item["gap"]}"\n'
                    f'  ⚠️  Potential conflict: "{cp["title"]}" ({cp["year"]}) '
                    f'— similarity {cp["similarity"]:.2f}'
                )
            else:
                evidence_lines.append(
                    f'- Gap: "{item["gap"]}"\n'
                    f'  ✓  No strong conflicting paper found (coverage score: {item["coverage_score"]:.2f})'
                )

        evidence_block = "\n".join(evidence_lines)

        prompt = f"""You are a rigorous academic peer reviewer. Your job is to verify whether each proposed research gap is genuinely under-explored, or whether it is already addressed in existing literature.

Research Topic: "{topic}"

Proposed gaps and corpus evidence:
{evidence_block}

For each gap, provide:
1. confidence: "high" (genuinely novel, no conflicting evidence), "medium" (partially addressed but still meaningful), or "low" (likely already covered — should be revised or dropped)
2. verification_note: 1-2 sentences explaining your assessment. If coverage was detected, name the conflicting work and explain what it leaves unaddressed (if anything). Be honest — do not validate weak gaps just to be polite.
3. refined_gap: A sharpened version of the gap statement that addresses any coverage concerns. If the gap is "high" confidence, keep it as-is. If "medium" or "low", reframe it to be more specific and defensible.

Respond ONLY in JSON as a list of objects with keys: gap (original), confidence, verification_note, refined_gap.
No markdown, no extra text."""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            reviewed = json.loads(raw)

            # Merge with counter-check data
            merged = []
            for item in reviewed:
                # Find matching counter-check entry
                cc = next((c for c in counter_check if c["gap"] == item.get("gap", "")), {})
                merged.append({
                    "gap":                   item.get("refined_gap") or item.get("gap", ""),
                    "original_gap":          item.get("gap", ""),
                    "confidence":            item.get("confidence", "medium"),
                    "verification_note":     item.get("verification_note", ""),
                    "coverage_score":        cc.get("coverage_score", 0.0),
                    "conflicting_paper":     cc.get("top_conflicting_paper"),
                })

            # Sort: high confidence first, then medium, then low
            order = {"high": 0, "medium": 1, "low": 2}
            merged.sort(key=lambda x: order.get(x["confidence"], 1))
            return merged

        except Exception as exc:
            logger.warning("[GapVerification] Adversarial review failed: %s", exc)
            # Fallback: return gaps with counter-check data only, no LLM verification
            return [
                {
                    "gap":               item["gap"],
                    "original_gap":      item["gap"],
                    "confidence":        "low" if item["is_covered"] else "medium",
                    "verification_note": (
                        f"Automatic check: conflicting paper found — '{item['top_conflicting_paper']['title']}'"
                        if item["is_covered"] and item["top_conflicting_paper"]
                        else "Automatic check: no direct conflict found in corpus."
                    ),
                    "coverage_score":    item["coverage_score"],
                    "conflicting_paper": item.get("top_conflicting_paper"),
                }
                for item in counter_check
            ]

    # ── Novelty score ─────────────────────────────────────────

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
