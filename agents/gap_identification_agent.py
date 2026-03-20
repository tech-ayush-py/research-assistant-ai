"""
Gap Identification Agent — v2 with Two-Layer Verification

Layer 1: LLM proposes 5-7 gaps from top papers (as before).
Layer 2a: Each gap is embedded and searched against the full corpus.
          If similarity > 0.72 a conflicting paper is flagged.
Layer 2b: A second adversarial LLM pass rates each gap high/medium/low
          and refines the wording — naming conflicting papers where found.

Gaps are sorted high → medium → low so the most reliable appear first.
"""
import json
import logging
import re
from typing import Dict, Any, List

from core.llm_factory import get_llm, invoke_with_retry
from core.vector_store import similarity_search, get_collection

logger = logging.getLogger(__name__)

_COVER_THRESHOLD = 0.72   # similarity above this → gap may already be covered


class GapIdentificationAgent:

    name = "Gap Identification Agent"
    description = (
        "Identifies research gaps via two-layer verification: "
        "LLM proposal + corpus counter-check + adversarial LLM review."
    )

    def run(self, research_topic: str) -> Dict[str, Any]:
        logger.info("[GapIdentification] Topic: %s", research_topic)

        related = similarity_search(research_topic, n_results=20)
        if not related:
            return {"agent": self.name, "error": "No papers found. Run Literature Mining first."}

        # Layer 1 — initial gap proposal
        raw_gaps = self._propose_gaps(research_topic, related)

        # Layer 2a — corpus counter-check (algorithmic, no LLM cost)
        checked  = self._counter_check(raw_gaps["gaps"])

        # Layer 2b — adversarial LLM review
        verified = self._adversarial_review(research_topic, checked)

        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        verified.sort(key=lambda g: order.get(g.get("confidence", "medium"), 1))

        novelty_score = self._compute_novelty_score(related)

        return {
            "agent":            self.name,
            "research_topic":   research_topic,
            "papers_analysed":  len(related),
            "identified_gaps":  [g["refined_gap"] for g in verified],
            "gap_details":      verified,          # full detail for UI
            "opportunity_areas": raw_gaps.get("opportunities", []),
            "gap_reasoning":    raw_gaps.get("reasoning", ""),
            "novelty_score":    novelty_score,
            "top_cited_papers": [
                {"title": p["title"], "year": p["year"], "similarity": p["similarity"]}
                for p in related[:5]
            ],
        }

    # ── Layer 1: LLM gap proposal ──────────────────────────────────────────────

    def _propose_gaps(self, topic: str, papers: List[Dict]) -> Dict[str, Any]:
        llm = get_llm(temperature=0.4)
        paper_summaries = "\n".join(
            f"- [{p['year']}] {p['title']}: {p['snippet'][:200]}"
            for p in papers[:15]
        )
        prompt = f"""You are an expert academic researcher identifying knowledge gaps. Your analysis must suit the specific discipline — do not default to computational or ML framing unless the topic is explicitly technical.

Research Topic: "{topic}"

Existing papers:
{paper_summaries}

Identify:
1. RESEARCH GAPS: 5-7 specific gaps (archival, theoretical, methodological, geographic, temporal — as appropriate for the discipline).
2. OPPORTUNITY AREAS: 3-4 high-impact directions.
3. REASONING: A concise paragraph on your gap analysis approach.

Respond only in JSON with keys: gaps (list of strings), opportunities (list of strings), reasoning (string)"""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[GapIdentification] Layer1 LLM error: %s", exc)
            return {"gaps": ["LLM unavailable"], "opportunities": [], "reasoning": str(exc)}

    # ── Layer 2a: Corpus counter-check ────────────────────────────────────────

    def _counter_check(self, gaps: List[str]) -> List[Dict]:
        """
        For each gap, embed and search the corpus.
        If the top hit > _COVER_THRESHOLD, flag it as potentially covered.
        """
        checked = []
        for gap_text in gaps:
            hits = similarity_search(gap_text, n_results=3)
            conflict = None
            is_covered = False
            if hits and hits[0]["similarity"] >= _COVER_THRESHOLD:
                is_covered = True
                conflict = {
                    "title":      hits[0]["title"],
                    "year":       hits[0]["year"],
                    "similarity": hits[0]["similarity"],
                }
            checked.append({
                "original_gap": gap_text,
                "is_covered":   is_covered,
                "conflict":     conflict,
            })
        return checked

    # ── Layer 2b: Adversarial LLM review ─────────────────────────────────────

    def _adversarial_review(self, topic: str, checked: List[Dict]) -> List[Dict]:
        """
        Second LLM call at low temperature (0.2) to play devil's advocate.
        Assigns confidence and sharpens each gap statement.
        """
        llm = get_llm(temperature=0.2)

        gap_block = ""
        for i, g in enumerate(checked, 1):
            gap_block += f"\nGap {i}: {g['original_gap']}\n"
            if g["is_covered"]:
                gap_block += (
                    f"  ⚠ Counter-evidence: '{g['conflict']['title']}' "
                    f"({g['conflict']['year']}) has similarity {g['conflict']['similarity']:.2f} "
                    f"to this gap statement — it may already address this.\n"
                )
            else:
                gap_block += "  ✓ No strong counter-evidence found in corpus.\n"

        prompt = f"""You are a rigorous academic peer reviewer evaluating proposed research gaps for the topic: "{topic}".

For each gap below, you are given whether a conflicting paper was found in the literature corpus.

Play devil's advocate. For each gap assign:
- confidence: "high" (genuinely novel, well-supported), "medium" (partially addressed but still worth pursuing with a narrower claim), or "low" (likely already covered, needs major revision)
- verification_note: 1-2 sentences. If conflict found, name the paper and explain what it still leaves unaddressed. If no conflict, briefly justify the high confidence.
- refined_gap: A sharpened version of the gap statement. For medium/low, narrow the claim. For high, keep or lightly polish the original.

{gap_block}

Respond ONLY with a valid JSON array (no markdown fences), one object per gap:
[
  {{
    "confidence": "high"|"medium"|"low",
    "verification_note": "...",
    "refined_gap": "..."
  }},
  ...
]"""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            reviewed = json.loads(raw)
            # Merge with counter-check data
            result = []
            for i, r in enumerate(reviewed):
                base = checked[i] if i < len(checked) else {}
                result.append({
                    "original_gap":     base.get("original_gap", ""),
                    "refined_gap":      r.get("refined_gap", base.get("original_gap", "")),
                    "confidence":       r.get("confidence", "medium"),
                    "verification_note": r.get("verification_note", ""),
                    "is_covered":       base.get("is_covered", False),
                    "conflict":         base.get("conflict"),
                })
            return result
        except Exception as exc:
            logger.warning("[GapIdentification] Layer2 LLM error: %s", exc)
            # Fallback: return checked gaps with neutral confidence
            return [
                {
                    "original_gap":     g["original_gap"],
                    "refined_gap":      g["original_gap"],
                    "confidence":       "medium",
                    "verification_note": "Verification unavailable.",
                    "is_covered":       g["is_covered"],
                    "conflict":         g["conflict"],
                }
                for g in checked
            ]

    # ── Novelty score ──────────────────────────────────────────────────────────

    def _compute_novelty_score(self, papers: List[Dict]) -> float:
        if not papers:
            return 1.0
        top5 = [p["similarity"] for p in papers[:5]]
        return round(1.0 - sum(top5) / len(top5), 3)
