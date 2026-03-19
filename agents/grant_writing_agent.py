"""
Grant Writing Agent
Generates structured, funding-ready grant proposals aligned with agency guidelines.
Supports NSF, NIH, DARPA, EU Horizon formats.
"""
import logging
import json
import re
from typing import Dict, Any, Optional

from core.llm_factory import get_llm, invoke_with_retry
from config.settings import GRANT_AGENCIES

logger = logging.getLogger(__name__)


class GrantWritingAgent:
    """
    Produces complete grant proposals in agency-specific formats.
    Takes research context from other agents and generates each section.
    """

    name = "Grant Writing Agent"
    description = (
        "Generates structured grant proposals (Problem → Method → Impact → Budget) "
        "aligned with NSF, NIH, DARPA, or EU Horizon guidelines."
    )

    def run(
        self,
        research_topic:    str,
        research_gap:      str,
        methodology:       Dict[str, Any],
        agency:            str = "NSF",
        pi_name:           str = "Principal Investigator",
        institution:       str = "Research Institution",
        budget_total:      str = "$500,000",
        duration_years:    int = 3,
        citation_style:    str = "IEEE",
        top_papers:        Optional[list] = None,
    ) -> Dict[str, Any]:

        # Look up known agency template; fall back to a generic structure for
        # any free-text agency the user types that isn't in GRANT_AGENCIES.
        _generic_cfg = {
            "sections": [
                "Executive Summary", "Problem Statement", "Proposed Approach",
                "Innovation & Significance", "Methodology", "Budget Justification",
                "Expected Outcomes", "References",
            ],
            "style": f"{agency} grant proposal guidelines",
        }
        agency_cfg = GRANT_AGENCIES.get(agency, _generic_cfg)
        sections   = agency_cfg["sections"]
        style_note = agency_cfg["style"]
        logger.info("[GrantWriting] Agency: %s | Topic: %s", agency, research_topic)

        generated_sections = {}
        for section in sections:
            content = self._write_section(
                section=section,
                research_topic=research_topic,
                research_gap=research_gap,
                methodology=methodology,
                agency=agency,
                style_note=style_note,
                pi_name=pi_name,
                institution=institution,
                budget_total=budget_total,
                duration_years=duration_years,
                top_papers=top_papers or [],
            )
            generated_sections[section] = content

        full_text = self._assemble_proposal(
            agency, pi_name, institution, research_topic,
            duration_years, budget_total, generated_sections, citation_style
        )

        return {
            "agent":          self.name,
            "agency":         agency,
            "research_topic": research_topic,
            "pi_name":        pi_name,
            "institution":    institution,
            "budget_total":   budget_total,
            "duration_years": duration_years,
            "sections":       generated_sections,
            "full_proposal":  full_text,
            "citation_style": citation_style,
        }

    # ── Section writer ────────────────────────────────────────

    def _write_section(self, section: str, **ctx) -> str:
        llm = get_llm(temperature=0.4)

        paper_refs = ""
        if ctx.get("top_papers"):
            paper_refs = "\n".join(
                f"  - {p.get('title','')} ({p.get('year','')})"
                for p in ctx["top_papers"][:6]
            )

        prompt = f"""You are an expert grant writer drafting the "{section}" section of a {ctx['agency']} proposal.

Agency format: {ctx['style_note']}
Research Topic: {ctx['research_topic']}
Research Gap: {ctx['research_gap']}
Proposed Methodology: {json.dumps(ctx.get('methodology', {}).get('approach', []))}
PI / Institution: {ctx['pi_name']} at {ctx['institution']}
Budget: {ctx['budget_total']} over {ctx['duration_years']} years

Key references from literature:
{paper_refs}

Write a compelling, detailed "{section}" section (300-500 words) following {ctx['agency']} guidelines.

Critical instructions:
- Match the academic register and methodology to the actual discipline of the research topic. If this is a humanities, social science, or interdisciplinary topic, use appropriate scholarly language — do NOT impose computational or ML framing.
- Be specific and grounded in the research topic. Do NOT use generic placeholder text.
- Use academic language appropriate to the field.
- Highlight the scholarly significance and originality of the work."""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as exc:
            logger.warning("[GrantWriting] Section '%s' failed: %s", section, exc)
            return f"[Section generation failed: {exc}]"

    # ── Assembler ─────────────────────────────────────────────

    def _assemble_proposal(
        self, agency, pi, institution, topic,
        duration, budget, sections, citation_style
    ) -> str:
        header = f"""
{'='*70}
GRANT PROPOSAL — {agency}
{'='*70}
Title: {topic}
Principal Investigator: {pi}
Institution: {institution}
Duration: {duration} Years
Total Budget: {budget}
Citation Style: {citation_style}
{'='*70}

"""
        body = ""
        for section, content in sections.items():
            body += f"\n## {section}\n\n{content}\n\n{'─'*60}\n"

        return header + body
