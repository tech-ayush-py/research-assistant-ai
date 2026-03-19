"""
Research Orchestrator
Coordinates all six agents in sequence using CrewAI-style task chaining.
Each agent's output feeds the next.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from agents.literature_mining_agent   import LiteratureMiningAgent
from agents.trend_analysis_agent      import TrendAnalysisAgent
from agents.gap_identification_agent  import GapIdentificationAgent
from agents.methodology_design_agent  import MethodologyDesignAgent
from agents.grant_writing_agent       import GrantWritingAgent
from agents.novelty_scoring_agent     import NoveltyScoringAgent

logger = logging.getLogger(__name__)


@dataclass
class ResearchRequest:
    topic:            str
    domain:           str               = "General AI"
    grant_agency:     str               = "NSF"
    pi_name:          str               = "Principal Investigator"
    institution:      str               = "Research Institution"
    budget_total:     str               = "$500,000"
    duration_years:   int               = 3
    citation_style:   str               = "IEEE"
    max_papers:       int               = 30
    custom_gap:       Optional[str]     = None     # override gap from agent


@dataclass
class ResearchReport:
    request:        ResearchRequest
    literature:     Dict[str, Any]      = field(default_factory=dict)
    trends:         Dict[str, Any]      = field(default_factory=dict)
    gaps:           Dict[str, Any]      = field(default_factory=dict)
    methodology:    Dict[str, Any]      = field(default_factory=dict)
    grant:          Dict[str, Any]      = field(default_factory=dict)
    novelty:        Dict[str, Any]      = field(default_factory=dict)
    errors:         list                = field(default_factory=list)


class ResearchOrchestrator:
    """
    Pipeline:
    1. LiteratureMiningAgent  → fetch & embed papers
    2. TrendAnalysisAgent     → detect field evolution
    3. GapIdentificationAgent → find research gaps
    4. MethodologyDesignAgent → design experiment
    5. GrantWritingAgent      → generate proposal
    6. NoveltyScoringAgent    → score novelty
    """

    def __init__(self):
        self.agents = {
            "literature": LiteratureMiningAgent(),
            "trends":     TrendAnalysisAgent(),
            "gaps":       GapIdentificationAgent(),
            "methodology": MethodologyDesignAgent(),
            "grant":      GrantWritingAgent(),
            "novelty":    NoveltyScoringAgent(),
        }

    def run(self, request: ResearchRequest, progress_callback=None) -> ResearchReport:
        report = ResearchReport(request=request)

        def _cb(step, msg):
            logger.info("[Orchestrator] %s: %s", step, msg)
            if progress_callback:
                progress_callback(step, msg)

        # ── Step 1: Literature Mining ──────────────────────────
        _cb("literature", f"Fetching papers for '{request.topic}'…")
        try:
            report.literature = self.agents["literature"].run(
                query=request.topic, max_papers=request.max_papers
            )
        except Exception as e:
            report.errors.append(f"Literature Mining: {e}")
            report.literature = {"error": str(e)}

        # ── Step 2: Trend Analysis ─────────────────────────────
        _cb("trends", "Analysing research trends…")
        try:
            report.trends = self.agents["trends"].run(topic=request.topic)
        except Exception as e:
            report.errors.append(f"Trend Analysis: {e}")
            report.trends = {"error": str(e)}

        # ── Step 3: Gap Identification ─────────────────────────
        _cb("gaps", "Identifying research gaps…")
        try:
            report.gaps = self.agents["gaps"].run(research_topic=request.topic)
        except Exception as e:
            report.errors.append(f"Gap Identification: {e}")
            report.gaps = {"error": str(e)}

        # ── Step 4: Methodology Design ─────────────────────────
        primary_gap = (
            request.custom_gap
            or (report.gaps.get("identified_gaps") or [""])[0]
            or request.topic
        )
        _cb("methodology", f"Designing methodology for gap: {primary_gap[:80]}…")
        try:
            report.methodology = self.agents["methodology"].run(
                research_gap=primary_gap, domain=request.domain
            )
        except Exception as e:
            report.errors.append(f"Methodology Design: {e}")
            report.methodology = {"error": str(e)}

        # ── Step 5: Grant Writing ──────────────────────────────
        _cb("grant", f"Generating {request.grant_agency} grant proposal…")
        try:
            report.grant = self.agents["grant"].run(
                research_topic=request.topic,
                research_gap=primary_gap,
                methodology=report.methodology,
                agency=request.grant_agency,
                pi_name=request.pi_name,
                institution=request.institution,
                budget_total=request.budget_total,
                duration_years=request.duration_years,
                citation_style=request.citation_style,
                top_papers=report.literature.get("top_papers", []),
            )
        except Exception as e:
            report.errors.append(f"Grant Writing: {e}")
            report.grant = {"error": str(e)}

        # ── Step 6: Novelty Scoring ────────────────────────────
        _cb("novelty", "Scoring proposal novelty…")
        try:
            proposal_text = (
                report.grant.get("sections", {}).get("Project Summary", "")
                or report.grant.get("sections", {}).get("Project Description", "")
                or request.topic
            )
            report.novelty = self.agents["novelty"].run(
                proposal_abstract=proposal_text[:1500]
            )
        except Exception as e:
            report.errors.append(f"Novelty Scoring: {e}")
            report.novelty = {"error": str(e)}

        _cb("done", "Pipeline complete ✅")
        return report
