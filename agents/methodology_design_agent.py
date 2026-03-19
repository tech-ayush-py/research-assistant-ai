"""
Methodology Design Agent
Suggests research methodology, sources, evaluation criteria, and experimental design.
Generalised to cover ALL academic disciplines — not just ML/CS.
"""
import logging
import json
import re
from typing import Dict, Any

from core.llm_factory import get_llm, invoke_with_retry

logger = logging.getLogger(__name__)

# ── Broad discipline classifier ────────────────────────────────────────────────
# Maps domain keywords to discipline families so we can pick appropriate
# data sources and evaluation criteria without defaulting to ML benchmarks.

_DISCIPLINE_MAP = {
    # Computational / Engineering
    "nlp":                  "computational",
    "natural language":     "computational",
    "computer vision":      "computational",
    "machine learning":     "computational",
    "deep learning":        "computational",
    "general ai":           "computational",
    "data science":         "computational",
    "robotics":             "computational",
    "reinforcement":        "computational",
    "multimodal":           "computational",
    "graph":                "computational",
    "network":              "computational",
    "cybersecurity":        "computational",
    "security":             "computational",
    # Life & Health Sciences
    "biomedical":           "life_science",
    "medicine":             "life_science",
    "biology":              "life_science",
    "health":               "life_science",
    "clinical":             "life_science",
    "genomics":             "life_science",
    "neuroscience":         "life_science",
    "environmental":        "life_science",
    "physics":              "physical_science",
    "chemistry":            "physical_science",
    "materials":            "physical_science",
    # Social Sciences
    "social":               "social_science",
    "economics":            "social_science",
    "psychology":           "social_science",
    "political":            "social_science",
    "sociology":            "social_science",
    "education":            "social_science",
    "public policy":        "social_science",
    "public health":        "social_science",
    "social work":          "social_science",
    # Humanities
    "history":              "humanities",
    "literature":           "humanities",
    "linguistics":          "humanities",
    "philosophy":           "humanities",
    "cultural":             "humanities",
    "gender":               "humanities",
    "anthropology":         "humanities",
    "art history":          "humanities",
    "archaeology":          "humanities",
    "art":                  "humanities",
    "film":                 "humanities",
    "law":                  "humanities",
    "legal":                "humanities",
    "media":                "humanities",
    "communication":        "humanities",
    "espionage":            "humanities",
    "intelligence":         "humanities",
    "war":                  "humanities",
    "colonial":             "humanities",
    "resistance":           "humanities",
    "women":                "humanities",
    "gender studies":       "humanities",
    # Business / Management
    "business":             "business",
    "management":           "business",
    "finance":              "business",
    "marketing":            "business",
    "entrepreneurship":     "business",
}

# ── Data source registries by discipline ──────────────────────────────────────
_DATA_SOURCES = {
    "computational": [
        "Benchmark datasets (task-specific)",
        "GitHub open-source repositories",
        "Hugging Face Datasets Hub",
        "UCI ML Repository",
        "Kaggle Datasets",
        "OpenML",
    ],
    "life_science": [
        "PubMed / MEDLINE",
        "MIMIC-III / MIMIC-IV clinical records",
        "UK Biobank",
        "ClinicalTrials.gov",
        "Gene Expression Omnibus (GEO)",
        "UniProt / PDB",
    ],
    "social_science": [
        "Survey microdata (e.g. DHS, IPUMS)",
        "World Bank Open Data",
        "OECD Statistics",
        "National administrative records",
        "Interview / focus group transcripts",
        "Social media archives",
    ],
    "humanities": [
        "Archival primary sources (national / institutional archives)",
        "Oral history collections",
        "Digitised newspaper corpora",
        "Government declassified documents",
        "Personal correspondence and diaries",
        "Published memoirs and autobiographies",
        "Museum and library special collections",
    ],
    "physical_science": [
        "Experimental lab measurements",
        "Simulation / modelling outputs",
        "Public repositories (e.g. ICSD, CSD, NIST)",
        "Remote sensing / satellite data",
        "Observational field data",
    ],
    "business": [
        "Company financial reports (SEC filings, annual reports)",
        "Industry databases (Bloomberg, Statista)",
        "Consumer survey data",
        "Case study archives",
        "Patent databases (USPTO, EPO)",
    ],
}

# ── Evaluation criteria by discipline ─────────────────────────────────────────
_EVAL_CRITERIA = {
    "computational": [
        "Accuracy / F1-Score / AUC-ROC",
        "Precision / Recall",
        "BLEU / ROUGE / BERTScore (generation tasks)",
        "Latency and computational efficiency",
        "Ablation study results",
    ],
    "life_science": [
        "Statistical significance (p-value, confidence intervals)",
        "Effect size (Cohen's d, odds ratio)",
        "Sensitivity / Specificity",
        "Reproducibility across cohorts",
        "Clinical relevance / translational impact",
    ],
    "social_science": [
        "Internal and external validity",
        "Statistical significance and effect size",
        "Reliability (inter-rater, test-retest)",
        "Triangulation across data sources",
        "Generalisability of findings",
    ],
    "humanities": [
        "Archival completeness and source diversity",
        "Interpretive rigour and theoretical grounding",
        "Peer review and scholarly consensus",
        "Contribution to historiography / canon",
        "Methodological transparency (positionality, reflexivity)",
    ],
    "physical_science": [
        "Measurement uncertainty and error margins",
        "Reproducibility and replication",
        "Statistical significance",
        "Model fit (R², RMSE)",
        "Comparison with established theoretical predictions",
    ],
    "business": [
        "Internal validity of causal claims",
        "Generalisability across industries / contexts",
        "Economic / practical significance",
        "Robustness checks",
        "Stakeholder impact assessment",
    ],
}


def _classify_discipline(domain: str, topic: str = "") -> str:
    """
    Return the broad discipline family for a given domain string and topic.
    Matches the longest keyword first so specific terms (e.g. 'art history')
    beat shorter ambiguous ones (e.g. 'art' or 'social').
    """
    combined = (domain + " " + topic).lower()
    # Sort by keyword length descending so more specific terms win
    for keyword, discipline in sorted(_DISCIPLINE_MAP.items(), key=lambda x: -len(x[0])):
        if keyword in combined:
            return discipline
    return "general"


class MethodologyDesignAgent:
    """
    Recommends research methodology, data sources, evaluation criteria,
    and a study design appropriate for ANY academic discipline.
    Uses LLM for the intellectual design work; registries only as
    discipline-appropriate starting points — not as hard-coded ML defaults.
    """

    name = "Methodology Design Agent"
    description = (
        "Designs a rigorous research methodology suited to the discipline — "
        "covering humanities, social sciences, life sciences, and STEM equally."
    )

    def run(self, research_gap: str, domain: str = "Other") -> Dict[str, Any]:
        logger.info("[MethodologyDesign] Gap: %s | Domain: %s", research_gap, domain)

        discipline  = _classify_discipline(domain, research_gap)
        llm_design  = self._generate_methodology(research_gap, domain, discipline)
        data_sources = self._suggest_data_sources(discipline)
        eval_criteria = self._suggest_eval_criteria(discipline)

        return {
            "agent":               self.name,
            "research_gap":        research_gap,
            "domain":              domain,
            "discipline_family":   discipline,
            "hypothesis":          llm_design.get("hypothesis", ""),
            "approach":            llm_design.get("approach", []),
            "baselines":           llm_design.get("comparators", []),
            "suggested_datasets":  data_sources,
            "evaluation_metrics":  eval_criteria,
            "timeline_weeks":      llm_design.get("timeline", []),
            "expected_outcomes":   llm_design.get("outcomes", []),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_methodology(self, gap: str, domain: str, discipline: str) -> Dict[str, Any]:
        llm = get_llm(temperature=0.35)

        # Discipline-aware framing hints so the LLM doesn't default to ML language
        framing_hints = {
            "computational":  "Use quantitative / experimental framing. Mention model architectures, benchmarks, and ablation studies where relevant.",
            "life_science":   "Use clinical or biological study design. Consider RCTs, cohort studies, or lab experiments as appropriate.",
            "social_science": "Use mixed-methods or survey-based framing. Consider interviews, questionnaires, econometric models, or ethnography.",
            "humanities":     "Use archival, interpretive, or historiographical framing. Do NOT suggest machine learning models or computational benchmarks unless explicitly relevant. Focus on primary sources, oral histories, and theoretical frameworks.",
            "physical_science": "Use experimental or modelling framing. Consider lab experiments, simulations, or observational studies.",
            "business":       "Use case-study, survey, or econometric framing. Consider qualitative and quantitative mixed methods.",
            "general":        "Use the most appropriate methodology for the discipline implied by the research gap. Do not default to machine learning or computational methods unless the topic is explicitly technical.",
        }
        hint = framing_hints.get(discipline, framing_hints["general"])

        prompt = f"""You are a senior academic research methodologist. Design a rigorous methodology for the following research gap.

Research Gap: "{gap}"
Academic Domain: {domain}
Discipline Family: {discipline}
Methodological Framing: {hint}

Produce a methodology appropriate for this SPECIFIC discipline. Do NOT default to machine learning, neural networks, or computational benchmarks unless the domain explicitly requires them.

Return ONLY valid JSON with these exact keys:
{{
  "hypothesis": "A clear, testable research hypothesis or central research question (1-2 sentences)",
  "approach": ["Step 1 ...", "Step 2 ...", "Step 3 ...", "Step 4 ...", "Step 5 ..."],
  "comparators": ["Existing study / framework / theory 1 to compare against", "..."],
  "timeline": ["Phase 1 (weeks 1-4): ...", "Phase 2 (weeks 5-8): ...", "Phase 3 (weeks 9-12): ..."],
  "outcomes": ["Expected contribution 1", "Expected contribution 2", "Expected contribution 3"]
}}

- approach: 5-7 concrete, discipline-appropriate steps
- comparators: 3-5 existing works, frameworks, or theories this study will benchmark or dialogue with (NOT ML model names for non-technical fields)
- timeline: 3-5 research phases with week ranges
- outcomes: 3-4 scholarly contributions this research will make

Respond with ONLY the JSON object. No preamble, no markdown fences."""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[MethodologyDesign] LLM error: %s", exc)
            return {"hypothesis": "", "approach": [], "comparators": [], "timeline": [], "outcomes": []}

    def _suggest_data_sources(self, discipline: str) -> list:
        """Return discipline-appropriate data sources — empty list for fields that don't use datasets."""
        return _DATA_SOURCES.get(discipline, [
            "Primary sources relevant to the research topic",
            "Peer-reviewed literature",
            "Institutional records or archives",
        ])

    def _suggest_eval_criteria(self, discipline: str) -> list:
        """Return discipline-appropriate evaluation criteria."""
        return _EVAL_CRITERIA.get(discipline, [
            "Internal validity of findings",
            "Methodological rigour",
            "Contribution to existing scholarship",
            "Reproducibility / verifiability",
        ])
