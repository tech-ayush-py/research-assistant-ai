"""
Methodology Design Agent
Suggests datasets, baselines, evaluation metrics, and experimental design.
"""
import logging
import json
import re
from typing import Dict, Any

from core.llm_factory import get_llm

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "NLP":           ["SQuAD 2.0", "GLUE", "SuperGLUE", "CommonCrawl", "C4"],
    "Computer Vision": ["ImageNet", "COCO", "CIFAR-10/100", "Open Images"],
    "Biomedical":    ["PubMed", "MIMIC-III", "UK Biobank", "OpenFDA"],
    "Graph/Network": ["OGB (Open Graph Benchmark)", "SNAP", "Cora", "CiteSeer"],
    "Multimodal":    ["CC3M", "LAION-5B", "MS-COCO Captions"],
    "Tabular":       ["UCI ML Repository", "Kaggle Datasets", "OpenML"],
    "RL":            ["OpenAI Gym", "MuJoCo", "D4RL"],
}

METRIC_REGISTRY = {
    "Classification": ["Accuracy", "F1-Score", "AUC-ROC", "Precision/Recall"],
    "Generation":     ["BLEU", "ROUGE", "BERTScore", "METEOR", "Perplexity"],
    "Retrieval":      ["MRR", "NDCG", "MAP", "Hit@k"],
    "Regression":     ["RMSE", "MAE", "R²", "MAPE"],
    "Clustering":     ["Silhouette Score", "ARI", "NMI", "Davies-Bouldin"],
}


class MethodologyDesignAgent:
    """
    Recommends experimental design, datasets, baselines,
    and evaluation metrics for a given research gap.
    """

    name = "Methodology Design Agent"
    description = (
        "Suggests datasets, evaluation metrics, baseline models, "
        "and experimental design for a given research hypothesis."
    )

    def run(self, research_gap: str, domain: str = "General AI") -> Dict[str, Any]:
        logger.info("[MethodologyDesign] Gap: %s | Domain: %s", research_gap, domain)

        llm_design = self._generate_methodology(research_gap, domain)
        datasets   = self._suggest_datasets(domain)
        metrics    = self._suggest_metrics(domain)

        return {
            "agent":          self.name,
            "research_gap":   research_gap,
            "domain":         domain,
            "hypothesis":     llm_design.get("hypothesis", ""),
            "approach":       llm_design.get("approach", []),
            "baselines":      llm_design.get("baselines", []),
            "suggested_datasets": datasets,
            "evaluation_metrics": metrics,
            "timeline_weeks":  llm_design.get("timeline", []),
            "expected_outcomes": llm_design.get("outcomes", []),
        }

    # ── Helpers ───────────────────────────────────────────────

    def _generate_methodology(self, gap: str, domain: str) -> Dict[str, Any]:
        llm = get_llm(temperature=0.35)
        prompt = f"""You are a senior research methodologist designing an experiment.

Research Gap / Problem: "{gap}"
Domain: {domain}

Design a rigorous research methodology including:
1. hypothesis: A clear, testable hypothesis statement
2. approach: Step-by-step experimental approach (5-7 steps)
3. baselines: 3-5 state-of-the-art baseline models to compare against
4. timeline: Suggested timeline in weeks for each phase
5. outcomes: 3-4 expected outcomes / contributions

Respond only in JSON with those exact keys. Lists of strings for approach, baselines, timeline, outcomes."""

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[MethodologyDesign] LLM error: %s", exc)
            return {"hypothesis": "", "approach": [], "baselines": [], "timeline": [], "outcomes": []}

    def _suggest_datasets(self, domain: str) -> list:
        for key in DATASET_REGISTRY:
            if key.lower() in domain.lower():
                return DATASET_REGISTRY[key]
        return DATASET_REGISTRY["Tabular"] + DATASET_REGISTRY["NLP"]

    def _suggest_metrics(self, domain: str) -> list:
        metrics = []
        for key, vals in METRIC_REGISTRY.items():
            if key.lower() in domain.lower() or domain.lower() in key.lower():
                metrics.extend(vals)
        return metrics or METRIC_REGISTRY["Classification"] + METRIC_REGISTRY["Generation"]
