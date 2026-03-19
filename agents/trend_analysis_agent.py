"""
Trend Analysis Agent
Detects how research has evolved using TF-IDF + sklearn (no torch/bertopic needed).
LLM summarises detected clusters into human-readable trends.

Discipline-aware: accepts an optional `domain` parameter that is forwarded to
the LLM prompt so the model frames trends appropriately for the field (archival,
theoretical, methodological, etc.) rather than defaulting to ML/CS language.
The keyword stopword list is also extended with CS/ML jargon so those terms
cannot surface as "trending" keywords in non-technical corpora.
"""
import logging
import json
import re
from collections import Counter
from typing import Dict, Any, List

from core.llm_factory import get_llm, invoke_with_retry
from core.vector_store import get_collection

logger = logging.getLogger(__name__)

# Generic stopwords — words that add no signal in any discipline
_GENERIC_STOPWORDS = {
    "the","a","of","in","and","to","for","with","on","is","are","this",
    "that","we","an","our","by","from","based","using","paper","method",
    "propose","show","model","data","result","approach","study","which","also",
    "their","these","those","have","been","were","they","them","its","also",
    "while","when","such","both","than","more","most","other","each","first",
    "between","through","after","before","about","would","could","should",
}

# CS/ML jargon stopwords — suppress these in non-technical corpora so they
# don't appear as "keywords" when the topic is history, literature, etc.
_CS_JARGON_STOPWORDS = {
    "neural","transformer","llm","bert","gpt","attention","embedding","layer",
    "gradient","token","batch","epoch","dataset","benchmark","baseline","accuracy",
    "precision","recall","finetune","pretraining","inference","latency","throughput",
    "convolutional","encoder","decoder","softmax","backprop","overfitting",
    "regularization","dropout","pytorch","tensorflow","cuda","gpu","multimodal",
    "vision","detection","segmentation","generation","diffusion","adversarial",
    "federated","reinforcement","reward","policy","agent","swarm","robotics",
}

# Combined set used for keyword extraction
_ALL_STOPWORDS = _GENERIC_STOPWORDS | _CS_JARGON_STOPWORDS

# Discipline framing hints keyed by domain dropdown value
_DISCIPLINE_FRAMING = {
    # Technical
    "General AI":             "computational and AI research",
    "NLP":                    "natural language processing research",
    "Computer Vision":        "computer vision research",
    "Biomedical":             "biomedical and life-science research",
    "Graph / Network":        "graph and network science research",
    "Multimodal":             "multimodal AI research",
    "Reinforcement Learning": "reinforcement learning research",
    "Robotics":               "robotics research",
    "Security":               "cybersecurity research",
    # Humanities
    "History":                        "historical and humanities research",
    "Literature & Cultural Studies":  "literary and cultural studies research",
    "Philosophy":                     "philosophy research",
    "Linguistics":                    "linguistics research",
    "Art History & Archaeology":      "art history and archaeology research",
    "Religion & Theology":            "religious studies and theology research",
    # Social Sciences
    "Economics":        "economics and social science research",
    "Political Science":"political science research",
    "Sociology":        "sociology research",
    "Psychology":       "psychology and behavioural science research",
    "Education":        "education research",
    "Law & Policy":     "law and public policy research",
    "Anthropology":     "anthropology research",
    # Natural Sciences
    "Physics":                 "physics research",
    "Chemistry":               "chemistry research",
    "Environmental Science":   "environmental science research",
    "Biology & Ecology":       "biology and ecology research",
    # Business
    "Business & Management":   "business and management research",
    "Finance":                 "finance research",
    # Fallback
    "Other":                   "research in this academic field",
}


class TrendAnalysisAgent:
    name = "Trend Analysis Agent"
    description = (
        "Uses TF-IDF keyword analysis to detect how a research field "
        "is evolving over time. Discipline-aware — works for humanities, "
        "social sciences, life sciences, and STEM equally."
    )

    def run(self, topic: str, domain: str = "Other") -> Dict[str, Any]:
        logger.info("[TrendAnalysis] Analysing trends for: %s (domain: %s)", topic, domain)
        collection = get_collection()
        if collection.count() == 0:
            return {"agent": self.name, "error": "No papers ingested yet."}

        result    = collection.get(include=["documents", "metadatas"])
        docs      = result["documents"]
        metadatas = result["metadatas"]

        year_keywords  = self._year_keyword_map(docs, metadatas)
        topic_clusters = self._extract_topics_llm(topic, domain, docs)

        return {
            "agent":          self.name,
            "topic":          topic,
            "total_papers":   len(docs),
            "year_keyword_distribution": year_keywords,
            "emerging_topics":  topic_clusters.get("emerging", []),
            "declining_topics": topic_clusters.get("declining", []),
            "trend_summary":    topic_clusters.get("summary", ""),
        }

    def _year_keyword_map(self, docs, metas):
        year_map = {}
        for doc, meta in zip(docs, metas):
            year = str(meta.get("year", "unknown"))
            words = [w.lower().strip(".,();:'\"") for w in doc.split() if len(w) > 4]
            words = [w for w in words if w not in _ALL_STOPWORDS and w.isalpha()]
            year_map.setdefault(year, Counter()).update(words)
        return {y: [kw for kw, _ in c.most_common(5)]
                for y, c in sorted(year_map.items()) if y != "unknown"}

    def _extract_topics_llm(self, topic: str, domain: str, docs: List[str]) -> Dict:
        llm = get_llm(temperature=0.2)
        corpus_sample = "\n---\n".join(docs[:20])

        # Build a discipline-specific framing description for the prompt
        field_description = _DISCIPLINE_FRAMING.get(domain, "research in this academic field")
        # Try to infer field more precisely from the topic text itself
        topic_lower = topic.lower()
        if any(w in topic_lower for w in ("histor", "archiv", "colonial", "medieval", "war", "empire")):
            field_description = "historical and humanities research"
        elif any(w in topic_lower for w in ("literatur", "novel", "poetr", "narrativ", "fiction")):
            field_description = "literary and cultural studies research"
        elif any(w in topic_lower for w in ("psycholog", "mental health", "behaviour", "cognit")):
            field_description = "psychology and behavioural science research"
        elif any(w in topic_lower for w in ("econom", "fiscal", "monetary", "labour", "market")):
            field_description = "economics and social science research"
        elif any(w in topic_lower for w in ("sociolog", "gender", "ethnograph", "anthropolog")):
            field_description = "sociology and anthropology research"
        elif any(w in topic_lower for w in ("law", "legal", "jurisprudence", "constitutional")):
            field_description = "legal and policy research"
        elif any(w in topic_lower for w in ("medic", "clinic", "patient", "health", "disease", "drug")):
            field_description = "biomedical and clinical research"
        elif any(w in topic_lower for w in ("physic", "chemistr", "material", "quantum")):
            field_description = "physical and natural science research"

        prompt = f"""You are a research trend analyst specialising in {field_description}.

Analyse the following paper abstracts about "{topic}" and identify:
1. 3-5 EMERGING sub-topics, themes, or approaches that are gaining traction in recent publications.
2. 2-3 DECLINING sub-topics, themes, or approaches that appear less frequently in recent work.
3. A 2-paragraph trend summary of how this field is evolving.

CRITICAL INSTRUCTIONS:
- This is {field_description}. Frame ALL trends in the natural language of THIS discipline.
- For humanities: use terms like "archival turn", "transnational focus", "postcolonial lens", "oral history methods", "digital humanities", etc.
- For social sciences: use terms like "mixed-methods approaches", "longitudinal studies", "intersectional analysis", etc.
- For life sciences: use terms like "precision medicine", "multi-omics", "clinical trials", etc.
- DO NOT mention machine learning, neural networks, LLMs, transformers, datasets, benchmarks, swarm robotics, or any AI/CS terminology UNLESS the topic itself is explicitly about AI or computer science.
- Base your answer ONLY on the actual content of the abstracts provided. Do not invent trends.

Abstracts:
{corpus_sample}

Respond ONLY in JSON with keys: emerging (list of strings), declining (list of strings), summary (string).
No markdown, no extra text."""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[TrendAnalysis] LLM failed: %s", exc)
            return {"emerging": [], "declining": [], "summary": f"Unavailable: {exc}"}
