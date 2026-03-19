"""
Trend Analysis Agent
Detects how research has evolved using TF-IDF + sklearn (no torch/bertopic needed).
LLM summarises detected clusters into human-readable trends.
"""
import logging
import json
import re
from collections import Counter
from typing import Dict, Any, List

from core.llm_factory import get_llm, invoke_with_retry
from core.vector_store import get_collection

logger = logging.getLogger(__name__)


class TrendAnalysisAgent:
    name = "Trend Analysis Agent"
    description = (
        "Uses TF-IDF keyword analysis to detect how a research field "
        "is evolving over time. No heavy ML frameworks required."
    )

    def run(self, topic: str) -> Dict[str, Any]:
        logger.info("[TrendAnalysis] Analysing trends for: %s", topic)
        collection = get_collection()
        if collection.count() == 0:
            return {"agent": self.name, "error": "No papers ingested yet."}

        result    = collection.get(include=["documents", "metadatas"])
        docs      = result["documents"]
        metadatas = result["metadatas"]

        year_keywords  = self._year_keyword_map(docs, metadatas)
        topic_clusters = self._extract_topics_llm(topic, docs)

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
        stopwords = {
            "the","a","of","in","and","to","for","with","on","is","are","this",
            "that","we","an","our","by","from","based","using","paper","method",
            "propose","show","model","data","result","approach","study","which","also"
        }
        year_map = {}
        for doc, meta in zip(docs, metas):
            year = str(meta.get("year", "unknown"))
            words = [w.lower().strip(".,();:") for w in doc.split() if len(w) > 4]
            words = [w for w in words if w not in stopwords and w.isalpha()]
            year_map.setdefault(year, Counter()).update(words)
        return {y: [kw for kw, _ in c.most_common(5)]
                for y, c in sorted(year_map.items()) if y != "unknown"}

    def _extract_topics_llm(self, topic, docs):
        llm = get_llm(temperature=0.2)
        corpus_sample = "\n---\n".join(docs[:20])
        prompt = f"""You are a research trend analyst working across all academic disciplines.
Given these paper abstracts about "{topic}", identify:
1. 3-5 EMERGING sub-topics or approaches (gaining traction in recent publications)
2. 2-3 DECLINING sub-topics or approaches (less common in recent work)
3. A 2-paragraph trend summary of where the field is heading.

Important: analyse the trends in terms natural to the discipline — theoretical shifts, methodological turns, new geographic or demographic foci, emerging frameworks, etc. Do not impose computational or ML framing unless the topic is explicitly technical.

Abstracts:
{corpus_sample}

Respond ONLY in JSON with keys: emerging (list), declining (list), summary (string).
No markdown, no extra text."""

        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_retry(llm, [HumanMessage(content=prompt)])
            raw = re.sub(r"```(?:json)?|```", "", response.content).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[TrendAnalysis] LLM failed: %s", exc)
            return {"emerging": [], "declining": [], "summary": f"Unavailable: {exc}"}
