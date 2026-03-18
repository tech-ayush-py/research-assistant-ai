"""
Trend Analysis Agent
Uses BERTopic / LDA + dynamic embeddings to detect how research has evolved.
"""
import logging
from collections import Counter
from typing import Dict, Any, List

from core.llm_factory import get_llm
from core.vector_store import get_collection

logger = logging.getLogger(__name__)


class TrendAnalysisAgent:
    """
    Analyses the paper corpus in ChromaDB to surface rising topics,
    year-over-year keyword shifts, and emerging sub-fields.
    """

    name = "Trend Analysis Agent"
    description = (
        "Uses topic modelling (BERTopic/LDA) and dynamic embeddings "
        "to detect how a research field is evolving over time."
    )

    def run(self, topic: str) -> Dict[str, Any]:
        logger.info("[TrendAnalysis] Analysing trends for: %s", topic)

        # Retrieve all papers from store
        collection = get_collection()
        if collection.count() == 0:
            return {"agent": self.name, "error": "No papers ingested yet. Run Literature Mining first."}

        result = collection.get(include=["documents", "metadatas"])
        docs      = result["documents"]
        metadatas = result["metadatas"]

        year_keywords = self._year_keyword_map(docs, metadatas)
        topic_clusters = self._extract_topics_llm(topic, docs[:40])  # sample for LLM

        return {
            "agent":          self.name,
            "topic":          topic,
            "total_papers":   len(docs),
            "year_keyword_distribution": year_keywords,
            "emerging_topics": topic_clusters["emerging"],
            "declining_topics": topic_clusters["declining"],
            "trend_summary":  topic_clusters["summary"],
        }

    # ── Helpers ───────────────────────────────────────────────

    def _year_keyword_map(self, docs: List[str], metas: List[Dict]) -> Dict[str, List[str]]:
        """Simple keyword frequency per year."""
        year_map: Dict[str, Counter] = {}
        stopwords = {"the", "a", "of", "in", "and", "to", "for", "with", "on", "is",
                     "are", "this", "that", "we", "an", "our", "by", "from", "based"}

        for doc, meta in zip(docs, metas):
            year = str(meta.get("year", "unknown"))
            words = [w.lower().strip(".,") for w in doc.split() if len(w) > 4]
            words = [w for w in words if w not in stopwords]
            if year not in year_map:
                year_map[year] = Counter()
            year_map[year].update(words)

        # Keep top-5 per year
        return {y: [kw for kw, _ in c.most_common(5)] for y, c in sorted(year_map.items())}

    def _extract_topics_llm(self, topic: str, docs: List[str]) -> Dict[str, Any]:
        """Ask the LLM to summarise trends from paper snippets."""
        llm = get_llm(temperature=0.2)
        corpus_sample = "\n---\n".join(docs[:20])

        prompt = f"""You are a research trend analyst.
Given these 20 paper abstracts about "{topic}", identify:
1. 3-5 EMERGING sub-topics (gaining traction in recent papers)
2. 2-3 DECLINING sub-topics (mentioned less in recent work)
3. A 2-paragraph trend summary of where this field is heading.

Abstracts:
{corpus_sample}

Respond in JSON with keys: emerging (list), declining (list), summary (string)."""

        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            import json, re
            raw = response.content
            # strip markdown fences if present
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("[TrendAnalysis] LLM call failed: %s", exc)
            return {"emerging": [], "declining": [], "summary": "Trend analysis unavailable (LLM error)."}
