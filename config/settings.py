"""
Central configuration for Research Assistant system.
Loads from .env and exposes typed settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-1.5-flash")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# ── Embeddings ─────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Vector Store ───────────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# ── External APIs ──────────────────────────────────────────────
S2_API_KEY      = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
ARXIV_MAX       = int(os.getenv("ARXIV_MAX_RESULTS", "50"))

# ── App ────────────────────────────────────────────────────────
APP_TITLE  = os.getenv("APP_TITLE", "AI Research Assistant & Grant Generator")
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")

# ── Grant agency templates ─────────────────────────────────────
GRANT_AGENCIES = {
    "NSF": {
        "sections": ["Project Summary", "Project Description", "Broader Impacts",
                     "Intellectual Merit", "Budget Justification", "References"],
        "max_pages": 15,
        "style": "NSF GPG guidelines"
    },
    "NIH": {
        "sections": ["Specific Aims", "Background & Significance", "Innovation",
                     "Approach", "Personnel", "Budget Narrative"],
        "max_pages": 12,
        "style": "NIH R01 format"
    },
    "DARPA": {
        "sections": ["Technical Approach", "Innovation", "Team Capabilities",
                     "Risk Mitigation", "Milestones", "Budget"],
        "max_pages": 20,
        "style": "DARPA BAA response"
    },
    "EU Horizon": {
        "sections": ["Excellence", "Impact", "Implementation", "Partners",
                     "Work Packages", "Budget Breakdown"],
        "max_pages": 30,
        "style": "Horizon Europe proposal"
    }
}

# ── Paper citation styles ──────────────────────────────────────
CITATION_STYLES = ["IEEE", "ACM", "APA", "MLA"]
