"""
Central configuration for Research Assistant system.
Loads from .env and exposes typed settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")   # free-tier compatible; 1000 req/day

# NOTE: API keys are intentionally NOT stored as module-level constants here.
# Reading them at import time produces stale values if secrets are injected
# into os.environ after this module loads (e.g. Streamlit Cloud secrets).
# Use os.getenv("GEMINI_API_KEY") at call time, or call validate_api_keys()
# once before running the pipeline.

_PROVIDER_KEY_MAP = {
    "gemini":    ("GEMINI_API_KEY",    "https://aistudio.google.com"),
    "openai":    ("OPENAI_API_KEY",    "https://platform.openai.com/api-keys"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com"),
}

def validate_api_keys() -> list:
    """
    Check that the API key for the active LLM_PROVIDER is present.
    Returns a list of human-readable error strings.
    An empty list means everything is configured correctly.
    """
    errors = []
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER)
    if provider not in _PROVIDER_KEY_MAP:
        errors.append(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Must be one of: {', '.join(_PROVIDER_KEY_MAP.keys())}."
        )
        return errors
    env_var, signup_url = _PROVIDER_KEY_MAP[provider]
    if not os.getenv(env_var, "").strip():
        errors.append(
            f"{env_var} is not set for provider '{provider}'. "
            f"Get your key at {signup_url} and add it to your .env file or Streamlit secrets."
        )
    return errors

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
