"""
ResearchAI — Academic Research Intelligence Platform
Light theme · Blue accents · High contrast · Simplified
API key hidden from UI (loaded from Streamlit secrets / .env only)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import streamlit as st

# Load secrets into env (Streamlit Cloud deployment)
# Bug 2 fix: original only synced 3 keys — OPENAI_API_KEY and ANTHROPIC_API_KEY
# were never written to os.environ when using alternative providers on Streamlit Cloud.
_ALL_SECRET_KEYS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "SEMANTIC_SCHOLAR_API_KEY",
]
try:
    for key in _ALL_SECRET_KEYS:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # Running locally without secrets.toml — fine, .env handles it

import plotly.graph_objects as go
import pandas as pd

from config.settings import GRANT_AGENCIES, CITATION_STYLES
try:
    from config.settings import validate_api_keys
except ImportError:
    # Fallback for environments where settings.py has not yet been updated.
    # Checks only the active LLM_PROVIDER key so the pipeline still validates correctly.
    import os as _os
    def validate_api_keys() -> list:
        _key_map = {
            "gemini":    "GEMINI_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        provider = _os.getenv("LLM_PROVIDER", "gemini")
        env_var  = _key_map.get(provider)
        if env_var and not _os.getenv(env_var, "").strip():
            return [f"{env_var} is not set. Add it to your Streamlit secrets or .env file."]
        return []

from core.orchestrator import ResearchOrchestrator, ResearchRequest
from core.vector_store import collection_stats
try:
    from core.vector_store import reset_collection
except ImportError:
    # Fallback for partial-deploy — define inline so the app still starts
    def reset_collection():
        try:
            import chromadb
            from chromadb.config import Settings
            from config.settings import CHROMA_DIR
            client = chromadb.PersistentClient(
                path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
            )
            try:
                client.delete_collection(name="research_papers")
            except Exception:
                pass
            client.get_or_create_collection(
                name="research_papers", metadata={"hnsw:space": "cosine"}
            )
        except Exception as _e:
            logger.warning("reset_collection fallback failed: %s", _e)
from utils.export import export_proposal_pdf, export_proposal_docx, export_report_markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ResearchAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #f4f6f9;
    --surface:   #ffffff;
    --border:    #d1d9e0;
    --blue:      #1a56db;
    --blue-dark: #1240a8;
    --blue-dim:  #e8eefb;
    --text:      #111827;
    --muted:     #374151;
    --subtle:    #6b7280;
    --green:     #166534;
    --red:       #991b1b;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"]     { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }
footer, #MainMenu            { display: none; }
.block-container { padding: 1.8rem 2.2rem 3rem !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 1px 0 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label {
    font-size: .7rem !important; font-weight: 600 !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
    color: var(--subtle) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-size: .85rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: var(--blue) !important;
    color: #ffffff !important;
    border: none !important; border-radius: 6px !important;
    font-size: .88rem !important; font-weight: 600 !important;
    padding: .72rem 1rem !important; width: 100% !important;
    transition: background .15s, box-shadow .15s !important;
    box-shadow: 0 2px 8px rgba(26,86,219,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--blue-dark) !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.4) !important;
}

/* ── Page header ── */
.page-header { padding: 1.6rem 0 1rem; border-bottom: 2px solid var(--blue); margin-bottom: 1.6rem; }
.page-title  { font-family: 'Playfair Display', serif; font-size: 2rem;
               font-weight: 700; color: var(--text); margin: 0 0 .25rem; }
.page-sub    { font-size: .85rem; color: var(--muted); }
.page-pills  { display: flex; gap: 8px; margin-top: .9rem; flex-wrap: wrap; }
.page-pill   { font-size: .67rem; font-weight: 600; letter-spacing: .09em;
               text-transform: uppercase; color: var(--blue);
               background: var(--blue-dim); border: 1px solid #c3d3f5;
               padding: 3px 10px; border-radius: 3px; }

/* ── Stat cards ── */
.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-top: 3px solid var(--blue); border-radius: 6px;
    padding: 1rem 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-card.green { border-top-color: #166534; }
.stat-card.slate { border-top-color: #6b7280; }
.stat-card.teal  { border-top-color: #0e7490; }
.stat-label { font-size: .65rem; font-weight: 700; letter-spacing: .1em;
              text-transform: uppercase; color: var(--subtle); margin-bottom: 5px; }
.stat-value { font-family: 'Playfair Display', serif; font-size: 2rem;
              font-weight: 700; color: var(--text); line-height: 1; }
.stat-delta { font-size: .72rem; color: var(--blue); margin-top: 4px; font-weight: 500; }

/* ── Data card ── */
.data-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.2rem 1.4rem; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.card-title { font-family: 'Playfair Display', serif; font-size: .95rem;
              font-weight: 700; color: var(--text); margin-bottom: 3px; }

/* ── Section label ── */
.sec-label { font-size: .65rem; font-weight: 700; letter-spacing: .12em;
             text-transform: uppercase; color: var(--blue);
             display: block; margin-bottom: 10px; }

/* ── Confidence bar ── */
.conf-row { margin: 5px 0 11px; }
.conf-header { display: flex; justify-content: space-between;
               font-size: .72rem; margin-bottom: 4px; }
.conf-name { font-weight: 600; color: var(--text); }
.conf-pct  { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }
.conf-track { height: 5px; background: #e5e7eb; border-radius: 3px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 3px; transition: width 1s ease; }

/* ── Gap item ── */
.gap-item {
    border-left: 3px solid var(--blue);
    background: var(--blue-dim);
    border-radius: 0 6px 6px 0;
    padding: .75rem 1rem; margin: .5rem 0;
    font-size: .84rem; color: var(--text); line-height: 1.65;
}
.gap-num { font-size: .62rem; font-weight: 700; letter-spacing: .1em;
           text-transform: uppercase; color: var(--blue); margin-bottom: 4px; }

/* ── Pills ── */
.pill { display: inline-block; font-size: .73rem; font-weight: 500;
        padding: 4px 11px; border-radius: 3px; margin: 3px; }
.pill-blue  { background: var(--blue-dim); color: var(--blue-dark);
              border: 1px solid #c3d3f5; }
.pill-green { background: #dcfce7; color: #14532d; border: 1px solid #bbf7d0; }
.pill-slate { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
.pill-red   { background: #fee2e2; color: #7f1d1d; border: 1px solid #fecaca; }

/* ── Step log ── */
.step-row { display: flex; align-items: flex-start; gap: 10px;
            padding: 6px 0; border-bottom: 1px solid var(--border);
            font-size: .8rem; }
.step-row:last-child { border-bottom: none; }
.step-dot { width: 7px; height: 7px; border-radius: 50%;
            flex-shrink: 0; margin-top: 5px; }
.dot-done    { background: #166534; }
.dot-running { background: var(--blue); }
.dot-wait    { background: #d1d5db; }
.step-text-done    { color: var(--text); }
.step-text-running { color: var(--blue); font-weight: 600; }
.step-text-wait    { color: #9ca3af; }

/* ── Hypothesis ── */
.hypothesis {
    border-left: 3px solid var(--blue); background: var(--blue-dim);
    border-radius: 0 6px 6px 0; padding: .9rem 1.1rem;
    font-size: .87rem; line-height: 1.75; color: var(--text);
    margin-bottom: 16px; font-style: italic;
}

/* ── Approach step ── */
.approach-step { display: flex; gap: 12px; align-items: flex-start;
                 padding: .65rem 0; border-bottom: 1px solid var(--border);
                 font-size: .83rem; line-height: 1.6; color: var(--text); }
.approach-step:last-child { border-bottom: none; }
.step-num { font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
            font-weight: 600; color: var(--blue); min-width: 24px; padding-top: 2px; }

/* ── Grant meta ── */
.grant-meta-row { display: flex; gap: 28px; padding: .9rem 0;
                  border-bottom: 1px solid var(--border); margin-bottom: 18px;
                  flex-wrap: wrap; }
.grant-meta-label { font-size: .62rem; font-weight: 700; letter-spacing: .1em;
                    text-transform: uppercase; color: var(--subtle); margin-bottom: 3px; }
.grant-meta-val   { font-weight: 600; font-size: .88rem; color: var(--text); }

/* ── Banners ── */
.info-bar { background: var(--blue-dim); border: 1px solid #c3d3f5;
            border-radius: 6px; padding: .7rem 1rem;
            font-size: .82rem; color: #1240a8; margin-bottom: 1rem; }
.warn-bar { background: #fee2e2; border: 1px solid #fca5a5;
            border-radius: 6px; padding: .7rem 1rem;
            font-size: .82rem; color: #7f1d1d; margin-bottom: 1rem; }

/* ── Tabs ── */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border) !important; }
[data-testid="stTabs"] button {
    font-size: .78rem !important; font-weight: 600 !important;
    letter-spacing: .07em !important; text-transform: uppercase !important;
    color: var(--subtle) !important; padding: .7rem 1.1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background: var(--blue) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary { font-weight: 600 !important;
    font-size: .84rem !important; color: var(--text) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid var(--blue) !important;
    color: var(--blue) !important; border-radius: 5px !important;
    font-weight: 600 !important; font-size: .8rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--blue) !important; color: #fff !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── General ── */
hr { border-color: var(--border) !important; }
.stMarkdown p, .stMarkdown li { color: var(--muted) !important; font-size: .85rem !important; line-height: 1.7 !important; }
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important; color: var(--text) !important; }
[data-testid="stAlert"] { background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────
for k, v in [("report", None), ("pipeline_log", []), ("chat_history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Page header ─────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">ResearchAI</div>
    <div class="page-sub">Academic Research Intelligence Platform</div>
    <div class="page-pills">
        <span class="page-pill">6 AI Agents</span>
        <span class="page-pill">ArXiv + Semantic Scholar</span>
        <span class="page-pill">ChromaDB RAG</span>
        <span class="page-pill">Powered by Gemini</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — research + grant settings only, no API fields
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:1.2rem 1rem .8rem;border-bottom:1px solid #e5e7eb'>"
        "<div style='font-family:Playfair Display,serif;font-size:1.05rem;"
        "font-weight:700;color:#111827'>Configuration</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Research topic
    research_topic = st.text_area(
        "Research Topic",
        placeholder="Describe your research in one or two sentences.\n\nExample: Federated Learning for Privacy-Preserving Medical Image Segmentation.",
        height=100,
    )
    domain = st.selectbox("Research Domain", [
        # Sciences
        "General AI / Machine Learning",
        "Natural Language Processing (NLP)",
        "Computer Vision",
        "Robotics",
        "Cybersecurity",
        "Data Science",
        # Life & Health Sciences
        "Biomedical / Medicine",
        "Public Health",
        "Neuroscience",
        "Environmental Science",
        "Physics / Chemistry",
        # Social Sciences
        "Psychology",
        "Sociology",
        "Economics",
        "Political Science",
        "Education",
        "Social Work",
        # Humanities
        "History",
        "Literature & Linguistics",
        "Philosophy",
        "Cultural Studies / Gender Studies",
        "Art History & Archaeology",
        "Media & Communication",
        "Law",
        # Business
        "Business / Management",
        "Finance",
        # Interdisciplinary
        "Interdisciplinary / Other",
    ])
    max_papers = st.slider("Papers to retrieve", 10, 80, 25,
                           help="More papers improve analysis quality but increase runtime.")

    st.divider()

    # Grant settings
    st.markdown(
        "<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;"
        "text-transform:uppercase;color:#6b7280;margin-bottom:10px'>Grant Proposal</div>",
        unsafe_allow_html=True,
    )
    grant_agency   = st.text_input("Funding Agency", "", placeholder="e.g. NSF, NIH, DARPA, EU Horizon")
    pi_name        = st.text_input("Principal Investigator", "")
    institution    = st.text_input("Institution", "")
    budget_total   = st.text_input("Total Budget", "")
    duration_years = st.number_input("Duration (years)", 1, 10, 3)
    citation_style = st.selectbox("Citation Style", CITATION_STYLES)

    st.divider()
    run_pipeline = st.button("Run Research Pipeline", type="primary", use_container_width=True)

    stats = collection_stats()
    st.markdown(
        f"<div style='text-align:center;margin-top:10px;font-size:.72rem;color:#9ca3af'>"
        f"{stats['total_papers']} papers in corpus</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════
AGENT_STEPS = [
    ("literature",  "Literature Mining",    "Retrieving and indexing papers from ArXiv and Semantic Scholar"),
    ("trends",      "Trend Analysis",       "Detecting how the research field has evolved over time"),
    ("gaps",        "Gap Identification",   "Mapping under-explored areas in the literature"),
    ("methodology", "Methodology Design",   "Designing experiment, datasets, baselines, and metrics"),
    ("grant",       "Grant Writing",        "Composing structured proposal sections"),
    ("novelty",     "Novelty Scoring",      "Benchmarking originality against the corpus"),
    ("done",        "Complete",             ""),
]
step_prog = {s: i/(len(AGENT_STEPS)-1) for i,(s,_,__) in enumerate(AGENT_STEPS)}

if run_pipeline:
    if not research_topic.strip():
        st.markdown('<div class="warn-bar">Please enter a research topic in the sidebar.</div>', unsafe_allow_html=True)
    else:
        # Bug 3 fix: original check hardcoded GEMINI_API_KEY, so switching to
        # openai or anthropic provider let an empty key through to the agent layer,
        # producing a cryptic LangChain auth error. validate_api_keys() checks
        # whichever provider is currently active.
        _key_errors = validate_api_keys()
        if _key_errors:
            for _err in _key_errors:
                st.markdown(f'<div class="warn-bar">{_err}</div>', unsafe_allow_html=True)
        else:
            st.session_state.pipeline_log = []
            st.markdown('<div class="info-bar">Pipeline running — this takes 1–3 minutes. Do not refresh the page.</div>', unsafe_allow_html=True)

            # Clear the corpus from any previous run so papers from a different
            # topic (e.g. robotics) do not contaminate results for the new topic.
            reset_collection()

            prog_bar    = st.progress(0)
            col_s, col_l = st.columns(2)
            prog_status  = col_s.empty()
            log_area     = col_l.empty()
            agent_states = {s:"wait" for s,_,__ in AGENT_STEPS[:-1]}

            def progress_cb(step, msg):
                st.session_state.pipeline_log.append((step, msg))
                prog_bar.progress(step_prog.get(step, 0))
                if step in agent_states:
                    agent_states[step] = "done"
                prog_status.markdown(
                    "<div style='font-size:.8rem;color:#1a56db;padding:.3rem 0'>"
                    "Running: " + msg + "</div>", unsafe_allow_html=True)
                rows = ""
                for s, lbl, desc in AGENT_STEPS[:-1]:
                    state = agent_states.get(s, "wait")
                    rows += (
                        "<div class='step-row'>"
                        "<span class='step-dot dot-" + state + "'></span>"
                        "<span class='step-text-" + state + "'><b>" + lbl + "</b> — " + desc + "</span>"
                        "</div>"
                    )
                log_area.markdown(f"<div class='data-card' style='padding:1rem'>{rows}</div>", unsafe_allow_html=True)

            request = ResearchRequest(
                topic=research_topic, domain=domain,
                grant_agency=grant_agency.strip() or "General",
                pi_name=pi_name, institution=institution,
                budget_total=budget_total,
                duration_years=duration_years, citation_style=citation_style,
                max_papers=max_papers,
            )
            try:
                report = ResearchOrchestrator().run(request, progress_callback=progress_cb)
                st.session_state.report = report
                prog_bar.progress(1.0)
                prog_status.markdown(
                    "<div style='font-size:.82rem;color:#166534;font-weight:600;padding:.3rem 0'>"
                    "Pipeline complete. View results below.</div>", unsafe_allow_html=True)
            except RuntimeError as e:
                err_msg = str(e)
                if "quota" in err_msg.lower() or "daily" in err_msg.lower():
                    st.markdown(
                        f'<div class="warn-bar"><b>Daily quota exhausted.</b> {err_msg}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Pipeline error: {e}")
                logger.exception(e)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                logger.exception(e)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
report = st.session_state.report

if report:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "OVERVIEW", "GAPS & TRENDS", "METHODOLOGY", "GRANT PROPOSAL", "ASSISTANT"
    ])

    # ── OVERVIEW ──────────────────────────────────────────────
    with tab1:
        novelty = report.novelty.get("novelty_score", 0)
        fetched = report.literature.get("fetched", 0)
        n_gaps  = len(report.gaps.get("identified_gaps", []))
        n_secs  = len(report.grant.get("sections", {}))

        st.markdown(
            f"<div class='stat-row'>"
            f"<div class='stat-card'>"
            f"<div class='stat-label'>Papers Retrieved</div>"
            f"<div class='stat-value'>{fetched}</div>"
            f"<div class='stat-delta'>{report.literature.get('new_ingested',0)} newly added</div></div>"
            f"<div class='stat-card teal'>"
            f"<div class='stat-label'>Research Gaps</div>"
            f"<div class='stat-value'>{n_gaps}</div>"
            f"<div class='stat-delta'>Identified from literature</div></div>"
            f"<div class='stat-card green'>"
            f"<div class='stat-label'>Novelty Score</div>"
            f"<div class='stat-value'>{novelty:.2f}</div>"
            f"<div class='stat-delta'>{report.novelty.get('novelty_label','').replace('_',' ').title()}</div></div>"
            f"<div class='stat-card slate'>"
            f"<div class='stat-label'>Proposal Sections</div>"
            f"<div class='stat-value'>{n_secs}</div>"
            f"<div class='stat-delta'>Ready to download</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            # Novelty gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=novelty,
                number={"font": {"size":32, "color":"#1a56db", "family":"IBM Plex Mono"}},
                gauge={
                    "axis": {"range":[0,1], "tickcolor":"#d1d9e0",
                             "tickfont":{"color":"#6b7280","size":10}},
                    "bar":  {"color":"#1a56db", "thickness":.22},
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                    "steps": [
                        {"range":[0,.35],  "color":"rgba(239,68,68,.12)"},
                        {"range":[.35,.55],"color":"rgba(234,179,8,.10)"},
                        {"range":[.55,.75],"color":"rgba(34,197,94,.10)"},
                        {"range":[.75,1],  "color":"rgba(26,86,219,.12)"},
                    ],
                },
                title={"text":"Novelty Score","font":{"color":"#6b7280","size":12,"family":"IBM Plex Sans"}},
            ))
            fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=20,r=20,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;font-size:.78rem;color:#374151;margin-top:-6px'>"
                f"{report.novelty.get('recommendation','')}</div>",
                unsafe_allow_html=True,
            )

        with col_b:
            papers = report.literature.get("top_papers", [])
            if papers:
                df = pd.DataFrame(papers)
                if "year" in df.columns:
                    yc = df["year"].value_counts().sort_index()
                    fig2 = go.Figure(go.Bar(
                        x=yc.index.astype(str), y=yc.values,
                        marker_color="#1a56db", marker_opacity=0.8,
                        hovertemplate="%{x}: %{y} papers<extra></extra>",
                    ))
                    fig2.update_layout(
                        title=dict(text="Papers by Year",
                                   font=dict(size=11,color="#6b7280",family="IBM Plex Sans")),
                        height=230,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10,r=10,t=36,b=10),
                        xaxis=dict(gridcolor="#f3f4f6", color="#6b7280", tickfont=dict(size=10)),
                        yaxis=dict(gridcolor="#f3f4f6", color="#6b7280", tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # Papers table
        if papers:
            st.markdown('<span class="sec-label">Retrieved Papers</span>', unsafe_allow_html=True)
            df = pd.DataFrame(papers)
            cols_available = [c for c in ["title","year","authors","source","citation_count","influential_citation_count","credibility","similarity","score"] if c in df.columns]
            df_show = df[cols_available].head(12).copy()
            rename_map = {
                "title": "Title", "year": "Year", "authors": "Authors",
                "source": "Source", "citation_count": "Citations",
                "influential_citation_count": "Influential Cites",
                "credibility": "Credibility", "similarity": "Relevance", "score": "Score",
            }
            df_show.columns = [rename_map.get(c, c) for c in df_show.columns]
            st.dataframe(df_show, use_container_width=True, height=280)

    # ── GAPS & TRENDS ─────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Research Gaps</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.78rem;color:#374151;margin-bottom:12px;line-height:1.6'>"
                "Areas identified as under-explored in the existing literature.</div>",
                unsafe_allow_html=True,
            )

            gap_details = report.gaps.get("gap_details", [])
            identified  = report.gaps.get("identified_gaps", [])

            for i, gap_text in enumerate(identified):
                detail = gap_details[i] if i < len(gap_details) else {}
                conf   = detail.get("confidence", "")
                note   = detail.get("verification_note", "")
                conflict = detail.get("conflict")

                # Confidence badge colours
                badge_color = {"high": "#166534", "medium": "#854d0e", "low": "#7f1d1d"}.get(conf, "#374151")
                badge_bg    = {"high": "#dcfce7", "medium": "#fef9c3", "low": "#fee2e2"}.get(conf, "#f3f4f6")
                badge_label = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}.get(conf, "")

                badge_html = (
                    f"<span style='font-size:.68rem;font-weight:700;padding:2px 8px;"
                    f"border-radius:10px;background:{badge_bg};color:{badge_color};"
                    f"margin-left:8px'>{badge_label}</span>"
                ) if badge_label else ""

                conflict_html = ""
                if conflict:
                    conflict_html = (
                        f"<div style='margin-top:6px;padding:5px 9px;background:#fee2e2;"
                        f"border-radius:5px;font-size:.76rem;color:#7f1d1d;line-height:1.5'>"
                        f"⚠ Possible overlap: <b>{conflict['title']}</b> ({conflict['year']}) "
                        f"— similarity {conflict['similarity']:.2f}</div>"
                    )

                note_html = (
                    f"<div style='font-size:.78rem;color:#4b5563;font-style:italic;"
                    f"margin-top:5px;line-height:1.5'>{note}</div>"
                ) if note else ""

                st.markdown(
                    f"<div class='gap-item'>"
                    f"<div class='gap-num' style='display:flex;align-items:center'>"
                    f"Gap {i+1}{badge_html}</div>"
                    f"{gap_text}"
                    f"{note_html}"
                    f"{conflict_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            if report.gaps.get("opportunity_areas"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Opportunity Areas</span>', unsafe_allow_html=True)
                for opp in report.gaps["opportunity_areas"]:
                    st.markdown(
                        f"<div style='padding:.55rem 0;border-bottom:1px solid #e5e7eb;"
                        f"font-size:.83rem;color:#1e293b;line-height:1.6'>{opp}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Emerging Topics</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#374151;margin-bottom:10px'>"
                "Gaining traction in recent publications.</div>", unsafe_allow_html=True)
            for t in report.trends.get("emerging_topics", []):
                st.markdown(f"<span class='pill pill-green'>{t}</span>", unsafe_allow_html=True)
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Declining Topics</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#374151;margin-bottom:10px'>"
                "Appearing less frequently in recent work.</div>", unsafe_allow_html=True)
            for t in report.trends.get("declining_topics", []):
                st.markdown(f"<span class='pill pill-red'>{t}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if report.trends.get("trend_summary"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Field Summary</span>', unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:.82rem;line-height:1.8;color:#1e293b'>"
                    f"{report.trends['trend_summary']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Novelty Analysis</span>', unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:.75rem;color:#374151;margin-top:6px'>"
                f"Rating: <b style='color:#1a56db'>"
                f"{report.novelty.get('novelty_label','').replace('_',' ').title()}</b>"
                f" — compared against {report.novelty.get('corpus_size',0)} papers</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── METHODOLOGY ───────────────────────────────────────────
    with tab3:
        hyp = report.methodology.get("hypothesis","")
        if hyp:
            st.markdown(
                f"<div class='hypothesis'>"
                f"<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;"
                f"text-transform:uppercase;color:#1a56db;font-style:normal;margin-bottom:8px'>"
                f"Hypothesis</div>{hyp}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Research Approach</span>', unsafe_allow_html=True)
            for i, step in enumerate(report.methodology.get("approach",[]), 1):
                st.markdown(
                    f"<div class='approach-step'><span class='step-num'>{i:02d}.</span>"
                    f"<span>{step}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if report.methodology.get("suggested_datasets"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Data Sources</span>', unsafe_allow_html=True)
                for ds in report.methodology.get("suggested_datasets",[]):
                    st.markdown(f"<span class='pill pill-blue'>{ds}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Comparators & Prior Work</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#374151;margin-bottom:10px'>"
                "Existing studies, frameworks, or theories this work engages with.</div>",
                unsafe_allow_html=True,
            )
            for bl in report.methodology.get("baselines",[]):
                st.markdown(
                    f"<div style='font-size:.83rem;"
                    f"color:#1e293b;padding:.3rem .7rem;margin:.25rem 0;"
                    f"border:1px solid #e2e8f0;border-radius:4px;background:#f8fafc'>"
                    f"{bl}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if report.methodology.get("evaluation_metrics"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Evaluation Criteria</span>', unsafe_allow_html=True)
                for m in report.methodology.get("evaluation_metrics",[]):
                    st.markdown(f"<span class='pill pill-slate'>{m}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        if report.methodology.get("expected_outcomes"):
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Expected Contributions</span>', unsafe_allow_html=True)
            for o in report.methodology["expected_outcomes"]:
                st.markdown(
                    f"<div style='padding:.5rem 0;border-bottom:1px solid #e5e7eb;"
                    f"font-size:.83rem;color:#1e293b;line-height:1.6'>{o}</div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── GRANT PROPOSAL ────────────────────────────────────────
    with tab4:
        st.markdown(
            f"<div class='grant-meta-row'>"
            f"<div><div class='grant-meta-label'>Agency</div><div class='grant-meta-val'>{report.grant.get('agency','')}</div></div>"
            f"<div><div class='grant-meta-label'>Principal Investigator</div><div class='grant-meta-val'>{report.grant.get('pi_name','')}</div></div>"
            f"<div><div class='grant-meta-label'>Institution</div><div class='grant-meta-val'>{report.grant.get('institution','')}</div></div>"
            f"<div><div class='grant-meta-label'>Budget</div><div class='grant-meta-val'>{report.grant.get('budget_total','')}</div></div>"
            f"<div><div class='grant-meta-label'>Duration</div><div class='grant-meta-val'>{report.grant.get('duration_years','')} years</div></div>"
            f"</div>", unsafe_allow_html=True)

        n_secs = len(report.grant.get("sections",{}))
        st.markdown(
            f"<div style='font-size:.8rem;color:#374151;margin-bottom:1rem'>"
            f"{n_secs} sections generated</div>",
            unsafe_allow_html=True,
        )

        for section, content in report.grant.get("sections",{}).items():
            with st.expander(section):
                st.markdown(f"<div style='font-size:.84rem;line-height:1.85;color:#1e293b'>{content}</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown('<span class="sec-label">Download Proposal</span>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.76rem;color:#374151;margin-bottom:12px'>"
            "PDF and DOCX are submission-ready. Markdown is best for further editing.</div>",
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            if st.button("Generate PDF", use_container_width=True):
                path = export_proposal_pdf(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path,"rb") as f:
                        st.download_button("Download PDF", f, file_name=os.path.basename(path),
                                           mime="application/pdf", use_container_width=True)
                else:
                    st.error("PDF generation failed.")
        with ec2:
            if st.button("Generate Word Document", use_container_width=True):
                path = export_proposal_docx(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path,"rb") as f:
                        st.download_button("Download DOCX", f, file_name=os.path.basename(path),
                                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                           use_container_width=True)
                else:
                    st.error("DOCX generation failed.")
        with ec3:
            if st.button("Generate Markdown Report", use_container_width=True):
                rd = {"literature": report.literature, "trends": report.trends,
                      "gaps": report.gaps, "methodology": report.methodology,
                      "grant": {k:v for k,v in report.grant.items() if k!="full_proposal"},
                      "novelty": report.novelty}
                path = export_report_markdown(rd, "./outputs")
                if path and os.path.exists(path):
                    with open(path) as f:
                        st.download_button("Download Markdown", f, file_name=os.path.basename(path),
                                           mime="text/markdown", use_container_width=True)

    # ── ASSISTANT ─────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='data-card' style='margin-bottom:1rem'>", unsafe_allow_html=True)
        st.markdown('<span class="sec-label">Research Assistant</span>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.82rem;color:#374151;line-height:1.7'>"
            "Ask questions about your research results, proposal content, gaps, or novelty scores. "
            "The assistant has full context of the current report.</div>"
            "<div style='font-size:.78rem;color:#6b7280;margin-top:.6rem'>"
            "<b>Try:</b> What does my novelty score mean? &nbsp;|&nbsp; "
            "Which gap should I prioritise? &nbsp;|&nbsp; "
            "Is this proposal ready for submission?"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about your research…")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.write(user_input)

            context = f"""You are a senior academic research assistant.
Report data:
- Topic: {report.request.topic}
- Agency: {report.grant.get('agency','')}
- Novelty score: {report.novelty.get('novelty_score',0):.2f} ({report.novelty.get('novelty_label','')})
- Papers retrieved: {report.literature.get('fetched',0)}
- Gaps: {'; '.join(report.gaps.get('identified_gaps',[])[:4])}
- Emerging trends: {'; '.join(report.trends.get('emerging_topics',[])[:3])}
- Hypothesis: {report.methodology.get('hypothesis','Not generated')}
- Proposal sections: {', '.join(report.grant.get('sections',{}).keys())}
- Recommendation: {report.novelty.get('recommendation','')}
Be direct, concise, and reference specific data from the report."""

            from core.llm_factory import get_llm, invoke_with_retry
            from langchain_core.messages import HumanMessage, SystemMessage
            try:
                llm  = get_llm(temperature=0.4)
                msgs = [SystemMessage(content=context),
                        *[HumanMessage(content=m["content"]) if m["role"]=="user"
                          else type("A",(),{"content":m["content"],"type":"ai"})()
                          for m in st.session_state.chat_history[-8:]]]
                reply = invoke_with_retry(llm, msgs).content
            except Exception as e:
                reply = f"Unable to generate a response: {e}"

            st.session_state.chat_history.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"):
                st.write(reply)

# ── EMPTY STATE ──────────────────────────────────────────────────
else:
    st.markdown(
        "<div class='info-bar' style='text-align:center'>"
        "Fill in the research parameters in the left panel and click "
        "<b>Run Research Pipeline</b> to begin."
        "</div>", unsafe_allow_html=True)

    cards = [
        ("What it does",
         "Runs six AI agents in sequence — retrieves papers from ArXiv and Semantic Scholar, "
         "detects how the field is evolving, identifies under-explored gaps, "
         "designs an experimental methodology, writes a grant proposal, "
         "and scores the originality of the proposed work."),
        ("What you need",
         "A research topic described in one or two sentences, and standard grant details — "
         "PI name, institution, funding agency, and budget. "
         "The Gemini API key is pre-configured and does not need to be entered."),
        ("What you receive",
         "A novelty score, a prioritised list of research gaps, a suggested experimental design "
         "with datasets and baselines, and a complete grant proposal ready to download as "
         "PDF, Word, or Markdown — formatted for NSF, NIH, DARPA, or EU Horizon."),
    ]
    c1, c2, c3 = st.columns(3)
    for col, (title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                "<div class='data-card'>"
                "<div class='card-title'>" + title + "</div>"
                "<div style='height:3px;width:28px;background:#1a56db;"
                "border-radius:2px;margin:.5rem 0 .85rem'></div>"
                "<div style='font-size:.82rem;line-height:1.8;color:#374151'>" + body + "</div>"
                "</div>", unsafe_allow_html=True)
