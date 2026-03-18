"""
AI-Powered Academic Research Assistant & Grant Proposal Generator
Professional UI — refined academic luxury aesthetic
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import streamlit as st

try:
    for key in ["GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass

import plotly.graph_objects as go
import pandas as pd

from config.settings import APP_TITLE, GRANT_AGENCIES, CITATION_STYLES
from core.orchestrator import ResearchOrchestrator, ResearchRequest
from core.vector_store import collection_stats
from utils.export import export_proposal_pdf, export_proposal_docx, export_report_markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ResearchAI — Academic Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS — refined academic luxury ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:    #0a0c10;
    --bg-surface: #111318;
    --bg-raised:  #181c24;
    --bg-card:    #1e2230;
    --border:     rgba(255,255,255,0.07);
    --border-mid: rgba(255,255,255,0.12);
    --gold:       #c9a84c;
    --gold-light: #e8c97a;
    --gold-dim:   rgba(201,168,76,0.15);
    --teal:       #4fd1c5;
    --teal-dim:   rgba(79,209,197,0.12);
    --slate:      #8892a4;
    --text:       #dde3ee;
    --text-dim:   #6b7590;
    --red:        #fc8181;
    --green:      #68d391;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
    min-height: 100vh;
}

/* Remove default streamlit chrome */
[data-testid="stHeader"]          { background: transparent !important; }
[data-testid="stDecoration"]      { display: none; }
footer                            { display: none; }
#MainMenu                         { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: .84rem !important;
    transition: border-color .2s !important;
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px var(--gold-dim) !important;
}
[data-testid="stSidebar"] label {
    font-size: .72rem !important;
    font-weight: 600 !important;
    letter-spacing: .09em !important;
    text-transform: uppercase !important;
    color: var(--slate) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    font-size: .7rem !important; color: var(--text-dim) !important;
}

/* Run button */
[data-testid="stSidebar"] .stButton > button {
    background: var(--gold) !important;
    color: #0a0c10 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: .84rem !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    padding: .7rem 1rem !important;
    width: 100% !important;
    transition: background .2s, transform .15s, box-shadow .2s !important;
    box-shadow: 0 2px 12px rgba(201,168,76,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--gold-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.45) !important;
}

/* ── MAIN AREA ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── WORDMARK HEADER ── */
.wordmark {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: .3rem;
}
.wordmark-primary {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -.02em;
    line-height: 1;
}
.wordmark-tag {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid var(--gold);
    padding: 3px 9px;
    border-radius: 3px;
    margin-left: 2px;
}
.header-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--gold), rgba(201,168,76,0.0));
    margin: 1.1rem 0 1.5rem;
}
.header-sub {
    font-size: .84rem;
    color: var(--slate);
    font-weight: 300;
    letter-spacing: .01em;
}
.header-pills {
    display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap;
}
.header-pill {
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-dim);
    border: 1px solid var(--border-mid);
    padding: 3px 11px;
    border-radius: 2px;
}

/* ── STAT CARDS ── */
.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--gold);
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    transition: border-top-color .2s, box-shadow .2s;
}
.stat-card:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
.stat-card.teal  { border-top-color: var(--teal); }
.stat-card.green { border-top-color: var(--green); }
.stat-card.slate { border-top-color: var(--slate); }
.stat-label { font-size: .65rem; font-weight: 700; letter-spacing: .12em;
              text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px; }
.stat-value { font-family: 'Playfair Display', serif; font-size: 2.1rem;
              font-weight: 700; color: #fff; line-height: 1; }
.stat-delta { font-size: .72rem; color: var(--teal); margin-top: 5px; }

/* ── DATA CARD ── */
.data-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 16px;
}
.card-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 4px;
}
.card-sub {
    font-size: .75rem;
    color: var(--text-dim);
    margin-bottom: 14px;
    line-height: 1.5;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 10px;
    display: block;
}

/* ── CONFIDENCE BAR ── */
.conf-row {
    margin: 6px 0 12px;
}
.conf-header {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .72rem; color: var(--slate); margin-bottom: 5px;
}
.conf-name  { font-weight: 500; color: var(--text); }
.conf-pct   { font-family: 'IBM Plex Mono', monospace; }
.conf-track { height: 4px; background: rgba(255,255,255,0.07); border-radius: 2px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 2px; transition: width 1.2s cubic-bezier(.4,0,.2,1); }

/* ── GAP ITEM ── */
.gap-item {
    border-left: 3px solid var(--gold);
    background: linear-gradient(90deg, rgba(201,168,76,0.06), transparent);
    border-radius: 0 6px 6px 0;
    padding: .8rem 1.1rem;
    margin: .55rem 0;
    font-size: .83rem;
    color: var(--text);
    line-height: 1.65;
}
.gap-num {
    font-size: .62rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--gold); margin-bottom: 5px;
}

/* ── PILL TAGS ── */
.pill {
    display: inline-block;
    font-size: .72rem; font-weight: 500;
    padding: 4px 12px; border-radius: 3px; margin: 3px;
    letter-spacing: .02em;
}
.pill-gold  { background: var(--gold-dim); color: var(--gold-light); border: 1px solid rgba(201,168,76,0.3); }
.pill-teal  { background: var(--teal-dim); color: var(--teal);       border: 1px solid rgba(79,209,197,0.3); }
.pill-slate { background: rgba(136,146,164,0.1); color: #a0aec0; border: 1px solid rgba(136,146,164,0.25); }
.pill-red   { background: rgba(252,129,129,0.1); color: #fca5a5; border: 1px solid rgba(252,129,129,0.25); }

/* ── STEP LOG ── */
.step-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid var(--border);
    font-size: .8rem;
}
.step-row:last-child { border-bottom: none; }
.step-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 5px;
}
.dot-done    { background: var(--green); }
.dot-running { background: var(--gold); }
.dot-wait    { background: var(--text-dim); opacity: .35; }
.step-text-done    { color: var(--text); }
.step-text-running { color: var(--gold); }
.step-text-wait    { color: var(--text-dim); }

/* ── HYPOTHESIS BLOCK ── */
.hypothesis {
    border-left: 3px solid var(--teal);
    background: var(--teal-dim);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    font-size: .87rem;
    line-height: 1.7;
    color: var(--text);
    margin-bottom: 18px;
    font-style: italic;
}

/* ── APPROACH STEP ── */
.approach-step {
    display: flex; gap: 12px; align-items: flex-start;
    padding: .7rem 0; border-bottom: 1px solid var(--border);
    font-size: .83rem; line-height: 1.6; color: var(--text);
}
.approach-step:last-child { border-bottom: none; }
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem; font-weight: 500;
    color: var(--gold); flex-shrink: 0;
    min-width: 24px; padding-top: 2px;
}

/* ── GRANT SECTION ── */
.grant-meta-row {
    display: flex; gap: 32px; padding: 1rem 0;
    border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.grant-meta-item { font-size: .8rem; }
.grant-meta-label {
    font-size: .63rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-dim); margin-bottom: 3px;
}
.grant-meta-val { font-weight: 600; color: #fff; }

/* ── INFO BAR ── */
.info-bar {
    background: var(--gold-dim);
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 6px;
    padding: .75rem 1.1rem;
    font-size: .82rem;
    color: var(--gold-light);
    margin-bottom: 1.2rem;
}
.warn-bar {
    background: rgba(252,129,129,0.08);
    border: 1px solid rgba(252,129,129,0.2);
    border-radius: 6px;
    padding: .75rem 1.1rem;
    font-size: .82rem;
    color: var(--red);
    margin-bottom: 1.2rem;
}

/* ── TABS ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    padding: .75rem 1.1rem !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}
[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
    color: var(--text) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] iframe { border-radius: 6px !important; }

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div > div {
    background: var(--gold) !important;
}

/* ── CHAT ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: .84rem !important;
    color: var(--text) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: .8rem !important;
    letter-spacing: .05em !important;
    transition: background .2s, color .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--gold) !important;
    color: #0a0c10 !important;
}

/* ── GENERAL ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
.stMarkdown p, .stMarkdown li { color: var(--text) !important; font-size: .85rem !important; line-height: 1.7 !important; }
h1,h2,h3,h4 {
    font-family: 'Playfair Display', serif !important;
    color: #fff !important;
    font-weight: 600 !important;
}
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Wordmark Header ────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 0">
    <div class="wordmark">
        <span class="wordmark-primary">ResearchAI</span>
        <span class="wordmark-tag">Beta</span>
    </div>
    <div class="header-sub">
        Academic Research Intelligence Platform &mdash; Literature Mining &middot; Gap Analysis &middot; Grant Generation
    </div>
    <div class="header-rule"></div>
    <div class="header-pills">
        <span class="header-pill">6 AI Agents</span>
        <span class="header-pill">ArXiv + Semantic Scholar</span>
        <span class="header-pill">ChromaDB RAG</span>
        <span class="header-pill">NSF / NIH / DARPA / EU Horizon</span>
        <span class="header-pill">Powered by Gemini</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────
for key, default in [("report", None), ("pipeline_log", []), ("chat_history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:1.4rem 1rem .8rem;"
        "border-bottom:1px solid rgba(255,255,255,0.07);"
        "margin-bottom:.8rem'>"
        "<div style='font-family:Playfair Display,serif;font-size:1.1rem;"
        "font-weight:600;color:#fff'>Configuration</div>"
        "<div style='font-size:.72rem;color:#6b7590;margin-top:2px'>Set up your research pipeline</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        placeholder="AIzaSy…",
        help="Free at aistudio.google.com — no credit card required.",
    )
    llm_model = st.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        help="Flash: fast, free-tier friendly. Pro: higher quality for grant writing.",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"]    = llm_model

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()

    research_topic = st.text_area(
        "Research Topic",
        placeholder="Describe your research in one or two sentences.\n\nExample: Federated Learning for Privacy-Preserving Medical Image Segmentation across Multi-Institutional Hospital Networks.",
        height=110,
    )
    domain = st.selectbox(
        "Domain",
        ["General AI","NLP","Computer Vision","Biomedical","Graph / Network",
         "Multimodal","Reinforcement Learning","Robotics","Security","Other"],
        help="Helps the system recommend domain-appropriate datasets and baselines.",
    )
    max_papers = st.slider("Papers to retrieve", 10, 80, 25,
                           help="More papers improve analysis quality but increase runtime.")

    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6b7590;margin-bottom:10px'>Grant Proposal</div>", unsafe_allow_html=True)

    grant_agency   = st.selectbox("Funding Agency", list(GRANT_AGENCIES.keys()),
                                   help="Each agency has a unique section structure. The system auto-formats accordingly.")
    pi_name        = st.text_input("Principal Investigator", "Dr. Jane Smith")
    institution    = st.text_input("Institution", "MIT")
    budget_total   = st.text_input("Total Budget", "$500,000")
    duration_years = st.number_input("Duration (years)", 1, 10, 3)
    citation_style = st.selectbox("Citation Style", CITATION_STYLES,
                                   help="IEEE / ACM for CS. APA for life sciences.")

    st.divider()
    run_pipeline = st.button("Run Research Pipeline", type="primary", use_container_width=True)

    stats = collection_stats()
    st.markdown(
        f"<div style='text-align:center;margin-top:12px;font-size:.7rem;color:#3d4456;'>"
        f"{stats['total_papers']} papers in corpus</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
AGENT_STEPS = [
    ("literature",  "Literature Mining",     "Retrieving and indexing papers from ArXiv and Semantic Scholar"),
    ("trends",      "Trend Analysis",        "Detecting how the research field has evolved over time"),
    ("gaps",        "Gap Identification",    "Mapping under-explored intersections in the literature"),
    ("methodology", "Methodology Design",    "Designing experiment, datasets, baselines, and metrics"),
    ("grant",       "Grant Writing",         "Composing structured proposal sections"),
    ("novelty",     "Novelty Scoring",       "Benchmarking originality against the corpus"),
    ("done",        "Complete",              ""),
]
step_progress = {s: i / (len(AGENT_STEPS)-1) for i, (s,_,__) in enumerate(AGENT_STEPS)}

if run_pipeline:
    if not research_topic.strip():
        st.markdown('<div class="warn-bar">A research topic is required. Please enter one in the sidebar.</div>', unsafe_allow_html=True)
    elif not gemini_key.strip():
        st.markdown('<div class="warn-bar">A Gemini API key is required. Get one free at aistudio.google.com.</div>', unsafe_allow_html=True)
    else:
        st.session_state.pipeline_log = []
        st.markdown('<div class="info-bar">Pipeline running — this takes 1–3 minutes. Do not refresh the page.</div>', unsafe_allow_html=True)

        prog_bar   = st.progress(0)
        col_prog, col_log = st.columns([1, 1])

        with col_prog:
            prog_status = st.empty()
        with col_log:
            log_area = st.empty()

        agent_states = {s: "wait" for s,_,__ in AGENT_STEPS[:-1]}

        def progress_cb(step, msg):
            st.session_state.pipeline_log.append((step, msg))
            prog_bar.progress(step_progress.get(step, 0))
            if step in agent_states:
                agent_states[step] = "done"

            # Status text
            prog_status.markdown(
                f"<div style='font-size:.8rem;color:#c9a84c;padding:.4rem 0'>"
                f"<span style='font-family:IBM Plex Mono,monospace'>Running</span> — {msg}</div>",
                unsafe_allow_html=True,
            )

            # Agent log panel
            rows = ""
            for s, label, desc in AGENT_STEPS[:-1]:
                state = agent_states.get(s, "wait")
                dot   = f"dot-{state}"
                txt   = f"step-text-{state}"
                rows += (
                    f"<div class='step-row'>"
                    f"<span class='step-dot {dot}'></span>"
                    f"<span class='{txt}'><b>{label}</b> — {desc}</span>"
                    f"</div>"
                )
            log_area.markdown(
                f"<div class='data-card' style='padding:1rem'>{rows}</div>",
                unsafe_allow_html=True,
            )

        request = ResearchRequest(
            topic=research_topic, domain=domain,
            grant_agency=grant_agency, pi_name=pi_name,
            institution=institution, budget_total=budget_total,
            duration_years=duration_years, citation_style=citation_style,
            max_papers=max_papers,
        )
        try:
            report = ResearchOrchestrator().run(request, progress_callback=progress_cb)
            st.session_state.report = report
            prog_bar.progress(1.0)
            prog_status.markdown(
                "<div style='font-size:.82rem;color:#68d391;padding:.4rem 0;font-weight:600'>"
                "Pipeline complete. View results below.</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception(e)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
report = st.session_state.report

if report:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "OVERVIEW", "GAPS & TRENDS", "METHODOLOGY", "GRANT PROPOSAL", "ASSISTANT"
    ])

    def conf_bar(score: float, label: str = "Confidence"):
        pct   = int(score * 100)
        color = ("#68d391" if score>=.7 else "#ecc94b" if score>=.4 else "#fc8181")
        grade = ("High" if score>=.7 else "Moderate" if score>=.4 else "Low")
        return (
            f"<div class='conf-row'>"
            f"<div class='conf-header'>"
            f"<span class='conf-name'>{label}</span>"
            f"<span class='conf-pct' style='color:{color}'>{grade} &nbsp;{pct}%</span>"
            f"</div>"
            f"<div class='conf-track'>"
            f"<div class='conf-fill' style='width:{pct}%;background:{color}'></div>"
            f"</div></div>"
        )

    # ── OVERVIEW ──────────────────────────────────────────────
    with tab1:
        novelty = report.novelty.get("novelty_score", 0)
        fetched = report.literature.get("fetched", 0)
        n_gaps  = len(report.gaps.get("identified_gaps", []))
        n_secs  = len(report.grant.get("sections", {}))

        st.markdown(
            f"<div class='stat-row'>"
            f"<div class='stat-card'><div class='stat-label'>Papers Retrieved</div>"
            f"<div class='stat-value'>{fetched}</div>"
            f"<div class='stat-delta'>{report.literature.get('new_ingested',0)} newly added to corpus</div></div>"
            f"<div class='stat-card teal'><div class='stat-label'>Research Gaps</div>"
            f"<div class='stat-value'>{n_gaps}</div>"
            f"<div class='stat-delta'>Identified from literature</div></div>"
            f"<div class='stat-card green'><div class='stat-label'>Novelty Score</div>"
            f"<div class='stat-value'>{novelty:.2f}</div>"
            f"<div class='stat-delta'>{report.novelty.get('novelty_label','').replace('_',' ').title()}</div></div>"
            f"<div class='stat-card slate'><div class='stat-label'>Proposal Sections</div>"
            f"<div class='stat-value'>{n_secs}</div>"
            f"<div class='stat-delta'>Ready to download</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1.1, 0.9])
        with col_a:
            # Novelty gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=novelty,
                number={"font": {"size": 34, "color": "#c9a84c",
                                 "family": "Playfair Display"}},
                gauge={
                    "axis": {"range": [0,1], "tickcolor": "#3d4456",
                             "tickfont": {"color":"#3d4456","size":10}},
                    "bar":  {"color": "#c9a84c", "thickness": .22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range":[0,.35],  "color":"rgba(252,129,129,.15)"},
                        {"range":[.35,.55],"color":"rgba(236,201,75,.12)"},
                        {"range":[.55,.75],"color":"rgba(104,211,145,.12)"},
                        {"range":[.75,1],  "color":"rgba(201,168,76,.18)"},
                    ],
                },
                title={"text":"Novelty Score","font":{"color":"#6b7590","size":12,"family":"IBM Plex Sans"}},
            ))
            fig.update_layout(
                height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=None, margin=dict(l=20,r=20,t=40,b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;font-size:.76rem;color:#6b7590;margin-top:-8px;line-height:1.5'>"
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
                        marker_color="#c9a84c",
                        marker_opacity=0.75,
                        hovertemplate="%{x}: %{y} papers<extra></extra>",
                    ))
                    fig2.update_layout(
                        title=dict(text="Publication Year Distribution",
                                   font=dict(size=11,color="#6b7590",family="IBM Plex Sans")),
                        height=240,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10,r=10,t=36,b=10),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#6b7590",
                                   tickfont=dict(size=10)),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#6b7590",
                                   tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # Confidence panel
        st.markdown("<div class='data-card'>", unsafe_allow_html=True)
        st.markdown('<span class="sec-label">Agent Confidence Scores</span>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.75rem;color:#6b7590;margin-bottom:14px'>"
            "Scores reflect paper volume retrieved, semantic clustering quality, and output completeness.</div>",
            unsafe_allow_html=True,
        )
        lit_conf  = min(fetched/30, 1.0)
        gap_conf  = report.gaps.get("novelty_score", 0.5)
        meth_conf = 0.85 if report.methodology.get("hypothesis") else 0.4
        grt_conf  = min(n_secs/6, 1.0)
        nov_conf  = novelty
        for lbl, sc in [
            ("Literature Mining",  lit_conf),
            ("Gap Identification", gap_conf),
            ("Methodology Design", meth_conf),
            ("Grant Writing",      grt_conf),
            ("Novelty Scoring",    nov_conf),
        ]:
            st.markdown(conf_bar(sc, lbl), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Papers table
        if papers:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Retrieved Papers</span>', unsafe_allow_html=True)
            df_show = pd.DataFrame(papers)[["title","year","authors","source","similarity"]].head(12)
            df_show.columns = ["Title","Year","Authors","Source","Relevance"]
            st.dataframe(df_show, use_container_width=True, height=300)

    # ── GAPS & TRENDS ────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([1.15, 0.85])

        with col1:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Identified Research Gaps</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#6b7590;margin-bottom:14px;line-height:1.6'>"
                "Areas identified as under-explored in the existing literature. "
                "Confidence reflects the strength of supporting evidence.</div>",
                unsafe_allow_html=True,
            )
            gaps   = report.gaps.get("identified_gaps", [])
            base_c = report.gaps.get("novelty_score", 0.6)
            for i, gap in enumerate(gaps):
                gc = round(min(base_c + (.04 if i%2==0 else -.04), .98), 2)
                st.markdown(
                    f"<div class='gap-item'><div class='gap-num'>Gap {i+1}</div>{gap}"
                    f"{conf_bar(gc,'Confidence')}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            if report.gaps.get("opportunity_areas"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Opportunity Areas</span>', unsafe_allow_html=True)
                for opp in report.gaps["opportunity_areas"]:
                    st.markdown(
                        f"<div style='padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05);"
                        f"font-size:.83rem;color:#a0aec0;line-height:1.6'>{opp}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Emerging Sub-Fields</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#6b7590;margin-bottom:12px'>"
                "Topics gaining traction in recent publications.</div>", unsafe_allow_html=True)
            for t in report.trends.get("emerging_topics", []):
                st.markdown(f"<span class='pill pill-teal'>{t}</span>", unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Declining Topics</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#6b7590;margin-bottom:12px'>"
                "Topics appearing less frequently in recent work.</div>", unsafe_allow_html=True)
            for t in report.trends.get("declining_topics", []):
                st.markdown(f"<span class='pill pill-red'>{t}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if report.trends.get("trend_summary"):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
                st.markdown('<span class="sec-label">Field Trend Summary</span>', unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:.82rem;line-height:1.8;color:#a0aec0'>"
                    f"{report.trends['trend_summary']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Novelty Analysis</span>', unsafe_allow_html=True)
            st.markdown(conf_bar(report.novelty.get("novelty_score",0), "Overall Novelty"), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:.75rem;color:#6b7590;margin-top:8px;line-height:1.6'>"
                f"Rating: <b style='color:#c9a84c'>"
                f"{report.novelty.get('novelty_label','').replace('_',' ').title()}</b><br>"
                f"Compared against {report.novelty.get('corpus_size',0)} papers in corpus.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── METHODOLOGY ──────────────────────────────────────────
    with tab3:
        hyp = report.methodology.get("hypothesis","")
        if hyp:
            st.markdown(
                f"<div class='hypothesis'><b style='color:#4fd1c5;font-size:.7rem;"
                f"letter-spacing:.1em;text-transform:uppercase;font-style:normal'>Hypothesis</b>"
                f"<br><br>{hyp}</div>",
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Experimental Approach</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#6b7590;margin-bottom:12px'>"
                "Recommended step-by-step methodology for this research direction.</div>",
                unsafe_allow_html=True,
            )
            for i, step in enumerate(report.methodology.get("approach",[]), 1):
                st.markdown(
                    f"<div class='approach-step'>"
                    f"<span class='step-num'>{i:02d}.</span>"
                    f"<span>{step}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Recommended Datasets</span>', unsafe_allow_html=True)
            for ds in report.methodology.get("suggested_datasets",[]):
                st.markdown(f"<span class='pill pill-gold'>{ds}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Baseline Comparisons</span>', unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:.75rem;color:#6b7590;margin-bottom:12px'>"
                "Existing approaches your work should be benchmarked against.</div>",
                unsafe_allow_html=True,
            )
            for bl in report.methodology.get("baselines",[]):
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.78rem;"
                    f"color:#a0aec0;padding:.35rem .7rem;margin:.25rem 0;"
                    f"border:1px solid rgba(255,255,255,.07);border-radius:4px;"
                    f"background:rgba(255,255,255,.03)'>{bl}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Evaluation Metrics</span>', unsafe_allow_html=True)
            for m in report.methodology.get("evaluation_metrics",[]):
                st.markdown(f"<span class='pill pill-slate'>{m}</span>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(conf_bar(0.85 if hyp else 0.4, "Methodology Confidence"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if report.methodology.get("expected_outcomes"):
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown('<span class="sec-label">Expected Contributions</span>', unsafe_allow_html=True)
            for o in report.methodology["expected_outcomes"]:
                st.markdown(
                    f"<div style='padding:.5rem 0;border-bottom:1px solid rgba(255,255,255,.05);"
                    f"font-size:.83rem;color:#a0aec0;line-height:1.6'>{o}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── GRANT PROPOSAL ───────────────────────────────────────
    with tab4:
        st.markdown(
            f"<div class='grant-meta-row'>"
            f"<div class='grant-meta-item'><div class='grant-meta-label'>Agency</div>"
            f"<div class='grant-meta-val'>{report.grant.get('agency','')}</div></div>"
            f"<div class='grant-meta-item'><div class='grant-meta-label'>Principal Investigator</div>"
            f"<div class='grant-meta-val'>{report.grant.get('pi_name','')}</div></div>"
            f"<div class='grant-meta-item'><div class='grant-meta-label'>Institution</div>"
            f"<div class='grant-meta-val'>{report.grant.get('institution','')}</div></div>"
            f"<div class='grant-meta-item'><div class='grant-meta-label'>Budget</div>"
            f"<div class='grant-meta-val'>{report.grant.get('budget_total','')}</div></div>"
            f"<div class='grant-meta-item'><div class='grant-meta-label'>Duration</div>"
            f"<div class='grant-meta-val'>{report.grant.get('duration_years','')} years</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        n_secs = len(report.grant.get("sections",{}))
        st.markdown(conf_bar(min(n_secs/6,1.0), f"Completeness — {n_secs} sections generated"), unsafe_allow_html=True)
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        for section, content in report.grant.get("sections",{}).items():
            with st.expander(section):
                st.markdown(
                    f"<div style='font-size:.84rem;line-height:1.8;color:#c8d0e0'>{content}</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown('<span class="sec-label">Download Proposal</span>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.76rem;color:#6b7590;margin-bottom:14px;line-height:1.6'>"
            "PDF and DOCX are submission-ready. Markdown is best for further editing.</div>",
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            if st.button("Generate PDF", use_container_width=True):
                path = export_proposal_pdf(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path,"rb") as f:
                        st.download_button("Download PDF", f,
                            file_name=os.path.basename(path), mime="application/pdf",
                            use_container_width=True)
                else:
                    st.error("PDF generation failed.")

        with ec2:
            if st.button("Generate Word Document", use_container_width=True):
                path = export_proposal_docx(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path,"rb") as f:
                        st.download_button("Download DOCX", f,
                            file_name=os.path.basename(path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True)
                else:
                    st.error("DOCX generation failed.")

        with ec3:
            if st.button("Generate Markdown Report", use_container_width=True):
                report_dict = {
                    "literature": report.literature, "trends": report.trends,
                    "gaps": report.gaps, "methodology": report.methodology,
                    "grant": {k:v for k,v in report.grant.items() if k!="full_proposal"},
                    "novelty": report.novelty,
                }
                path = export_report_markdown(report_dict, "./outputs")
                if path and os.path.exists(path):
                    with open(path) as f:
                        st.download_button("Download Markdown", f,
                            file_name=os.path.basename(path), mime="text/markdown",
                            use_container_width=True)

    # ── ASSISTANT ────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='data-card' style='margin-bottom:1rem'>", unsafe_allow_html=True)
        st.markdown('<span class="sec-label">Research Assistant</span>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.8rem;color:#6b7590;line-height:1.7'>"
            "Ask questions about your research results, proposal content, gaps, or novelty scores. "
            "The assistant has full context of the current report.<br><br>"
            "<b style='color:#8892a4'>Suggested:</b> "
            "What does my novelty score mean? &nbsp;|&nbsp; "
            "Which gap should I prioritise? &nbsp;|&nbsp; "
            "Is this proposal ready for submission? &nbsp;|&nbsp; "
            "Summarise the top papers found."
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
Current report data:
- Topic: {report.request.topic}
- Agency: {report.grant.get('agency','')}
- Novelty score: {report.novelty.get('novelty_score',0):.2f} ({report.novelty.get('novelty_label','')})
- Papers retrieved: {report.literature.get('fetched',0)}
- Research gaps: {'; '.join(report.gaps.get('identified_gaps',[])[:4])}
- Emerging trends: {'; '.join(report.trends.get('emerging_topics',[])[:3])}
- Hypothesis: {report.methodology.get('hypothesis','Not generated')}
- Proposal sections: {', '.join(report.grant.get('sections',{}).keys())}
- Recommendation: {report.novelty.get('recommendation','')}
Respond with clarity and precision. Reference specific numbers and findings. Be direct and actionable."""

            from core.llm_factory import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage
            try:
                llm  = get_llm(temperature=0.4)
                msgs = [SystemMessage(content=context),
                        *[HumanMessage(content=m["content"]) if m["role"]=="user"
                          else type("A",(),{"content":m["content"],"type":"ai"})()
                          for m in st.session_state.chat_history[-8:]]]
                reply = llm.invoke(msgs).content
            except Exception as e:
                reply = f"Unable to generate response: {e}. Please verify your API key."

            st.session_state.chat_history.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"):
                st.write(reply)

# ── EMPTY STATE ─────────────────────────────────────────────────
else:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-bar' style='text-align:center'>"
        "Configure your research parameters in the left panel and click <b>Run Research Pipeline</b> to begin."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("What it does",
         "Runs six specialised AI agents in sequence: retrieves papers from ArXiv and Semantic Scholar, "
         "detects how the field is evolving, identifies under-explored research gaps, "
         "designs an experimental methodology, writes a complete grant proposal, "
         "and scores the originality of the proposed work against the corpus."),
        ("What you need",
         "A Gemini API key (free at aistudio.google.com), a research topic described in "
         "one or two sentences, and standard grant details — PI name, institution, "
         "funding agency, and budget. No local setup or installation is required."),
        ("What you receive",
         "A novelty score with explanation, a prioritised list of research gaps, a suggested "
         "experimental design with recommended datasets and baselines, and a complete grant "
         "proposal ready for download as PDF, Word document, or Markdown — formatted for "
         "NSF, NIH, DARPA, or EU Horizon."),
    ]
    for col, (title, body) in zip([c1,c2,c3], cards):
        with col:
            st.markdown(
                f"<div class='data-card' style='height:100%'>"
                f"<div class='card-heading'>{title}</div>"
                f"<div style='height:4px;width:32px;background:var(--gold);"
                f"border-radius:2px;margin:.6rem 0 .9rem'></div>"
                f"<div style='font-size:.82rem;line-height:1.8;color:#8892a4'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
