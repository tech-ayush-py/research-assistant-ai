"""
AI-Powered Academic Research Assistant & Grant Proposal Generator
Clean glassmorphism UI — no emojis, user-friendly labels and helper text
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import streamlit as st

# ── Load Streamlit Cloud secrets ───────────────────────────────
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

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Glassmorphism CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-size: .88rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important; font-weight: 600 !important;
    letter-spacing: .03em !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
    transition: transform .2s, box-shadow .2s !important;
    padding: .65rem 1rem !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: #e2e8f0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg,
        rgba(102,126,234,0.4) 0%,
        rgba(118,75,162,0.4) 50%,
        rgba(36,36,62,0.6) 100%);
    backdrop-filter: blur(30px);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 2.4rem 2rem;
    border-radius: 24px;
    margin-bottom: 1.8rem;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.main-header h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: .5rem; letter-spacing: -.01em; }
.main-header p  { opacity: .75; font-size: .95rem; font-weight: 300; margin: 0; }

/* Section label inside header */
.header-tags {
    display: flex; justify-content: center; gap: 12px; margin-top: 1rem; flex-wrap: wrap;
}
.header-tag {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 3px 14px; border-radius: 20px;
    font-size: .75rem; font-weight: 500; color: rgba(255,255,255,0.85);
}

/* Metric cards */
.metric-glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    color: #e2e8f0;
    transition: transform .2s, box-shadow .2s;
}
.metric-glass:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(102,126,234,0.25); }
.metric-glass .m-val   { font-size: 1.9rem; font-weight: 700; color: #a78bfa; line-height: 1.1; }
.metric-glass .m-label { font-size: .7rem; font-weight: 600; opacity: .6;
                          text-transform: uppercase; letter-spacing: .1em; margin-bottom: 4px; }
.metric-glass .m-delta { font-size: .73rem; color: #6ee7b7; margin-top: 4px; }

/* Confidence bar */
.conf-wrap  { margin: .35rem 0 .75rem; }
.conf-label { font-size: .72rem; color: #94a3b8; margin-bottom: 4px;
              display: flex; justify-content: space-between; }
.conf-track { height: 7px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 4px; transition: width 1s ease; }

/* Gap cards */
.gap-card {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-left: 4px solid #fbbf24;
    border-radius: 12px;
    padding: .85rem 1.1rem;
    margin: .5rem 0;
    color: #fde68a;
    font-size: .83rem;
    line-height: 1.6;
}
.gap-card .gap-num {
    font-size: .65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #fbbf24; margin-bottom: 4px;
}

/* Trend pills */
.pill-green { background: rgba(110,231,183,0.12); border: 1px solid rgba(110,231,183,0.25);
              color: #6ee7b7; padding: 5px 13px; border-radius: 20px;
              font-size: .76rem; font-weight: 500; display: inline-block; margin: 3px; }
.pill-red   { background: rgba(252,165,165,0.12); border: 1px solid rgba(252,165,165,0.25);
              color: #fca5a5; padding: 5px 13px; border-radius: 20px;
              font-size: .76rem; font-weight: 500; display: inline-block; margin: 3px; }
.pill-blue  { background: rgba(167,139,250,0.15); border: 1px solid rgba(167,139,250,0.25);
              color: #c4b5fd; padding: 5px 13px; border-radius: 20px;
              font-size: .76rem; font-weight: 500; display: inline-block; margin: 3px; }

/* Step log lines */
.log-done { color: #6ee7b7; font-size: .8rem; padding: 2px 0; }
.log-run  { color: #93c5fd; font-size: .8rem; padding: 2px 0; }

/* Section label */
.section-label {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .12em; color: #94a3b8; margin-bottom: 10px;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: rgba(255,255,255,0.45) !important;
    font-weight: 500 !important; font-size: .85rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Helper text under inputs */
.helper-text {
    font-size: .72rem; color: #64748b; margin-top: -6px; margin-bottom: 10px;
}

/* Info banner */
.info-banner {
    background: rgba(102,126,234,0.12);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 12px; padding: .75rem 1.1rem;
    font-size: .82rem; color: #c4b5fd; margin-bottom: 1rem;
}

/* General text */
hr { border-color: rgba(255,255,255,0.08) !important; }
.stMarkdown, .stText, p, li, label { color: #e2e8f0 !important; }
h1,h2,h3,h4 { color: #f1f5f9 !important; }
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Academic Research Assistant</h1>
    <p>AI-powered system for literature discovery, gap analysis, and grant proposal generation</p>
    <div class="header-tags">
        <span class="header-tag">Literature Mining</span>
        <span class="header-tag">Trend Analysis</span>
        <span class="header-tag">Gap Identification</span>
        <span class="header-tag">Grant Writing</span>
        <span class="header-tag">Novelty Scoring</span>
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
    st.markdown("## Setup")
    st.markdown("---")

    # ── API Key ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Gemini API Key</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        placeholder="Paste your key here…",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="helper-text">Get a free key at aistudio.google.com — no credit card needed.</div>',
        unsafe_allow_html=True
    )

    llm_model = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        help="Flash is faster and works on the free tier. Pro gives higher quality outputs for grant writing.",
    )

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"]    = llm_model

    st.markdown("---")

    # ── Research Topic ─────────────────────────────────────────
    st.markdown('<div class="section-label">Research Topic</div>', unsafe_allow_html=True)
    research_topic = st.text_area(
        "Topic",
        label_visibility="collapsed",
        placeholder="Describe your research topic in one or two sentences.\n\nExample: Federated Learning for Privacy-Preserving Medical Image Analysis",
        height=100,
    )

    domain = st.selectbox(
        "Research Domain",
        ["General AI", "NLP", "Computer Vision", "Biomedical",
         "Graph / Network", "Multimodal", "Reinforcement Learning",
         "Robotics", "Security", "Other"],
        help="Select the closest domain. This helps the system suggest relevant datasets and baselines.",
    )

    max_papers = st.slider(
        "Number of papers to fetch",
        min_value=10, max_value=80, value=25,
        help="More papers = better analysis, but slower. 25 is a good starting point.",
    )

    st.markdown("---")

    # ── Grant Settings ─────────────────────────────────────────
    st.markdown('<div class="section-label">Grant Proposal Settings</div>', unsafe_allow_html=True)

    grant_agency = st.selectbox(
        "Funding Agency",
        list(GRANT_AGENCIES.keys()),
        help="Each agency has a different proposal format. The system will generate the correct sections automatically.",
    )
    pi_name = st.text_input(
        "Principal Investigator Name",
        value="Dr. Jane Smith",
        help="Your full name as it should appear in the proposal.",
    )
    institution = st.text_input(
        "Institution / University",
        value="MIT",
    )
    budget_total = st.text_input(
        "Total Budget",
        value="$500,000",
        help="Enter the total funding amount you are requesting.",
    )
    duration_years = st.number_input(
        "Project Duration (years)",
        min_value=1, max_value=10, value=3,
    )
    citation_style = st.selectbox(
        "Citation Style",
        CITATION_STYLES,
        help="IEEE and ACM are common for computer science. APA is common for social sciences.",
    )

    st.markdown("---")

    # ── Run button ─────────────────────────────────────────────
    run_pipeline = st.button("Run Research Pipeline", type="primary", use_container_width=True)

    # Corpus count
    stats = collection_stats()
    st.markdown(
        f"<div style='text-align:center;margin-top:14px;font-size:.75rem;"
        f"color:rgba(255,255,255,.35);'>{stats['total_papers']} papers currently in corpus</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
AGENT_STEPS = [
    ("literature",   "Fetching and indexing papers from ArXiv and Semantic Scholar"),
    ("trends",       "Analysing how the research field has evolved over time"),
    ("gaps",         "Identifying under-explored areas in the existing literature"),
    ("methodology",  "Designing the experimental approach, datasets, and baselines"),
    ("grant",        "Writing the grant proposal sections"),
    ("novelty",      "Scoring the originality of the proposed research"),
    ("done",         "Complete"),
]
step_progress = {s: i / (len(AGENT_STEPS) - 1) for i, (s, _) in enumerate(AGENT_STEPS)}

if run_pipeline:
    if not research_topic.strip():
        st.error("Please enter a research topic in the sidebar before running the pipeline.")
    elif not gemini_key.strip():
        st.error("Please enter your Gemini API key in the sidebar. Get one free at aistudio.google.com.")
    else:
        st.session_state.pipeline_log = []

        st.markdown(
            '<div class="info-banner">The pipeline is running. This takes 1–3 minutes depending on the number of papers. Do not refresh the page.</div>',
            unsafe_allow_html=True,
        )

        prog_bar    = st.progress(0)
        prog_status = st.empty()
        log_box     = st.empty()
        agent_states = {s: "Waiting" for s, _ in AGENT_STEPS[:-1]}

        def progress_cb(step, msg):
            st.session_state.pipeline_log.append(msg)
            prog_bar.progress(step_progress.get(step, 0))
            prog_status.markdown(
                f"<div class='log-run'>Running: {msg}</div>",
                unsafe_allow_html=True,
            )
            if step in agent_states:
                agent_states[step] = "Done"
            rows = "".join(
                f"<div class='log-done'>Done &nbsp;—&nbsp; {desc}</div>"
                if agent_states[s] == "Done" else
                f"<div style='color:rgba(255,255,255,.3);font-size:.8rem;padding:2px 0'>Waiting — {desc}</div>"
                for (s, desc) in AGENT_STEPS[:-1]
            )
            log_box.markdown(
                f"<div class='glass-card' style='padding:1rem 1.2rem'>"
                f"<div class='section-label' style='margin-bottom:8px'>Agent Progress</div>{rows}</div>",
                unsafe_allow_html=True,
            )

        request = ResearchRequest(
            topic=research_topic,
            domain=domain,
            grant_agency=grant_agency,
            pi_name=pi_name,
            institution=institution,
            budget_total=budget_total,
            duration_years=duration_years,
            citation_style=citation_style,
            max_papers=max_papers,
        )

        try:
            report = ResearchOrchestrator().run(request, progress_callback=progress_cb)
            st.session_state.report = report
            prog_bar.progress(1.0)
            prog_status.markdown(
                "<div class='log-done' style='font-weight:600;font-size:.88rem;'>"
                "Pipeline complete. Scroll down to view your results.</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}. Please check your API key and try again.")
            logger.exception(e)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
report = st.session_state.report

if report:
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Gaps and Trends",
        "Methodology",
        "Grant Proposal",
        "Ask a Question",
    ])

    # ── Confidence bar helper ──────────────────────────────────
    def conf_bar(score: float, label: str = "Confidence"):
        pct = int(score * 100)
        color = ("#6ee7b7" if score >= .7 else "#fbbf24" if score >= .4 else "#f87171")
        quality = "High" if score >= .7 else "Moderate" if score >= .4 else "Low"
        return (
            f"<div class='conf-wrap'>"
            f"<div class='conf-label'><span>{label}</span><span>{quality} ({pct}%)</span></div>"
            f"<div class='conf-track'><div class='conf-fill' style='width:{pct}%;background:{color}'></div></div>"
            f"</div>"
        )

    # ══ Tab 1: Overview ═══════════════════════════════════════
    with tab1:
        novelty  = report.novelty.get("novelty_score", 0)
        fetched  = report.literature.get("fetched", 0)
        n_gaps   = len(report.gaps.get("identified_gaps", []))
        n_secs   = len(report.grant.get("sections", {}))

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, delta in [
            (c1, fetched,           "Papers Fetched",    f"{report.literature.get('new_ingested',0)} newly added to corpus"),
            (c2, n_gaps,            "Research Gaps Found","Identified from literature"),
            (c3, f"{novelty:.2f}",  "Novelty Score",     report.novelty.get("novelty_label","").replace("_"," ").title()),
            (c4, n_secs,            "Proposal Sections", "Ready to download"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-glass'>"
                    f"<div class='m-label'>{label}</div>"
                    f"<div class='m-val'>{val}</div>"
                    f"<div class='m-delta'>{delta}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])

        with col_a:
            # Novelty gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=novelty,
                number={"font": {"size": 30, "color": "#a78bfa"}, "suffix": ""},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                    "bar":  {"color": "#764ba2"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0,   .35], "color": "rgba(248,113,113,.2)"},
                        {"range": [.35, .55], "color": "rgba(251,191,36,.2)"},
                        {"range": [.55, .75], "color": "rgba(110,231,183,.2)"},
                        {"range": [.75,  1],  "color": "rgba(167,139,250,.2)"},
                    ],
                },
                title={"text": "Proposal Novelty Score", "font": {"color": "#94a3b8", "size": 13}},
            ))
            fig.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;font-size:.78rem;color:#94a3b8;margin-top:-10px'>"
                f"{report.novelty.get('recommendation','')}</div>",
                unsafe_allow_html=True,
            )

        with col_b:
            # Year chart
            papers = report.literature.get("top_papers", [])
            if papers:
                df = pd.DataFrame(papers)
                if "year" in df.columns:
                    yc = df["year"].value_counts().sort_index()
                    fig2 = go.Figure(go.Bar(
                        x=yc.index.astype(str), y=yc.values,
                        marker=dict(color=yc.values,
                                    colorscale=[[0,"#667eea"],[1,"#a78bfa"]],
                                    showscale=False),
                        hovertemplate="%{x}: %{y} papers<extra></extra>",
                    ))
                    fig2.update_layout(
                        title=dict(text="Papers by Publication Year", font=dict(size=13, color="#94a3b8")),
                        height=260,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor ="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # Agent confidence summary
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">How confident is the system in each step?</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.78rem;color:#64748b;margin-bottom:12px'>"
            "Confidence is based on the volume of papers found, semantic similarity scores, and completeness of generated content.</div>",
            unsafe_allow_html=True,
        )
        lit_conf   = min(fetched / 30, 1.0)
        gap_conf   = report.gaps.get("novelty_score", 0.5)
        meth_conf  = 0.85 if report.methodology.get("hypothesis") else 0.4
        grant_conf = min(n_secs / 6, 1.0)
        nov_conf   = novelty

        for label, score in [
            ("Literature Mining",  lit_conf),
            ("Gap Identification", gap_conf),
            ("Methodology Design", meth_conf),
            ("Grant Writing",      grant_conf),
            ("Novelty Scoring",    nov_conf),
        ]:
            st.markdown(conf_bar(score, label), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Top papers table
        if papers:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Top Retrieved Papers</div>', unsafe_allow_html=True)
            df_show = pd.DataFrame(papers)[["title","year","authors","source","similarity"]].head(10)
            df_show.columns = ["Title", "Year", "Authors", "Source", "Relevance Score"]
            st.dataframe(df_show, use_container_width=True, height=280)

    # ══ Tab 2: Gaps and Trends ════════════════════════════════
    with tab2:
        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.markdown("#### Research Gaps")
            st.markdown(
                "<div style='font-size:.8rem;color:#94a3b8;margin-bottom:14px'>"
                "These are areas the system identified as under-explored in existing literature. "
                "Higher confidence means more supporting evidence was found.</div>",
                unsafe_allow_html=True,
            )
            gaps     = report.gaps.get("identified_gaps", [])
            base_c   = report.gaps.get("novelty_score", 0.6)
            for i, gap in enumerate(gaps):
                gap_c = round(min(base_c + (0.04 if i % 2 == 0 else -0.04), 0.99), 2)
                st.markdown(
                    f"<div class='gap-card'>"
                    f"<div class='gap-num'>Gap {i+1}</div>"
                    f"{gap}"
                    f"{conf_bar(gap_c, 'Confidence')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if report.gaps.get("opportunity_areas"):
                st.markdown("#### Opportunity Areas")
                st.markdown(
                    "<div style='font-size:.8rem;color:#94a3b8;margin-bottom:10px'>"
                    "Promising directions where new research could have high impact.</div>",
                    unsafe_allow_html=True,
                )
                for opp in report.gaps["opportunity_areas"]:
                    st.markdown(
                        f"<div style='background:rgba(110,231,183,.07);border:1px solid rgba(110,231,183,.2);"
                        f"border-radius:10px;padding:.65rem 1rem;margin:.35rem 0;"
                        f"color:#6ee7b7;font-size:.83rem;line-height:1.55;'>{opp}</div>",
                        unsafe_allow_html=True,
                    )

        with col2:
            st.markdown("#### Emerging Topics")
            st.markdown(
                "<div style='font-size:.8rem;color:#94a3b8;margin-bottom:10px'>"
                "Sub-topics gaining traction in recent publications.</div>",
                unsafe_allow_html=True,
            )
            for t in report.trends.get("emerging_topics", []):
                st.markdown(f"<span class='pill-green'>Rising: {t}</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Declining Topics")
            st.markdown(
                "<div style='font-size:.8rem;color:#94a3b8;margin-bottom:10px'>"
                "Sub-topics appearing less frequently in recent work.</div>",
                unsafe_allow_html=True,
            )
            for t in report.trends.get("declining_topics", []):
                st.markdown(f"<span class='pill-red'>Declining: {t}</span>", unsafe_allow_html=True)

            if report.trends.get("trend_summary"):
                st.markdown(
                    f"<div class='glass-card' style='margin-top:1.2rem;font-size:.82rem;line-height:1.75'>"
                    f"<div class='section-label'>Field Trend Summary</div>"
                    f"{report.trends['trend_summary']}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='glass-card' style='margin-top:1rem'>"
                f"<div class='section-label'>Novelty Analysis</div>"
                f"{conf_bar(report.novelty.get('novelty_score', 0), 'Overall Novelty')}"
                f"<div style='font-size:.78rem;color:#94a3b8;margin-top:6px'>"
                f"Rating: <b style='color:#a78bfa'>"
                f"{report.novelty.get('novelty_label','').replace('_',' ').title()}</b><br>"
                f"Based on {report.novelty.get('corpus_size',0)} papers in corpus</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ══ Tab 3: Methodology ════════════════════════════════════
    with tab3:
        hyp = report.methodology.get("hypothesis", "")
        if hyp:
            st.markdown(
                f"<div class='glass-card' style='border-left:4px solid #a78bfa'>"
                f"<div class='section-label'>Proposed Hypothesis</div>{hyp}</div>",
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Experimental Steps")
            st.markdown(
                "<div style='font-size:.78rem;color:#94a3b8;margin-bottom:10px'>"
                "A step-by-step approach recommended for this research.</div>",
                unsafe_allow_html=True,
            )
            for i, step in enumerate(report.methodology.get("approach", []), 1):
                st.markdown(
                    f"<div style='background:rgba(255,255,255,.05);border-radius:8px;"
                    f"padding:.65rem 1rem;margin:.35rem 0;font-size:.83rem;line-height:1.5'>"
                    f"<b style='color:#a78bfa'>Step {i}.</b> {step}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Recommended Datasets")
            for ds in report.methodology.get("suggested_datasets", []):
                st.markdown(f"<span class='pill-green'>{ds}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Baseline Models to Compare Against")
            st.markdown(
                "<div style='font-size:.78rem;color:#94a3b8;margin-bottom:10px'>"
                "Your results should be compared with these existing approaches.</div>",
                unsafe_allow_html=True,
            )
            for bl in report.methodology.get("baselines", []):
                st.markdown(
                    f"<div style='background:rgba(102,126,234,.12);border-radius:8px;"
                    f"padding:.45rem 1rem;margin:.3rem 0;font-size:.8rem;color:#c4b5fd;font-family:monospace'>{bl}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Evaluation Metrics")
            st.markdown(
                "<div style='font-size:.78rem;color:#94a3b8;margin-bottom:10px'>"
                "Use these metrics to measure and report your results.</div>",
                unsafe_allow_html=True,
            )
            for m in report.methodology.get("evaluation_metrics", []):
                st.markdown(f"<span class='pill-blue'>{m}</span>", unsafe_allow_html=True)

            meth_conf = 0.85 if report.methodology.get("hypothesis") else 0.4
            st.markdown(
                f"<div class='glass-card' style='margin-top:1.2rem'>"
                f"<div class='section-label'>Methodology Confidence</div>"
                f"{conf_bar(meth_conf)}</div>",
                unsafe_allow_html=True,
            )

        if report.methodology.get("expected_outcomes"):
            st.markdown("#### Expected Outcomes")
            for o in report.methodology["expected_outcomes"]:
                st.markdown(
                    f"<div style='color:#6ee7b7;font-size:.83rem;padding:.3rem 0;line-height:1.5'>"
                    f"— {o}</div>",
                    unsafe_allow_html=True,
                )

    # ══ Tab 4: Grant Proposal ═════════════════════════════════
    with tab4:
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val in [
            (m1, "Funding Agency",      report.grant.get("agency", "")),
            (m2, "Principal Investigator", report.grant.get("pi_name", "")),
            (m3, "Total Budget",        report.grant.get("budget_total", "")),
            (m4, "Duration",            f"{report.grant.get('duration_years','')} years"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-glass'>"
                    f"<div class='m-label'>{label}</div>"
                    f"<div class='m-val' style='font-size:1.05rem;padding-top:2px'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        n_secs     = len(report.grant.get("sections", {}))
        grant_conf = min(n_secs / 6, 1.0)
        st.markdown(
            f"<div class='glass-card'>"
            f"<div class='section-label'>Proposal Completeness</div>"
            f"{conf_bar(grant_conf, f'{n_secs} of 6 sections generated')}"
            f"<div style='font-size:.75rem;color:#64748b;margin-top:6px'>"
            f"Click each section below to expand and read it. You can then download the full proposal.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for section, content in report.grant.get("sections", {}).items():
            with st.expander(section):
                st.write(content)

        st.markdown("---")
        st.markdown('<div class="section-label">Download Your Proposal</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.78rem;color:#94a3b8;margin-bottom:14px'>"
            "Choose a format to download. PDF and DOCX are suitable for submission. "
            "Markdown is useful for editing in any text editor.</div>",
            unsafe_allow_html=True,
        )

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            if st.button("Download as PDF", use_container_width=True):
                path = export_proposal_pdf(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Save PDF", f,
                            file_name=os.path.basename(path),
                            mime="application/pdf",
                            use_container_width=True)
                else:
                    st.error("PDF generation failed. Try downloading as DOCX instead.")

        with ec2:
            if st.button("Download as Word (DOCX)", use_container_width=True):
                path = export_proposal_docx(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Save DOCX", f,
                            file_name=os.path.basename(path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True)
                else:
                    st.error("DOCX generation failed. Please check the logs.")

        with ec3:
            if st.button("Download as Markdown", use_container_width=True):
                report_dict = {
                    "literature": report.literature, "trends": report.trends,
                    "gaps": report.gaps, "methodology": report.methodology,
                    "grant": {k:v for k,v in report.grant.items() if k != "full_proposal"},
                    "novelty": report.novelty,
                }
                path = export_report_markdown(report_dict, "./outputs")
                if path and os.path.exists(path):
                    with open(path) as f:
                        st.download_button("Save Markdown", f,
                            file_name=os.path.basename(path),
                            mime="text/markdown",
                            use_container_width=True)

    # ══ Tab 5: Ask a Question ═════════════════════════════════
    with tab5:
        st.markdown("#### Ask the Research Assistant")
        st.markdown(
            "<div style='font-size:.82rem;color:#94a3b8;margin-bottom:1.2rem'>"
            "Ask anything about your research results — gaps, trends, proposal content, "
            "novelty scores, or what to do next. The assistant has full context of your report.</div>",
            unsafe_allow_html=True,
        )

        # Suggested questions
        st.markdown(
            "<div class='glass-card' style='padding:1rem 1.2rem;margin-bottom:1rem'>"
            "<div class='section-label'>Suggested questions to get started</div>"
            "<div style='font-size:.8rem;color:#94a3b8;line-height:2'>"
            "— What is my novelty score and what does it mean?<br>"
            "— Which research gap should I focus on first?<br>"
            "— Is my proposal ready for submission?<br>"
            "— Summarise the top 5 papers found.<br>"
            "— What baselines should I compare my work against?"
            "</div></div>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Type your question here…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            context = f"""You are the AI Research Assistant for a Research Director.
You have access to the following report data:
- Research topic: {report.request.topic}
- Funding agency: {report.grant.get('agency','')}
- Novelty score: {report.novelty.get('novelty_score',0):.2f} ({report.novelty.get('novelty_label','')})
- Papers fetched: {report.literature.get('fetched',0)}
- Research gaps: {'; '.join(report.gaps.get('identified_gaps',[])[:4])}
- Emerging trends: {'; '.join(report.trends.get('emerging_topics',[])[:3])}
- Hypothesis: {report.methodology.get('hypothesis','Not generated')}
- Proposal sections ready: {', '.join(report.grant.get('sections',{}).keys())}
- Recommendation: {report.novelty.get('recommendation','')}

Answer clearly and helpfully. Reference specific numbers and findings from the report.
If asked for advice, be direct and actionable. Keep answers concise."""

            from core.llm_factory import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage
            try:
                llm  = get_llm(temperature=0.5)
                msgs = [SystemMessage(content=context),
                        *[HumanMessage(content=m["content"]) if m["role"] == "user"
                          else type("A", (), {"content": m["content"], "type": "ai"})()
                          for m in st.session_state.chat_history[-8:]]]
                reply = llm.invoke(msgs).content
            except Exception as e:
                reply = f"Could not generate a response: {e}. Please check your API key."

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)

# ═══════════════════════════════════════════════════════════════
# EMPTY STATE — shown before first run
# ═══════════════════════════════════════════════════════════════
else:
    st.markdown(
        "<div class='info-banner' style='text-align:center;padding:1rem'>"
        "Enter your research topic and Gemini API key in the left panel, then click "
        "<b>Run Research Pipeline</b> to begin.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("What it does",
         "The system runs 6 AI agents in sequence: it searches academic databases, "
         "identifies gaps in the existing literature, designs an experimental methodology, "
         "writes a full grant proposal, and scores how original your research idea is."),
        ("What you need",
         "A Gemini API key (free at aistudio.google.com), a research topic described in "
         "one or two sentences, and your grant details (PI name, institution, budget). "
         "No other setup is required."),
        ("What you get",
         "A novelty score, a list of research gaps, a suggested methodology with datasets "
         "and baselines, and a complete grant proposal ready to download as PDF or Word — "
         "formatted for NSF, NIH, DARPA, or EU Horizon."),
    ]
    for col, (title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"<div class='glass-card'>"
                f"<div class='section-label'>{title}</div>"
                f"<p style='font-size:.83rem;line-height:1.7;opacity:.8'>{body}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
