"""
AI-Powered Academic Research Assistant & Grant Proposal Generator
Main Streamlit Dashboard for the Research Director
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import streamlit as st

# ── Load Streamlit Cloud secrets into env vars (if deployed) ──
try:
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "LLM_PROVIDER", "LLM_MODEL"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # local dev – no secrets file, uses .env instead
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from config.settings import APP_TITLE, GRANT_AGENCIES, CITATION_STYLES
from core.orchestrator import ResearchOrchestrator, ResearchRequest
from core.vector_store import collection_stats
from utils.export import export_proposal_pdf, export_proposal_docx, export_report_markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .agent-card {
        background: #f8f9ff; border: 1px solid #e0e4ff;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border-radius: 8px; padding: 1rem;
        text-align: center;
    }
    .gap-item {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 4px;
    }
    .success-badge {
        background: #d4edda; color: #155724;
        padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🔬 {APP_TITLE}</h1>
    <p style="opacity:0.85; font-size:1.1rem;">
        Multi-Agent AI System for Literature Discovery · Gap Analysis · Grant Generation
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state.report      = None
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR – Configuration
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.header("⚙️ Research Configuration")

    with st.expander("🔑 API Keys", expanded=True):
        gemini_key = st.text_input("Gemini API Key ⭐", type="password",
                                   value=os.getenv("GEMINI_API_KEY",""),
                                   help="Get free key at aistudio.google.com")
        openai_key = st.text_input("OpenAI API Key (optional)", type="password",
                                   value=os.getenv("OPENAI_API_KEY",""))
        anthropic_key = st.text_input("Anthropic API Key (optional)", type="password",
                                      value=os.getenv("ANTHROPIC_API_KEY",""))
        llm_provider = st.selectbox("LLM Provider", ["gemini","openai","anthropic"], index=0)
        model_options = {
            "gemini":    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        }
        llm_model = st.selectbox("Model", model_options.get(llm_provider, ["gemini-1.5-flash"]))

        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["LLM_MODEL"]    = llm_model

    st.divider()
    st.subheader("📋 Research Parameters")

    research_topic = st.text_area(
        "Research Topic *",
        placeholder="e.g., Federated Learning for Privacy-Preserving Medical Image Analysis",
        height=90,
    )
    domain = st.selectbox(
        "Research Domain",
        ["General AI", "NLP", "Computer Vision", "Biomedical", "Graph/Network",
         "Multimodal", "Reinforcement Learning", "Robotics", "Security", "Other"],
    )
    max_papers = st.slider("Max papers to fetch", 10, 100, 30)

    st.divider()
    st.subheader("📝 Grant Proposal Settings")
    grant_agency     = st.selectbox("Funding Agency", list(GRANT_AGENCIES.keys()))
    pi_name          = st.text_input("Principal Investigator", "Dr. Jane Smith")
    institution      = st.text_input("Institution", "MIT")
    budget_total     = st.text_input("Total Budget", "$500,000")
    duration_years   = st.number_input("Duration (years)", 1, 10, 3)
    citation_style   = st.selectbox("Citation Style", CITATION_STYLES)
    custom_gap       = st.text_area("Custom Research Gap (optional)",
                                    placeholder="Override AI-detected gap with your own…",
                                    height=60)

    st.divider()
    run_pipeline = st.button("🚀 Run Research Pipeline", type="primary", use_container_width=True)

    st.divider()
    stats = collection_stats()
    st.metric("📚 Papers in Corpus", stats["total_papers"])

# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
if run_pipeline:
    if not research_topic.strip():
        st.error("⚠️ Please enter a research topic in the sidebar.")
    else:
        st.session_state.pipeline_log = []
        log_container = st.container()

        with log_container:
            st.subheader("🤖 Agent Pipeline Running…")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            STEPS = ["literature", "trends", "gaps", "methodology", "grant", "novelty", "done"]
            step_map = {s: i/(len(STEPS)-1) for i, s in enumerate(STEPS)}

            def progress_cb(step, msg):
                st.session_state.pipeline_log.append(f"[{step.upper()}] {msg}")
                progress_bar.progress(step_map.get(step, 0))
                status_text.info(f"**{step.upper()}** — {msg}")

            request = ResearchRequest(
                topic          = research_topic,
                domain         = domain,
                grant_agency   = grant_agency,
                pi_name        = pi_name,
                institution    = institution,
                budget_total   = budget_total,
                duration_years = duration_years,
                citation_style = citation_style,
                max_papers     = max_papers,
                custom_gap     = custom_gap.strip() or None,
            )

            with st.spinner("Multi-agent pipeline executing…"):
                try:
                    orchestrator = ResearchOrchestrator()
                    report = orchestrator.run(request, progress_callback=progress_cb)
                    st.session_state.report = report
                    status_text.success("✅ Pipeline complete!")
                    progress_bar.progress(1.0)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    logger.exception(e)

# ═══════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════
report = st.session_state.report

if report:
    st.divider()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "📚 Literature",
        "📈 Trends",
        "🔍 Gaps",
        "⚗️ Methodology",
        "📄 Grant Proposal",
        "💬 Chat Assistant",
    ])

    # ── Tab 1: Overview ─────────────────────────────────────
    with tab1:
        st.subheader("🎯 Research Intelligence Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers Fetched",     report.literature.get("fetched", 0))
        c2.metric("New Ingested",        report.literature.get("new_ingested", 0))
        c3.metric("Research Gaps Found", len(report.gaps.get("identified_gaps", [])))
        c4.metric("Novelty Score",
                  f"{report.novelty.get('novelty_score', 0):.2f} / 1.0",
                  delta=report.novelty.get("novelty_label", ""))

        st.divider()
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("🏆 Novelty Gauge")
            score = report.novelty.get("novelty_score", 0)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                number={"suffix": "  novelty", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#764ba2"},
                    "steps": [
                        {"range": [0, 0.35],  "color": "#ff6b6b"},
                        {"range": [0.35, 0.55], "color": "#ffd93d"},
                        {"range": [0.55, 0.75], "color": "#6bcb77"},
                        {"range": [0.75, 1.0],  "color": "#4d96ff"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": score},
                },
                title={"text": "Proposal Novelty Score"},
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("📋 Pipeline Execution Log")
            for log in st.session_state.pipeline_log:
                icon = "✅" if "complete" in log.lower() or "done" in log.lower() else "⚙️"
                st.markdown(f"{icon} `{log}`")

        if report.errors:
            with st.expander("⚠️ Pipeline Warnings"):
                for err in report.errors:
                    st.warning(err)

    # ── Tab 2: Literature ───────────────────────────────────
    with tab2:
        st.subheader("📚 Top Retrieved Papers")
        papers = report.literature.get("top_papers", [])
        if papers:
            df = pd.DataFrame(papers)[["title", "year", "authors", "source", "similarity"]]
            df["similarity"] = df["similarity"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df, use_container_width=True, height=400)

            # Year distribution
            st.subheader("📅 Publication Year Distribution")
            year_counts = df["year"].value_counts().sort_index()
            fig = px.bar(x=year_counts.index, y=year_counts.values,
                         labels={"x": "Year", "y": "Paper Count"},
                         color=year_counts.values,
                         color_continuous_scale="Viridis")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No papers retrieved yet. Run the pipeline with a research topic.")

    # ── Tab 3: Trends ───────────────────────────────────────
    with tab3:
        st.subheader("📈 Research Trend Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🚀 Emerging Topics")
            for t in report.trends.get("emerging_topics", []):
                st.markdown(f'<div class="agent-card">🔼 {t}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 📉 Declining Topics")
            for t in report.trends.get("declining_topics", []):
                st.markdown(f'<div class="agent-card">🔽 {t}</div>', unsafe_allow_html=True)

        if report.trends.get("trend_summary"):
            st.subheader("🧠 Trend Summary")
            st.markdown(report.trends["trend_summary"])

        year_kw = report.trends.get("year_keyword_distribution", {})
        if year_kw:
            st.subheader("🗓️ Year × Keyword Map")
            rows = []
            for year, kws in year_kw.items():
                for i, kw in enumerate(kws):
                    rows.append({"Year": year, "Keyword": kw, "Rank": i+1})
            if rows:
                df_kw = pd.DataFrame(rows)
                fig = px.scatter(df_kw, x="Year", y="Keyword", size="Rank",
                                 color="Keyword", height=350)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Gaps ─────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Research Gap Analysis")

        st.markdown(f"**Topic:** {report.gaps.get('research_topic','')}")
        st.markdown(f"**Papers Analysed:** {report.gaps.get('papers_analysed', 0)}  |  "
                    f"**Novelty Score:** {report.gaps.get('novelty_score', 0):.3f}")

        st.markdown("### 🕳️ Identified Research Gaps")
        for gap in report.gaps.get("identified_gaps", []):
            st.markdown(f'<div class="gap-item">💡 {gap}</div>', unsafe_allow_html=True)

        st.markdown("### 🌟 Opportunity Areas")
        for opp in report.gaps.get("opportunity_areas", []):
            st.markdown(f'<div class="agent-card">🚀 {opp}</div>', unsafe_allow_html=True)

        if report.gaps.get("gap_reasoning"):
            with st.expander("📖 Gap Analysis Methodology"):
                st.write(report.gaps["gap_reasoning"])

        top_cited = report.gaps.get("top_cited_papers", [])
        if top_cited:
            st.subheader("🏅 Most Relevant Existing Papers")
            df_cited = pd.DataFrame(top_cited)
            st.dataframe(df_cited, use_container_width=True)

    # ── Tab 5: Methodology ──────────────────────────────────
    with tab5:
        st.subheader("⚗️ Suggested Research Methodology")

        if report.methodology.get("hypothesis"):
            st.info(f"**Hypothesis:** {report.methodology['hypothesis']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🪜 Experimental Approach")
            for i, step in enumerate(report.methodology.get("approach", []), 1):
                st.markdown(f"**{i}.** {step}")

            st.markdown("### 📦 Suggested Datasets")
            for ds in report.methodology.get("suggested_datasets", []):
                st.markdown(f"- {ds}")

        with col2:
            st.markdown("### 🤖 Baseline Models")
            for bl in report.methodology.get("baselines", []):
                st.markdown(f"- `{bl}`")

            st.markdown("### 📐 Evaluation Metrics")
            for m in report.methodology.get("evaluation_metrics", []):
                st.markdown(f"- {m}")

        if report.methodology.get("timeline_weeks"):
            st.subheader("🗓️ Research Timeline")
            timeline = report.methodology["timeline_weeks"]
            df_timeline = pd.DataFrame(
                {"Phase": range(1, len(timeline)+1), "Description": timeline}
            )
            st.dataframe(df_timeline, use_container_width=True)

        if report.methodology.get("expected_outcomes"):
            st.subheader("🎯 Expected Outcomes")
            for o in report.methodology["expected_outcomes"]:
                st.markdown(f"✅ {o}")

    # ── Tab 6: Grant Proposal ───────────────────────────────
    with tab6:
        st.subheader(f"📄 {report.grant.get('agency','')} Grant Proposal")

        # Metadata
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agency",      report.grant.get("agency", ""))
        m2.metric("PI",          report.grant.get("pi_name", ""))
        m3.metric("Budget",      report.grant.get("budget_total", ""))
        m4.metric("Duration",    f"{report.grant.get('duration_years','')} years")

        st.markdown("---")

        # Display each section
        for section, content in report.grant.get("sections", {}).items():
            with st.expander(f"📌 {section}", expanded=True):
                st.write(content)

        st.markdown("---")
        st.subheader("📥 Export Options")
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            if st.button("⬇️ Download PDF", use_container_width=True):
                path = export_proposal_pdf(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Save PDF", f, file_name=os.path.basename(path),
                                           mime="application/pdf", use_container_width=True)
                else:
                    st.error("PDF export failed. Check logs.")

        with ec2:
            if st.button("⬇️ Download DOCX", use_container_width=True):
                path = export_proposal_docx(report.grant, "./outputs")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Save DOCX", f, file_name=os.path.basename(path),
                                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                           use_container_width=True)
                else:
                    st.error("DOCX export failed. Check logs.")

        with ec3:
            if st.button("⬇️ Full Report (Markdown)", use_container_width=True):
                import dataclasses
                report_dict = {
                    "literature": report.literature,
                    "trends": report.trends,
                    "gaps": report.gaps,
                    "methodology": report.methodology,
                    "grant": {k: v for k, v in report.grant.items() if k != "full_proposal"},
                    "novelty": report.novelty,
                }
                path = export_report_markdown(report_dict, "./outputs")
                if path and os.path.exists(path):
                    with open(path) as f:
                        st.download_button("Save Markdown", f, file_name=os.path.basename(path),
                                           mime="text/markdown", use_container_width=True)

        with st.expander("👁️ Full Proposal Text"):
            st.text(report.grant.get("full_proposal", ""))

    # ── Tab 7: Chat Assistant ───────────────────────────────
    with tab7:
        st.subheader("💬 Research Director Chat Assistant")
        st.markdown("Ask me anything about the current research report, proposal status, or novelty scores.")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about your research, proposals, or citation networks…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.write(user_input)

            # Build context from report
            context = f"""
You are the AI Research Assistant helping a Research Director.
Current research topic: {report.request.topic}
Agency: {report.grant.get('agency','')}
Novelty score: {report.novelty.get('novelty_score', 0):.2f} ({report.novelty.get('novelty_label','')})
Papers fetched: {report.literature.get('fetched', 0)}
Identified gaps: {'; '.join(report.gaps.get('identified_gaps', [])[:3])}
Emerging trends: {'; '.join(report.trends.get('emerging_topics', [])[:3])}
Hypothesis: {report.methodology.get('hypothesis', 'Not generated')}
Proposal sections available: {', '.join(report.grant.get('sections', {}).keys())}

Answer helpfully and concisely. Reference specific data from the report.
"""
            from core.llm_factory import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            try:
                llm = get_llm(temperature=0.5)
                messages = [
                    SystemMessage(content=context),
                    *[HumanMessage(content=m["content"]) if m["role"] == "user"
                      else type('AssistantMsg', (), {"content": m["content"], "type": "ai"})()
                      for m in st.session_state.chat_history[-6:]],
                ]
                response = llm.invoke(messages)
                assistant_reply = response.content
            except Exception as e:
                assistant_reply = f"⚠️ Chat error: {e}. Please check your API key in the sidebar."

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            with st.chat_message("assistant"):
                st.write(assistant_reply)

# ── Empty state ─────────────────────────────────────────────────
else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🤖 6 Specialized AI Agents
        - **Literature Mining** – ArXiv + Semantic Scholar
        - **Trend Analysis** – BERTopic/LDA
        - **Gap Identification** – Semantic clustering
        - **Methodology Design** – Experiment planning
        - **Grant Writing** – NSF/NIH/DARPA/EU formats
        - **Novelty Scoring** – Plagiarism check
        """)
    with col2:
        st.markdown("""
        ### ⚡ How to Use
        1. Enter your **research topic** in the sidebar
        2. Select your **domain** and **grant agency**
        3. Add your **API key** (OpenAI or Anthropic)
        4. Click **Run Research Pipeline**
        5. Explore tabs: Trends, Gaps, Proposals
        6. Export PDF/DOCX grant proposal
        """)
    with col3:
        st.markdown("""
        ### 📊 Research Director Dashboard
        - Track **proposal status** & novelty scores
        - Explore **citation networks**
        - Compare **research gaps** across topics
        - Chat with the **AI assistant**
        - Download ready-to-submit **grant proposals**
        """)
