# 🔬 AI-Powered Academic Research Assistant & Grant Proposal Generator

A **multi-agent agentic AI system** that assists faculty and researchers with literature discovery, research gap identification, experimental design, and funding-ready grant proposal generation.

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    RESEARCH DIRECTOR DASHBOARD                   │
│                  (Streamlit Conversational UI)                   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
           ┌───────────▼──────────────┐
           │   Research Orchestrator  │  ← CrewAI-style pipeline
           │  (core/orchestrator.py)  │
           └──────────┬───────────────┘
                      │
    ┌─────────────────┼─────────────────────────────┐
    │                 │                             │
    ▼                 ▼                             ▼
┌───────────┐  ┌───────────────┐         ┌──────────────────┐
│Literature │  │Trend Analysis │         │ Gap Identification│
│  Mining   │  │    Agent      │         │     Agent        │
│  Agent    │  └───────────────┘         └──────────────────┘
│(ArXiv+S2) │         │                         │
└─────┬─────┘         │                         │
      │                ▼                         ▼
      │       ┌────────────────┐      ┌──────────────────────┐
      │       │ Methodology    │      │   Grant Writing      │
      │       │ Design Agent   │────► │      Agent           │
      │       └────────────────┘      │(NSF/NIH/DARPA/EU)    │
      │                               └──────────────────────┘
      │                                         │
      ▼                                         ▼
┌─────────────┐                    ┌────────────────────────┐
│  ChromaDB   │                    │  Novelty & Plagiarism  │
│ Vector Store│                    │    Scoring Agent       │
│  (RAG Layer)│                    └────────────────────────┘
└─────────────┘
```

---

## 🤖 Agents Overview

| Agent | Role | Key Tech |
|-------|------|----------|
| **Literature Mining Agent** | Crawls ArXiv + Semantic Scholar, embeds papers | `arxiv`, `sentence-transformers`, ChromaDB |
| **Trend Analysis Agent** | Detects emerging/declining topics using topic modelling | BERTopic / LDA, LLM summarisation |
| **Gap Identification Agent** | Finds under-explored research intersections | Semantic clustering, graph analysis, LLM |
| **Methodology Design Agent** | Suggests datasets, baselines, metrics, timeline | LLM-powered + curated registries |
| **Grant Writing Agent** | Generates per-section proposals in agency format | LangChain, structured prompting |
| **Novelty Scoring Agent** | Semantic similarity vs. corpus, plagiarism flag | `sentence-transformers`, cosine similarity |

---

## 🗂️ Project Structure

```
research_assistant/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py                 # Central config (LLM, ChromaDB, agencies)
├── core/
│   ├── llm_factory.py              # OpenAI / Anthropic LLM selector
│   ├── vector_store.py             # ChromaDB RAG layer
│   └── orchestrator.py            # Multi-agent pipeline coordinator
├── agents/
│   ├── literature_mining_agent.py  # ArXiv + Semantic Scholar crawler
│   ├── trend_analysis_agent.py     # Topic modelling & trend detection
│   ├── gap_identification_agent.py # Research gap finder
│   ├── methodology_design_agent.py # Experiment designer
│   ├── grant_writing_agent.py      # Proposal generator
│   └── novelty_scoring_agent.py   # Plagiarism & novelty scorer
├── utils/
│   └── export.py                   # PDF, DOCX, Markdown exporters
└── data/
    └── chroma_db/                  # Persisted vector store (auto-created)
```

---

## ⚙️ Setup & Installation

### 1. Clone & install dependencies
```bash
git clone <repo-url>
cd research_assistant
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   LLM_PROVIDER=openai
#   LLM_MODEL=gpt-4o
```

### 3. Run the application
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Usage Guide

### Basic Workflow
1. **Enter Research Topic** – e.g., *"Federated Learning for Privacy-Preserving Medical Imaging"*
2. **Select Domain** – NLP, CV, Biomedical, etc.
3. **Choose Grant Agency** – NSF, NIH, DARPA, or EU Horizon
4. **Fill PI Details** – Name, Institution, Budget
5. **Click "Run Research Pipeline"**
6. **Explore Results** across 7 dashboard tabs

### Dashboard Tabs
| Tab | Contents |
|-----|----------|
| 📊 Overview | Novelty gauge, pipeline log, key metrics |
| 📚 Literature | Retrieved papers, year distribution chart |
| 📈 Trends | Emerging/declining topics, year-keyword map |
| 🔍 Gaps | Research gaps, opportunity areas, reasoning |
| ⚗️ Methodology | Hypothesis, approach, datasets, baselines, metrics |
| 📄 Grant Proposal | Full proposal by section, PDF/DOCX/MD export |
| 💬 Chat | Research Director chat interface |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | OpenAI GPT-4o / Anthropic Claude 3.5 Sonnet |
| **Orchestration** | CrewAI-style pipeline (custom), LangChain |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Literature APIs** | ArXiv Python SDK, Semantic Scholar REST API |
| **Topic Modelling** | BERTopic / LDA |
| **UI** | Streamlit + Plotly |
| **Export** | FPDF2 (PDF), python-docx (DOCX) |
| **Language** | Python 3.10+ |

---

## 📊 Evaluation Rubric Mapping

| Rubric Criterion | Implementation |
|-----------------|----------------|
| **30% – Presentation** | Streamlit dashboard with live demo; Plotly charts |
| **20% – Project Report** | Full Markdown/PDF export with all sections |
| **40% – Viva** | Each agent independently testable; clear architecture |
| **10% – Novelty** | BERTopic trend analysis + novelty scoring agent |

---

## 🌐 API Keys Required

| Service | Where to get |
|---------|-------------|
| OpenAI | https://platform.openai.com/api-keys |
| Anthropic (optional) | https://console.anthropic.com/ |
| Semantic Scholar (optional, higher rate limits) | https://www.semanticscholar.org/product/api |

---

## 📤 Deliverables

- [x] Working multi-agent application with Streamlit UI
- [x] GitHub-ready structured codebase with README
- [x] Automated grant proposal export (PDF / DOCX)
- [x] Novelty scoring and plagiarism check
- [x] Conversational Research Director chat

---

## 👥 Team Roles

| Member | Responsibility |
|--------|---------------|
| Member 1 | Literature Mining Agent + ChromaDB RAG |
| Member 2 | Trend Analysis + Gap Identification Agents |
| Member 3 | Methodology Design + Grant Writing Agents |
| Member 4 | Streamlit Dashboard + Orchestrator + Export |

---

## 📚 References

- Lewis et al. (2020) – *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Grootendorst (2022) – *BERTopic: Neural topic modeling with a class-based TF-IDF*
- ArXiv API: https://arxiv.org/help/api
- Semantic Scholar API: https://api.semanticscholar.org/
- ChromaDB Docs: https://docs.trychroma.com/
