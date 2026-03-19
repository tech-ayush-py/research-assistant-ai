# ResearchAI — Academic Research Intelligence Platform

A production-grade, multi-agent AI system that assists faculty and researchers with literature discovery, research gap identification, experimental design, and funding-ready grant proposal generation. Built on Google Gemini, LangChain, and ChromaDB.

---

## Overview

ResearchAI automates the most time-intensive parts of academic research by orchestrating six specialised AI agents in a sequential pipeline. A researcher enters a topic and receives — within minutes — a novelty-scored analysis of the existing literature, a prioritised list of research gaps, a suggested experimental methodology, and a complete grant proposal formatted for their chosen funding agency.

The system is deployed as an interactive web application using Streamlit and is accessible without any local setup.

---

## Live Demo

**Deployed at:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchAI Dashboard                      │
│                 Streamlit Web Application                    │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Research           │
              │  Orchestrator       │
              │  (Pipeline Coord.)  │
              └──────────┬──────────┘
                         │
      ┌──────────────────┼──────────────────────────┐
      │                  │                          │
      ▼                  ▼                          ▼
┌───────────┐    ┌──────────────┐         ┌─────────────────┐
│Literature │    │ Trend        │         │ Gap             │
│Mining     │    │ Analysis     │         │ Identification  │
│Agent      │    │ Agent        │         │ Agent           │
└─────┬─────┘    └──────────────┘         └─────────────────┘
      │
      ▼                  ▼                          ▼
┌───────────┐    ┌──────────────┐         ┌─────────────────┐
│ ChromaDB  │    │ Methodology  │         │ Grant Writing   │
│ RAG Layer │    │ Design Agent │────────▶│ Agent           │
│(Persisted)│    └──────────────┘         └────────┬────────┘
└───────────┘                                      │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Novelty & Plagiarism│
                                        │ Scoring Agent       │
                                        └─────────────────────┘
```

---

## Agent Pipeline

| # | Agent | Responsibility | Key Technology |
|---|-------|---------------|----------------|
| 1 | **Literature Mining** | Retrieves papers from ArXiv and Semantic Scholar, embeds and indexes them | `arxiv`, `sentence-transformers`, ChromaDB |
| 2 | **Trend Analysis** | Detects emerging and declining sub-topics using TF-IDF keyword analysis and LLM summarisation | `scikit-learn`, Gemini |
| 3 | **Gap Identification** | Identifies under-explored research areas through semantic clustering and LLM reasoning | ChromaDB similarity search, Gemini |
| 4 | **Methodology Design** | Recommends datasets, baseline models, evaluation metrics, and experimental timeline | Curated registries + Gemini |
| 5 | **Grant Writing** | Generates complete proposals section by section, formatted per agency guidelines | LangChain, Gemini |
| 6 | **Novelty Scoring** | Computes semantic similarity of the proposal against the corpus to produce a novelty score | `sentence-transformers`, cosine similarity |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Flash / Pro / 2.0 Flash |
| Orchestration | Custom pipeline (LangChain-based) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| Literature APIs | ArXiv Python SDK, Semantic Scholar REST API |
| Frontend | Streamlit |
| Visualisation | Plotly |
| Export | FPDF2 (PDF), python-docx (DOCX) |
| Language | Python 3.10+ |

---

## Supported Grant Agencies

| Agency | Sections Generated |
|--------|-------------------|
| NSF | Project Summary, Project Description, Broader Impacts, Intellectual Merit, Budget Justification, References |
| NIH | Specific Aims, Background & Significance, Innovation, Approach, Personnel, Budget Narrative |
| DARPA | Technical Approach, Innovation, Team Capabilities, Risk Mitigation, Milestones, Budget |
| EU Horizon | Excellence, Impact, Implementation, Partners, Work Packages, Budget Breakdown |

---

## Project Structure

```
research_assistant/
├── app.py                           # Streamlit dashboard — main entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore
│
├── config/
│   └── settings.py                  # Central config — LLM, ChromaDB, agency templates
│
├── core/
│   ├── llm_factory.py               # LLM provider selector (Gemini / OpenAI / Anthropic)
│   ├── vector_store.py              # ChromaDB RAG layer — ingest, embed, retrieve
│   └── orchestrator.py              # Multi-agent pipeline coordinator
│
├── agents/
│   ├── literature_mining_agent.py   # ArXiv + Semantic Scholar crawler
│   ├── trend_analysis_agent.py      # TF-IDF keyword trend detection
│   ├── gap_identification_agent.py  # Research gap finder
│   ├── methodology_design_agent.py  # Experimental design recommender
│   ├── grant_writing_agent.py       # Grant proposal generator
│   └── novelty_scoring_agent.py     # Plagiarism and novelty scorer
│
├── utils/
│   └── export.py                    # PDF, DOCX, and Markdown exporters
│
├── .streamlit/
│   ├── config.toml                  # Streamlit theme and server config
│   └── secrets.toml.example         # Secrets template for cloud deployment
│
└── data/
    └── chroma_db/                   # Persisted vector store (auto-created on first run)
```

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- A Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/research-assistant-ai.git
cd research-assistant-ai

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here

# Run the application
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment on Streamlit Cloud

1. Push the repository to GitHub (ensure `.env` and `data/chroma_db/` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository
3. Set the main file path to `app.py`
4. Under **App Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your_api_key_here"
LLM_PROVIDER   = "gemini"
LLM_MODEL      = "gemini-1.5-flash"
```

5. Click **Deploy**

---

## Usage

1. Enter your **Gemini API key** and select a model in the sidebar
2. Describe your **research topic** in one or two sentences
3. Select your **research domain** and adjust the paper retrieval count
4. Fill in **grant proposal details** — agency, PI name, institution, budget
5. Click **Run Research Pipeline**
6. Explore results across five tabs: Overview, Gaps & Trends, Methodology, Grant Proposal, Assistant
7. Download the completed proposal as PDF, Word document, or Markdown

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | — |
| `LLM_PROVIDER` | LLM provider: `gemini`, `openai`, `anthropic` | `gemini` |
| `LLM_MODEL` | Model name | `gemini-1.5-flash` |
| `OPENAI_API_KEY` | OpenAI API key (optional fallback) | — |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional fallback) | — |
| `EMBEDDING_MODEL` | HuggingFace sentence-transformers model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | Path for ChromaDB persistence | `./data/chroma_db` |

---

## Evaluation Criteria Mapping

This project was developed as a final-year academic submission. The implementation maps to the evaluation rubric as follows:

| Criterion | Weight | Implementation |
|-----------|--------|----------------|
| Presentation | 30% | Interactive Streamlit dashboard with live Plotly visualisations and downloadable outputs |
| Project Report | 20% | Automated full-report Markdown/PDF export from the pipeline |
| Viva | 40% | Each agent is independently testable; architecture is modular and clearly documented |
| Novelty | 10% | Semantic novelty scoring agent benchmarks the proposal against 1,000+ papers |

---

## Team

| Member | Responsibility |
|--------|---------------|
| Member 1 | Literature Mining Agent, ChromaDB RAG layer |
| Member 2 | Trend Analysis Agent, Gap Identification Agent |
| Member 3 | Methodology Design Agent, Grant Writing Agent |
| Member 4 | Streamlit Dashboard, Orchestrator, Export utilities |

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Kaissis et al. (2021). *End-to-end privacy preserving deep learning on multi-institutional medical imaging.* Nature Machine Intelligence.
- ChromaDB Documentation — [docs.trychroma.com](https://docs.trychroma.com)
- Semantic Scholar API — [api.semanticscholar.org](https://api.semanticscholar.org)
- Google Gemini API — [ai.google.dev](https://ai.google.dev)

---

## License

This project is submitted as an academic capstone. All rights reserved by the authors.
