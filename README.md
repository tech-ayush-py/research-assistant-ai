# ResearchAI — Academic Research Intelligence Platform

A production-grade, multi-agent AI system that assists faculty and researchers with literature discovery, research gap identification, experimental design, and funding-ready grant proposal generation. Built on Google Gemini, LangChain, and ChromaDB.

---

## Overview

ResearchAI automates the most time-intensive parts of academic research by orchestrating six specialised AI agents in a sequential pipeline. A researcher enters a topic and receives — within minutes — a novelty-scored analysis of the existing literature, a prioritised list of research gaps, a suggested experimental methodology, and a complete grant proposal formatted for their chosen funding agency.

The system is deployed as an interactive web application using Streamlit and is accessible without any local setup.

---

## Live Demo

**Deployed at:** https://research-assistant-ai-8q2pe9m99bimjejvgcpkd9.streamlit.app/

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
| LLM | Google Gemini 3.1 Flash Lite Preview|
| Orchestration | Custom pipeline (LangChain-based) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| Literature APIs | ArXiv Python SDK, Semantic Scholar REST API |
| Frontend | Streamlit |
| Visualisation | Plotly |
| Export | FPDF2 (PDF), python-docx (DOCX) |
| Language | Python 3.10+ |

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
| `LLM_PROVIDER` | LLM provider: `gemini` | `gemini` |
| `LLM_MODEL` | Model name | `gemini-3.1-flash-lite-preview` |
| `EMBEDDING_MODEL` | HuggingFace sentence-transformers model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | Path for ChromaDB persistence | `./data/chroma_db` |

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
