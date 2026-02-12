<div align="center">

# DataWhisperer

**Automated Exploratory Data Analysis powered by LLMs and Multi-Agent Systems**

Upload a dataset. Get insights in minutes. No data science experience required.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

**Local-first by design — your data never leaves your machine.**

</div>

---

## Features

- **Auto Profiling** — types, distributions, missing values, duplicates, PII detection
- **Anomaly Detection** — IQR-based univariate + Isolation Forest multivariate outlier detection
- **Smart Visualizations** — LLM-guided chart selection with Plotly interactive charts
- **Statistical Analysis** — normality tests, chi-square, correlation analysis with plain-English interpretations
- **Narrative Reports** — LLM writes a data story explaining key findings
- **Chat with Your Data** — ask questions in natural language, get SQL-backed answers via DuckDB
- **Export** — interactive HTML reports or PDF, one click

## Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/your-username/data-whisperer.git
cd data-whisperer
cp .env.example .env
docker-compose up
```

The Ollama model is pulled automatically on first run.

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Option 2: Local Setup

```bash
git clone https://github.com/your-username/data-whisperer.git
cd data-whisperer
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env
```

Install and start Ollama:

```bash
ollama pull llama3.1
ollama serve
```

Start the app (two terminals):

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501 and upload a CSV or Excel file.

---

## Privacy & LLM Options

| Mode | Provider | Privacy | Requirements |
|------|----------|---------|--------------|
| **Local (default)** | Ollama | Your data never leaves your machine | 8GB+ RAM |
| **Cloud (opt-in)** | Groq | Data summaries sent to Groq servers | Free API key |
| **Fallback** | None | Full privacy | No LLM needed |

### Ollama — fully local, fully private (default)

All LLM inference runs on your machine. Nothing is sent externally. Recommended for sensitive or proprietary data.

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

### Groq — cloud inference, fast (opt-in)

Uses Groq's free API for faster inference without a local GPU. Dataset metadata and summaries **will be sent to Groq's servers**. Only use for non-sensitive or public data.

```env
LLM_PROVIDER=groq
GROQ_API_KEY=<your-key>
GROQ_MODEL=llama-3.1-8b-instant
```

Get a free API key (no credit card): https://console.groq.com/keys

### Fallback — no LLM needed

If no LLM is available, all agents use rule-based analysis. You still get profiling, outlier detection, correlations, statistics, and visualizations — just without the LLM-powered narrative reports, chat Q&A, and smart chart selection.

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  Streamlit   │────▶│            FastAPI Backend            │
│  Frontend    │◀────│                                      │
└─────────────┘     │  ┌──────────┐  ┌──────────────────┐  │
                    │  │  Upload   │  │   LangGraph       │  │
                    │  │  + Parse  │  │   Orchestrator    │  │
                    │  │  (Polars) │  │                    │  │
                    │  └──────────┘  │  ┌──────────────┐  │  │
                    │               │  │  Detective    │  │  │
                    │  ┌──────────┐  │  │  Statistician │  │  │
                    │  │  DuckDB  │  │  │  Visualizer   │  │  │
                    │  │  (SQL)   │  │  │  Storyteller  │  │  │
                    │  └──────────┘  │  └──────────────┘  │  │
                    │               └──────────────────┘  │
                    │  ┌──────────┐  ┌──────────────────┐  │
                    │  │ ChromaDB │  │ Ollama / Groq    │  │
                    │  │ (RAG)    │  │ (LLM)            │  │
                    │  └──────────┘  └──────────────────┘  │
                    └──────────────────────────────────────┘
```

### Agents

| Agent | Role | Fallback |
|-------|------|----------|
| **Detective** | Finds anomalies, data quality issues, PII | Rule-based pattern matching |
| **Statistician** | Runs statistical tests, interprets results | Automated test selection |
| **Visualizer** | Selects and creates appropriate charts | Heuristic chart picker |
| **Storyteller** | Writes narrative report of findings | Template-based summary |

---

## Project Structure

```
data-whisperer/
├── app/
│   ├── agents/              # LangGraph multi-agent system
│   │   ├── orchestrator.py  # Workflow graph definition
│   │   ├── detective.py     # Anomaly & quality detection
│   │   ├── statistician.py  # Statistical testing
│   │   ├── visualizer.py    # Chart generation (Plotly)
│   │   └── storyteller.py   # Report narrative
│   ├── analyzers/           # Core analysis modules
│   │   ├── profiler.py      # Dataset profiling
│   │   ├── correlations.py  # Pearson + Cramér's V
│   │   ├── outliers.py      # IQR + Isolation Forest
│   │   └── timeseries.py    # Time series analysis
│   ├── exports/             # Report generation
│   │   ├── html_gen.py      # Interactive HTML reports
│   │   └── pdf_gen.py       # PDF reports (ReportLab)
│   ├── utils/               # Shared utilities
│   │   ├── data_loader.py   # CSV/Excel/Parquet loading
│   │   ├── llm_client.py    # Ollama + Groq LLM client
│   │   └── embeddings.py    # ChromaDB vector store
│   └── main.py              # FastAPI application
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── tests/                   # Pytest test suite
├── data/sample_datasets/    # Example datasets to try
├── docker-compose.yml       # One-command deployment
├── Dockerfile
├── requirements.txt
└── .env.example             # Configuration template
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend | FastAPI, Polars, DuckDB |
| LLM | Ollama (local) or Groq (cloud) |
| Agents | LangGraph |
| Visualization | Plotly, Matplotlib, Seaborn |
| Vector Store | ChromaDB |
| Frontend | Streamlit |
| PDF Export | ReportLab + Kaleido |
| Deployment | Docker, Docker Compose |

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `groq` (cloud) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | Any Ollama model |
| `GROQ_API_KEY` | — | Required only for Groq |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | ChromaDB storage path |
| `OUTPUT_DIR` | `./outputs` | Report output directory |

## Sample Datasets

Try these from `data/sample_datasets/`:

| File | Description |
|------|-------------|
| `sales_data.csv` | E-commerce transactions with quality issues (negative quantities, price anomalies, duplicates) |
| `customer_churn.csv` | Telecom churn data with missing values, impossible age (200), negative charge |
| `stock_prices.csv` | Daily stock prices with missing close prices and suspicious volume spike |

## Self-Hosting

Deploy on any VPS or server:

```bash
git clone https://github.com/your-username/data-whisperer.git
cd data-whisperer
cp .env.example .env
# edit .env if needed
docker-compose up -d
```

Access at `http://<server-ip>:8501`. For HTTPS, put Nginx or Caddy in front.

**Minimum requirements:** 2 CPU cores, 8GB RAM (for Ollama). With Groq, 1GB RAM is sufficient.

## Contributing

Pull requests welcome. Please add tests for new features. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
