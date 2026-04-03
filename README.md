# Meta AI

An intelligent system for automated machine learning pipeline generation. Meta AI is a local-first application for tabular data: you supply your own datasets (typically CSV), configure the target column, and run ingestion through training, analysis, and optional deployment export in a single Gradio-based workspace.

**New to this project?** Read **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** first.

**How to deploy this project** (servers, Docker, inference API): that means **hosting the codebase or an Export bundle**, not the MLOps tab inside the app. See **[docs/HOW_TO_DEPLOY.md](docs/HOW_TO_DEPLOY.md)** for copy-paste steps.

**Put this repo on GitHub** (install Git, create the remote repo, commit, push): **[docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md)**.

This README is the full reference: installation variants, every dashboard tab, optional services, and troubleshooting. It assumes Python 3.10 or newer unless otherwise noted.

---

## Table of contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running the application](#running-the-application)
5. [Dashboard structure (tabs)](#dashboard-structure-tabs)
6. [Typical workflow](#typical-workflow)
7. [Optional services](#optional-services)
8. [Production export](#production-export)
9. [Deployment (hosting the project)](#deployment-hosting-the-project)
10. [Product specification (PRD)](#product-specification-prd)
11. [Repository layout](#repository-layout)
12. [Development](#development)
13. [Troubleshooting](#troubleshooting)
14. [Security and data handling](#security-and-data-handling)
15. [Contributing and license](#contributing-and-license)

---

## Overview

Meta AI targets **your tabular data**, not bundled sample datasets. You upload files through the UI, select a prediction target, and drive cleaning, feature engineering, model training (including Optuna and stacking where available), explainability, drift-oriented checks, and packaging for inference. The canonical entry point is `quick_start.py`, which loads `dashboard_v3.py` and binds the Gradio application to an available local port (commonly in the 7860 range; the process prints the exact URL).

A separate FastAPI module, `backend_api_main.py`, exists for API-oriented workflows and some container configurations. It is optional for interactive dashboard use.

---

## Requirements

- **Python**: 3.10 or newer (see `pyproject.toml` for the declared constraint).
- **Operating system**: Windows, macOS, or Linux. Paths in examples use POSIX shells unless noted; Windows users can use `scripts\setup.bat` and `RUN_GRADIO_APP.bat` as described below.
- **Hardware**: Sufficient RAM for your dataset and chosen models; GPU is optional and depends on optional deep-learning stacks you enable.
- **Network**: Not required for core local use. Optional integrations (for example cloud LLM APIs) require credentials via environment variables.

Core dependencies are listed in `requirements.txt` (Gradio, scikit-learn, pandas, FastAPI, MLflow, Optuna, and related scientific stack). Install time can be significant on first run because of compiled wheels and large packages.

---

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <your-fork-or-remote-url> meta_ai_builder_pp
cd meta_ai_builder_pp

python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell or cmd)**: `.venv\Scripts\activate`
- **macOS / Linux**: `source .venv/bin/activate`

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows shortcut**: Run `scripts\setup.bat` once to create common directories under `data/` and `exports/` and to install from `requirements.txt`. Then start the app with `RUN_GRADIO_APP.bat` or `python quick_start.py` as usual.

---

## Running the application

Start the main dashboard:

```bash
python quick_start.py
```

Keep the terminal session open. The console prints a local URL (for example `http://127.0.0.1:7860` or a nearby port if the default is busy). Open that URL in a browser. Stopping the process closes the UI; the browser will show connection errors if you refresh after exit.

**Windows**: Double-click `RUN_GRADIO_APP.bat` in the repository root to run `quick_start.py` using a detected virtual environment (`.venv312`, `.venv`, or system Python).

**Port selection**: You can set `GRADIO_SERVER_PORT` in the environment to prefer a fixed port if your environment requires it.

---

## Dashboard structure (tabs)

The interface launched by `quick_start.py` is defined in `dashboard_v3.py`. Top-level **tabs** and nested **sub-tabs** are listed below in the same order as the UI.

### Data Ingestion

- **Manual Upload** — CSV upload, target column selection, preview and summary statistics.
- **Semantic Detection** — Column-level semantic typing and confidence reporting.
- **Data Validation** — Pydantic schema generation and row-level validation results.
- **Drift Baseline** — Statistical fingerprint / baseline capture for later drift comparison.
- **Data Lineage** — Lineage visualization and pipeline-stage metrics.

### Data Reconstruction

- **Systematic Bias Detection** — Missing-data bias patterns and related reporting.
- **Bayesian Reconstruction** — MICE-style iterative imputation and reconstruction views.
- **Outlier Detection** — Isolation Forest scoring with configurable contamination.

### EDA

- **Hypothesis Generation** — Correlation-driven hypotheses, heatmaps, and tables.
- **Dimensionality Reduction** — UMAP or t-SNE projections (2D/3D) plus optional AI cluster explanation controls.

### Feature Engineering

- **Agentic Feature Creation** — Domain-style derived features from semantic column analysis.
- **Feature Selection (RFE)** — Recursive feature elimination with ranking outputs.
- **Auto Feature Engineering** — Polynomial features, binning, datetime features, encodings, and reports.

### Model Training

- **Normal Training** — Multi-model baseline comparison with default hyperparameters.
- **Optuna Optimization** — Trial-based hyperparameter search with configurable trial count.
- **Stacking Ensemble** — Multi-estimator stacking with a meta-learner.

### Analysis & XAI

- **Performance Analysis** — Task-appropriate metrics, plots, and result tables (classification and regression).
- **SHAP Explainability** — Global and local SHAP views with sample index selection.
- **Fairness & Bias Audit** — Group-wise metrics and fairness visualizations.

### Agentic Auditing

- **Scientific Post-Mortem** — LLM-assisted structured summary of the experiment.
- **Counterfactual Reasoning** — Target-accuracy-oriented improvement suggestions.
- **Ask the Agent** — Free-form Q&A over experiment context.

### MLOps and Production

- **Readiness Check** — Pre-deployment checklist and readiness visualization.
- **API Generation** — Snippets for FastAPI-oriented deployment, with nested views:
  - **API Code (api.py)**
  - **Dockerfile**
  - **requirements.txt**
- **Drift Detection** — Training versus current data drift analysis and thresholds.
- **Monitoring Dashboard** — Simulated production monitoring horizon and charts.
- **MLflow Tracking** — Connection status and guidance for the MLflow UI (experiment `Meta-AI-Models` when logging is enabled).

### Export

- Single top-level area for **production ZIP export** (model artifacts, API scaffold, requirements, Docker material, README as generated) and session reset where provided.

---

## Typical workflow

1. **Upload data**: Use the ingestion area (for example **Data Ingestion → Manual Upload**) and load a UTF-8 CSV when possible.
2. **Define the task**: Select the target column. The UI infers classification versus regression from the data where applicable.
3. **Preparation**: Run semantic detection, validation, and drift baselines as needed for your project.
4. **Cleaning and features**: Apply reconstruction, cleaning, outlier handling, and automated feature engineering according to your quality bar.
5. **Training**: Train from **Model Training** using normal training, Optuna-driven search, or stacking ensembles, depending on what the tab exposes for your session.
6. **Analysis**: Review performance and explainability in **Analysis & XAI**.
7. **Operations**: Use **MLOps and Production** for readiness checks, API-oriented snippets, drift views, and MLflow status when enabled.
8. **Export**: After a successful training run in the same session, generate a deployment ZIP (model artifacts, FastAPI application scaffold, requirements, Docker-related files as generated).

Large files may increase memory use and training time. Close unrelated heavy processes if you are resource-constrained.

---

## Optional services

### MLflow tracking

Training from the dashboard can log runs to MLflow on a best-effort basis. Defaults often use a SQLite-backed store such as `sqlite:///./mlflow.db`; you may override behavior with `MLFLOW_TRACKING_URI`. The UI may reference experiment naming such as `Meta-AI-Models`.

Start the MLflow UI in a **separate** terminal (example: port 5000):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`. Inside the app, use **MLOps and Production → MLflow Tracking** to inspect status when that panel is available.

### FastAPI backend (`backend_api_main.py`)

For API-centric development or Docker stacks that expect a Python ASGI app at the repository root, run Uvicorn explicitly (example):

```bash
uvicorn backend_api_main:app --host 127.0.0.1 --port 8000 --reload
```

Health and documentation endpoints depend on how `backend_api_main.py` is configured; consult that module or your deployment manifest for paths.

### Static React dashboard (port 3000)

To serve the contents of `frontend/react-dashboard` over HTTP for local preview:

```bash
python scripts/run_static_dashboard.py
```

Use `scripts/check_and_start_servers.py` or `scripts/quick_launcher.py` only if you intend to coordinate this static server with the FastAPI backend; they are optional utilities.

---

## Production export

After training succeeds in the dashboard session, **Export** produces a ZIP containing the serialized model, generated FastAPI entrypoint (`api.py`), schema, pinned-style `requirements.txt`, Dockerfile material as generated, and supporting files. Unzip into a clean directory, install dependencies, and run:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is typically available at `http://127.0.0.1:8000/docs` when FastAPI docs are enabled. Run export only in the same session as a completed train if the UI requires a trained model and feature metadata in memory.

Automated checks for export packaging live under `tests/test_export_hardening.py`.

---

## Deployment (hosting the project)

**What this means:** how to run **this repository** (or a model **Export** ZIP) on a server, in Docker, or behind a load balancer so other people or systems can reach it.

**What this is not:** it is **not** the **MLOps and Production** *area inside the Gradio UI* (readiness panels, API snippet tabs, MLflow status). That is in-app tooling. **Hosting the project** is documented here and in **[docs/HOW_TO_DEPLOY.md](docs/HOW_TO_DEPLOY.md)** (copy-paste commands for each option).

### Quick reference

| Approach | Entry |
|----------|--------|
| Dashboard on a network | `GRADIO_SERVER_NAME=0.0.0.0` + `python quick_start.py` (put TLS/auth in front) |
| Single Docker image | `Dockerfile.prod` — see [HOW_TO_DEPLOY.md](docs/HOW_TO_DEPLOY.md) |
| UI + `backend_api_main` API | `docker compose -f docker-compose.metaai.yml up --build` |
| Scoring API only | Train in UI, **Export** ZIP, then `uvicorn api:app` on the unzipped bundle |
| Kubernetes | Start from `kubernetes_deployment.yaml` and customize |

### “Same session” and export

**Session** means one continuous run of `python quick_start.py`. **Export** often requires training and export in the **same** run before you stop the process. That is in-memory workflow, not server **deployment**.

### Production checklist (short)

| Topic | Guidance |
|-------|----------|
| TLS | Terminate at load balancer or ingress; do not expose plain HTTP on untrusted networks. |
| Authentication | Enforce at proxy, API gateway, or mesh; do not assume multi-tenant security from Gradio alone. |
| Secrets | Use environment variables or a secret manager for API keys (OpenAI, Groq, and so on). |
| Data | Mount persistent volumes for `data/`, `exports/`, and MLflow stores if you keep them on disk. |
| Scale-out | Scale **stateless inference** from export; the Gradio training UI is typically **single-user / few users** per instance. |
| Observability | Add structured logging, metrics, and tracing outside this repo’s defaults for regulated environments. |

---

## Product specification (PRD)

The formal **scope, non-goals, and success criteria** are written in a **constitution-style** product document (not a slide deck):

**[docs/PRD.md](docs/PRD.md)**

Use it for alignment on what Meta AI promises to do, what it explicitly does not promise, and how deployment modes relate at the product level. **Operational hosting steps** are in **[docs/HOW_TO_DEPLOY.md](docs/HOW_TO_DEPLOY.md)**.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `quick_start.py` | Primary launcher for the Gradio dashboard |
| `dashboard_v3.py` | Main UI composition and tab wiring |
| `backend_api_main.py` | Optional FastAPI application for API and some deployments |
| `RUN_GRADIO_APP.bat` | Windows helper to invoke `quick_start.py` |
| `scripts/setup.bat` | Windows-first install helper |
| `scripts/run_static_dashboard.py` | Optional static file server for `frontend/react-dashboard` |
| `scripts/check_and_start_servers.py` | Optional multi-service starter |
| `core/` | Ingestion, training, export, and supporting ML logic |
| `interface/` | Additional Gradio interfaces and related modules |
| `mlops/` | Tracking and operational helpers |
| `llm/`, `agents/`, `chatbot/` | Optional language-model and agent-related code paths |
| `api/`, `app/` | Additional API or application packaging |
| `tests/` | Pytest collection |
| `exports/` | Default area for generated export ZIPs (artifacts; configure ignores for version control) |
| `requirements-dev.txt` | Pytest, Ruff, Bandit (use with `make setup-dev`) |
| `pyproject.toml` | Python version, Ruff and Bandit settings |
| `docs/README.md` | Index of documentation files |
| `docs/GETTING_STARTED.md` | Beginner walkthrough from zero |
| `docs/HOW_TO_DEPLOY.md` | Host the project (server, Docker, Compose, export API) |
| `docs/GITHUB_SETUP.md` | Beginner steps to connect this folder to a GitHub repository |
| `docs/PRD.md` | Product specification (constitution-style PRD) |

Generated or machine-local paths such as `mlruns/`, `logs/`, `models/`, `exports/`, and local databases are best excluded from commits; see `.gitignore` and your own policies.

---

## Development

Install dev tools (pytest, ruff, bandit) once:

```bash
pip install -r requirements-dev.txt
# or: make setup-dev
```

Run checks:

```bash
pytest -q
pytest tests/test_export_hardening.py -q
make lint
```

`pyproject.toml` configures **Ruff**: very large UI files (`dashboard_v3.py`, `interface/gradio_demo.py`, `interface/metaai_pro_tabs.py`) are **excluded** from lint so CI stays fast and focused on `core/`, `mlops/`, `tests/`, and the rest. Line length ignores **E501**.

CI (`.github/workflows/ci.yml`) runs tests on Python 3.10 and 3.11, Ruff, Bandit, and a **Dockerfile.prod** build.

Refer to `CONTRIBUTING.md` for branch and review expectations. Do not commit secrets or full production datasets.

---

## Troubleshooting

| Symptom | Likely cause |
|--------|----------------|
| Browser shows connection refused for the Gradio URL | `quick_start.py` is not running, crashed, or the port changed; restart and use the printed URL |
| Connection refused on port 5000 | MLflow UI not started in a second process |
| Connection refused on port 8000 | Neither the exported `uvicorn api:app` nor `backend_api_main` is running on that host and port |
| Export reports missing model or features | Train again in the same session so in-memory state matches export prerequisites |
| Slow or unstable training | Dataset size, hyperparameter search load, or system memory pressure; reduce scope or data size |

---

## Security and data handling

- Store API keys and secrets in environment variables or a local `.env` file that is **not** committed.
- Treat uploaded data according to your organization’s retention and privacy policies; this repository does not replace a data governance process.
- Review generated export packages before deploying them to networked environments; they contain serialized models and may embed preprocessing logic.

---

## Contributing and license

Contributing guidelines are in `CONTRIBUTING.md`. Choose and record an explicit license for this repository (for example MIT, Apache-2.0, or a proprietary notice) in place of this placeholder section.

---

**Meta AI** is intended as a workstation tool for reproducible tabular ML workflows from data upload through optional deployment packaging. For production systems, add monitoring, authentication, rate limiting, and infrastructure hardening appropriate to your threat model.
