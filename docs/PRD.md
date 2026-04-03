# Meta AI — Product Specification

This document is the **authoritative product definition** for Meta AI. It is written as a **compact constitution**: short articles, explicit schedules, and stated non-goals. It is not a marketing brief; it is a reference for scope, trade-offs, and success.

---

## Preamble

Meta AI is a **local-first, interactive system** for building supervised machine learning workflows on **tabular data** supplied by the user. The primary surface is a **Gradio dashboard** (`quick_start.py` → `dashboard_v3.py`). Optional paths include a **FastAPI** companion (`backend_api_main.py`) and **packaged inference** produced by the in-app **Export** flow. Nothing in this specification guarantees suitability for regulated production without additional controls chosen by the operator.

---

## Article I — Purpose

**Meta AI exists to shorten the path from a CSV (or similar tabular file) to a trained model, explainability artifacts, operational checks, and a deployable inference package—within a single coherent UI.**

Secondary purposes: experimentation, teaching, and internal tooling where a full MLOps platform is not yet justified.

---

## Article II — Primary actor

The **primary user** is a practitioner (data scientist, ML engineer, or technical analyst) who:

- Controls the machine or container where the app runs.
- Provides datasets and chooses the prediction target.
- Accepts responsibility for data privacy, retention, and downstream deployment policy.

The system does **not** assume multi-tenant SaaS operation unless an operator explicitly builds that layer.

---

## Article III — Core obligations (what the product must do)

1. **Ingest** user-uploaded tabular data and expose column-level understanding (preview, typing, validation) through dedicated UI areas.
2. **Prepare** data with optional bias, reconstruction, and outlier workflows exposed as distinct sub-areas.
3. **Explore** data with hypothesis and dimensionality-reduction tooling appropriate to tabular ML.
4. **Engineer** features through agent-assisted creation, selection (RFE), and automated pipelines.
5. **Train** models via baseline comparison, Optuna-driven search, and stacking, subject to resource limits.
6. **Explain** model behavior through performance views, SHAP, and fairness-oriented audits where applicable.
7. **Support operations** through readiness checks, API-oriented generation, drift and monitoring views, and MLflow status integration (best-effort logging).
8. **Export** a reproducible deployment bundle after successful training in-session, including model artifacts and generated serving code where implemented.

These obligations map to the **Dashboard structure** documented in the root `README.md` (tabs and sub-tabs).

---

## Article IV — System constraints (what the product assumes)

| Constraint | Implication |
|------------|-------------|
| Python runtime | Deployment and development assume a supported Python version per `pyproject.toml` / `requirements.txt`. |
| Local or single-tenant execution | No built-in multi-user auth in the core Gradio path; operators add identity and authorization externally if needed. |
| Session-scoped training state | Export and some analyses assume a coherent train completed in the **same** interactive session unless otherwise designed. |
| Optional LLM keys | Agentic and LLM-assisted features may require API keys via environment; behavior degrades or falls back when keys are absent. |
| Resource bounds | Large data and heavy search increase memory and CPU; the product does not guarantee fixed latency. |

---

## Article V — Explicit non-goals

The following are **out of scope** for the core product definition unless separately specified and implemented:

- Guaranteed regulatory compliance (HIPAA, GDPR operational compliance, etc.) without customer-specific controls.
- Fully managed cloud training at hyperscale without customer infrastructure.
- Automatic real-time retraining in production without operator automation.
- Non-tabular modalities (video, audio, raw text corpora) as first-class citizens in the main dashboard contract.
- A commitment to a single cloud vendor or a single deployment topology.

---

## Article VI — Deployment modes (product-level, not operational runbooks)

The product supports **conceptually distinct** deployment postures:

1. **Workstation**: Single user, local process, browser to Gradio URL.
2. **Containerized UI**: Gradio bound to `0.0.0.0` behind Docker or orchestration, with volumes for `data/` and artifacts.
3. **Inference service**: **Exported ZIP** run as `uvicorn api:app` (or equivalent) separate from the training UI.
4. **Optional API**: `backend_api_main.py` or compose-defined services where present in the repository, subject to file and path consistency.

Operational commands belong in `README.md` (Deployment section), not in this specification.

---

## Article VII — Success criteria

The product **succeeds** for a release when:

1. A user can **upload** data, **select** a target, **train** at least one path to completion, and **obtain** metrics and artifacts without undocumented manual patching.
2. **Export** produces an installable bundle that can serve predictions under documented assumptions (`tests/test_export_hardening.py` guards regression where applicable).
3. **Documentation** in the root `README.md` matches the actual entrypoints and tab structure.
4. **Security posture** is honest: secrets are environmental; no silent exfiltration of user data is a design goal.

---

## Schedule A — Capability inventory (summary)

| Area | Intent |
|------|--------|
| Data Ingestion | Load, understand, validate, baseline, lineage |
| Data Reconstruction | Bias-aware repair and anomaly screening |
| EDA | Hypotheses and structure visualization |
| Feature Engineering | Derived features, selection, automation |
| Model Training | Baselines, search, ensembles |
| Analysis & XAI | Metrics, SHAP, fairness |
| Agentic Auditing | Summaries, counterfactuals, Q&A (where keys allow) |
| MLOps and Production | Readiness, snippets, drift, monitoring, MLflow |
| Export | Packaged inference and deployment artifacts |

---

## Schedule B — Glossary

| Term | Meaning |
|------|---------|
| **Session** | One continuous run of the dashboard process with in-memory application state. |
| **Export bundle** | ZIP produced by the Export tab, intended for inference deployment. |
| **Best-effort logging** | MLflow or similar may fail softly; training should still complete. |

---

## Document control

| Field | Value |
|-------|--------|
| **Format** | Constitution-style PRD (articles + schedules) |
| **Companion** | Root `README.md` for run, deploy, and tab reference |
| **Change policy** | Update this file when user-facing scope or non-goals change materially |

---

*End of product specification.*
