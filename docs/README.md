# Documentation index

| Document | Purpose |
|----------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Install Python, run the app on your PC, first CSV workflow |
| [HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md) | **Deploy / host this project** (server, Docker, Compose, inference ZIP) — not the MLOps *tab* in the UI |
| [PRD.md](PRD.md) | Product scope and non-goals (constitution-style) |
| [README.md](../README.md) | Full reference: tabs, MLflow, troubleshooting |
| [GITHUB_SETUP.md](GITHUB_SETUP.md) | **Put the project on GitHub** (install Git, create repo, commit, push) — beginner steps |

**Quality:** Install dev tools with `pip install -r requirements-dev.txt` (or `make setup-dev`). CI runs pytest, Ruff, Bandit (scoped paths), and a `Dockerfile.prod` build on push to `main`, `master`, or `develop`.

Windows helpers from repo root: `RUN_GRADIO_APP.bat`, `scripts\setup.bat`.
