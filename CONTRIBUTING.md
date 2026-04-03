# Contributing to META-AI

Thanks for contributing. This project is moving toward a production-ready architecture, so changes should stay focused, testable, and safe to roll back.

## Development Workflow
1. Create a feature branch from `main`.
2. Keep each pull request scoped to one change theme.
3. Run tests locally before opening a pull request.
4. Include a short risk and rollback note for behavior changes.

## Local Checks
```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
bandit -q -ll -r core/ mlops/ utils/ monitoring/ tests/ -s B101
```

## Pull Request Guidance
- Describe what changed and why.
- Add migration notes if config or environment behavior changed.
- Include testing evidence with the commands you ran and the outcome.

## Security Rules
- Never commit secrets or credentials.
- If the project uses a local `.env` for API keys, do not commit it; document any new variables in `README.md` when behavior depends on them.
- Avoid logs that leak user prompts, document contents, or tokens.

## Architecture Direction
Contributions aligned with these themes are preferred:
- provider abstraction and fallback safety
- persistent memory
- RAG quality and source-grounded outputs
- observability and reliability
