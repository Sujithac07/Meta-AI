.PHONY: setup setup-dev test unit lint run-dashboard run-api

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest -q

unit:
	pytest -q tests/unit

lint:
	ruff check .

run-dashboard:
	python quick_start.py

run-api:
	uvicorn backend_api_main:app --host 127.0.0.1 --port 8000 --reload
