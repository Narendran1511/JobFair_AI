# JobFair AI — Makefile
# Usage: make <target>
# Windows users: run equivalent commands in PowerShell (see README)

.PHONY: install install-dev run test lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	streamlit run app/Home.py

test:
	pytest tests/ -v --tb=short --cov=core --cov-report=term-missing

lint:
	ruff check . --fix

check:
	python setup_check.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
