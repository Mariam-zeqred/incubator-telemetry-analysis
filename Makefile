.PHONY: install check test lint figures notebooks clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

test:
	$(PY) -m pytest -q

# "make check" is the smoke test: lint + unit tests + regenerate figures from the real data.
check: lint test figures

figures:
	$(PY) scripts/make_figures.py

notebooks:
	$(PY) scripts/build_notebooks.py

clean:
	rm -rf .pytest_cache .ruff_cache figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
