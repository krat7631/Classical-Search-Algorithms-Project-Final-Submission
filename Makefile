PYTHON ?= python3

.PHONY: install test lint experiments benchmark figures reproduce

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

# Course experiments: theory sweep and constrained sweep on generated grids (seconds).
experiments:
	$(PYTHON) main.py --theory-sweep --max-size 10 --format json > results/theory_results.json
	$(PYTHON) export_to_excel.py results/theory_results.json results/final_theory.csv
	$(PYTHON) main.py --sweep --max-size 10 --format json > results/sweep_results.json
	$(PYTHON) export_to_excel.py results/sweep_results.json results/sweep_results.csv

# Moving AI benchmark: downloads ~6.5 MB of maps, then about 5 minutes of search.
benchmark:
	$(PYTHON) scripts/download_benchmarks.py
	$(PYTHON) -m evaluation.benchmark
	$(PYTHON) -m evaluation.summarize_benchmark

figures:
	MPLBACKEND=Agg $(PYTHON) make_figures.py

reproduce: experiments benchmark figures
