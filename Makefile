.PHONY: help install demo test preflight sample show pipeline api validate clean

help:
	@echo "Call-Centre Radar"
	@echo ""
	@echo "  make install    install python deps"
	@echo "  make test       run contract + validator tests (no API key needed)"
	@echo "  make demo       serve the prebuilt database         (60 seconds)"
	@echo "  make preflight  verify channel/duration assumptions BEFORE bulk run"
	@echo "  make sample     process 10 real calls end to end    (~2 minutes)"
	@echo "  make show       print processed transcripts + evidence to the terminal"
	@echo "  make pipeline   process the FULL dataset            (~1 hour)"
	@echo "  make validate   check attention scores vs customer survey ratings"
	@echo "  make api        run the API on :8000"
	@echo "  make clean      remove generated artefacts"

install:
	pip install -r requirements.txt

# Override with e.g. `make test PYTHON=.venv/bin/python` - the API tests need
# fastapi, which the pipeline tests do not.
PYTHON ?= python3

test:
	ASR_BACKEND=mock LLM_BACKEND=mock $(PYTHON) tests/test_contract.py
	ASR_BACKEND=mock LLM_BACKEND=mock $(PYTHON) tests/test_api.py

# Runs with zero API keys - proves the plumbing works before you spend anything.
smoke:
	ASR_BACKEND=mock LLM_BACKEND=mock python3 scripts/run_batch.py --limit 3 --force

preflight:
	python3 scripts/preflight.py --sample 20

sample:
	python3 scripts/run_batch.py --limit 10

# Eyeball the result: turn order, timestamps, and which side gave the greeting.
show:
	python3 scripts/show_transcript.py --full

pipeline:
	python3 scripts/run_batch.py

validate:
	python3 scripts/validate_scores.py

demo api:
	python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf data/work __pycache__ */__pycache__
	@echo "kept data/callradar.db - delete it manually if you really mean to"
