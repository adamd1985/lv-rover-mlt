.PHONY: help install assets fonts-check smoke test lint corpus synth export finetune eval audit clean

PY ?= python3
export PYTHONPATH := $(CURDIR)

help:
	@echo "install      pip install the replication extras"
	@echo "assets       fetch weights, lexicon and confusion table from Hugging Face"
	@echo "smoke        end-to-end check (./run.sh)"
	@echo "test         pytest"
	@echo "lint         ruff"
	@echo ""
	@echo "fonts-check  validate that fonts render the Maltese diacritics"
	@echo "corpus       pull korpus_malti text shards"
	@echo "synth        render a synthetic paragraph shard"
	@echo "export       cut line crops and .lstmf for tesstrain"
	@echo "finetune     fine-tune the Tesseract LSTM"
	@echo ""
	@echo "eval         stratified per-bucket CER on a dev set"
	@echo "audit        paired bootstrap + permutation test over the full chain"

install:
	$(PY) -m pip install -r requirements-replication.txt

assets:
	bash scripts/fetch_assets.sh

smoke:
	./run.sh

test:
	$(PY) -m pytest -q tests/

lint:
	ruff check .

fonts-check:
	$(PY) -m src.datagen.check_fonts

corpus:
	$(PY) -m src.datagen.pull_corpus --config configs/synth_v1.yaml

synth:
	$(PY) -m src.datagen.render --config configs/synth_v1.yaml --mode paragraph

export:
	$(PY) -m src.datagen.tesstrain_export --config configs/synth_v1.yaml

finetune:
	$(PY) scripts/finetune_tesseract.py

eval:
	$(PY) -m src.eval.stratified --config configs/synth_v1.yaml --dev-dir fixtures/dev

audit:
	$(PY) scripts/audit_bootstrap_full_chain.py
	$(PY) scripts/audit_ensemble_diagnostics.py


clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
