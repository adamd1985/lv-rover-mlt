#!/usr/bin/env bash
# End-to-end smoke run: check the environment, fetch weights, transcribe the
# bundled fixtures, and run the test suite.
#
#   ./run.sh              # full check
#   ./run.sh --no-fetch   # skip the Hugging Face download (already cached)
#   ./run.sh --tests-only # just run pytest
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

FETCH=1; TESTS_ONLY=0
for a in "$@"; do
  case "$a" in
    --no-fetch) FETCH=0 ;;
    --tests-only) TESTS_ONLY=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  esac
done

fail=0
step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    ok   %s\n' "$1"; }
bad()  { printf '    FAIL %s\n' "$1"; fail=1; }

step "environment"
python3 -c 'import sys; assert sys.version_info>=(3,9)' 2>/dev/null \
  && ok "python $(python3 -V 2>&1 | cut -d' ' -f2)" || bad "python 3.9+ required"

if command -v tesseract >/dev/null 2>&1; then
  ok "tesseract $(tesseract --version 2>&1 | head -1 | cut -d' ' -f2)"
else
  bad "tesseract not on PATH - install tesseract-ocr (5.x)"
fi

for m in PIL pytesseract malti huggingface_hub; do
  python3 -c "import $m" 2>/dev/null && ok "$m" || bad "$m missing - pip install -r requirements.txt"
done

if [[ "$TESTS_ONLY" -eq 0 ]]; then
  if [[ "$fail" -ne 0 ]]; then
    printf '\nenvironment incomplete; fix the above and re-run\n'; exit 1
  fi

  if [[ "$FETCH" -eq 1 ]]; then
    step "weights"
    bash scripts/fetch_assets.sh || bad "could not fetch weights"
  fi

  step "transcribing fixtures"
  python3 - <<'PY'
import json, sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, ".")
try:
    from competition_transcriber import CompetitionTranscriber
except Exception as e:
    print(f"    FAIL import: {e}"); sys.exit(1)

dev = Path("fixtures/dev")
rows = json.loads((dev / "texts.json").read_text(encoding="utf-8"))
try:
    t = CompetitionTranscriber()
except Exception as e:
    print(f"    FAIL init: {e}")
    print("    weights unavailable; run scripts/fetch_assets.sh")
    sys.exit(1)

import jiwer
refs, hyps = [], []
for r in rows:
    hyp = t.transcribe(Image.open(dev / r["image"]))
    refs.append(r["text"]); hyps.append(hyp)
    mark = "ok  " if hyp.strip() else "EMPTY"
    print(f"    {mark} {r['image']}: {hyp[:60]}")
out = jiwer.process_characters(refs, hyps)
cer = (out.substitutions + out.deletions + out.insertions) / sum(len(x) for x in refs)
print(f"\n    fixture CER: {cer:.4f}  (synthetic render, not the benchmark)")
PY
  [[ $? -ne 0 ]] && bad "fixture transcription"
fi

step "tests"
if python3 -c "import pytest" 2>/dev/null; then
  python3 -m pytest -q tests/ 2>&1 | tail -15
  [[ ${PIPESTATUS[0]} -ne 0 ]] && bad "pytest"
else
  printf '    skipped (pip install pytest)\n'
fi

printf '\n'
if [[ "$fail" -eq 0 ]]; then printf '\033[1mall checks passed\033[0m\n'; else printf '\033[1msome checks failed\033[0m\n'; fi
exit "$fail"
