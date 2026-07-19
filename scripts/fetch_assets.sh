#!/usr/bin/env bash
# Pull the model weights, lexicon and confusion table from Hugging Face.
#
# competition_transcriber.py downloads these itself on first construction;
# running this ahead of time makes that first run offline and lets you inspect
# what the pipeline actually loads.
#
#   scripts/fetch_assets.sh            # weights, lexicon, confusion table
#   scripts/fetch_assets.sh --fonts    # also fetch the synth font pool
set -euo pipefail

REPO="${DOCENG_HF_REPO:-radmada/lv-rover-mlt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> fetching $REPO"
python3 - "$REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1])
print(f"    cached at {path}")
PY

if [[ "${1:-}" == "--fonts" ]]; then
  DEST="$ROOT/data/fonts/printed"
  mkdir -p "$DEST"
  echo "==> fetching fonts into $DEST"
  # DejaVu (public domain / Bitstream Vera derivative) covers every Maltese
  # canary glyph and is enough to regenerate synthetic shards. The full 68-face
  # pool used for the paper mixes SIL OFL, GUST and Apache 2.0 faces; see
  # src/datagen/check_fonts.py to validate any face you add yourself.
  if command -v fc-list >/dev/null 2>&1; then
    fc-list --format='%{file}\n' 2>/dev/null \
      | grep -Ei 'dejavu(serif|sans)[^/]*\.ttf$' \
      | while read -r f; do cp -n "$f" "$DEST/" 2>/dev/null || true; done
  fi
  n=$(find "$DEST" -name '*.ttf' 2>/dev/null | wc -l)
  echo "    $n faces available"
  if [[ "$n" -eq 0 ]]; then
    echo "    install system fonts first, e.g. apt-get install fonts-dejavu"
  fi
  echo "==> validating diacritic coverage"
  cd "$ROOT" && python3 -m src.datagen.check_fonts || true
fi

echo "done"
