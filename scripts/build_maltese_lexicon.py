"""Build Maltese word frequency table from local corpus sources.

Used by the lexicon router to decide whether a neural-OCR word is a real
Maltese word that should override the Tesseract anchor on diacritic-bearing
disagreements.

Sources walked:
- `data/mira_pairs/{L0..L4}/shard_*.jsonl` (gold strings)
- `data/synth/shard_0002_korpus/*.json` (label fields)

Output: `data/maltese_lexicon.json` mapping word -> count (lowercased after
NFC normalisation, stripped of punctuation).

Run:
    /tmp/judges_env/bin/python scripts/build_maltese_lexicon.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUNCT = ".,;:!?\"'()[]{}«»—–-"


def _norm_word(w: str) -> str:
    w = unicodedata.normalize("NFC", w).strip(PUNCT)
    return w


def _iter_corpus_texts():
    # mira_pairs gold strings
    for jp in sorted((ROOT / "data" / "mira_pairs").rglob("shard_*.jsonl")):
        with jp.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                yield r.get("gold", "")
    # synth corpus labels
    syn = ROOT / "data" / "synth" / "shard_0002_korpus"
    if syn.is_dir():
        for jp in sorted(syn.glob("*.json")):
            try:
                r = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            yield r.get("label", "")


def main() -> None:
    counts: Counter = Counter()
    n_docs = 0
    for text in _iter_corpus_texts():
        if not text:
            continue
        n_docs += 1
        for w in text.split():
            nw = _norm_word(w)
            if 2 <= len(nw) <= 40:
                counts[nw] += 1
                counts[nw.lower()] += 1  # also store lowercased

    out_path = ROOT / "data" / "maltese_lexicon.json"
    # Keep only words seen >= 2 times to filter typos
    pruned = {w: c for w, c in counts.items() if c >= 2}
    out_path.write_text(json.dumps(pruned, ensure_ascii=False), encoding="utf-8")
    print(f"[lexicon] docs scanned: {n_docs}")
    print(f"[lexicon] total unique words: {len(counts)}")
    print(f"[lexicon] pruned (count>=2): {len(pruned)}")
    print(f"[lexicon] saved -> {out_path}")


if __name__ == "__main__":
    main()
