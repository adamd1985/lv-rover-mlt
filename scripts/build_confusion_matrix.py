"""Build a per-character substitution confusion matrix from synth pairs.

Walks `data/mira_pairs/`, uses the existing `tesseract` field in each row
(already computed during pair generation) plus the gold string. Aligns
char-by-char via edit distance, logs every (gold_char, tess_char) pair.
Aggregates `P(gold | tess)` = count(gold, tess) / count(tess).

Output: `data/tess_confusion.json` with structure:
    {
      "by_tess_char": {
        "c": {"c": 0.92, "ċ": 0.03, "e": 0.02, ...},
        "I": {"l": 0.04, "I": 0.85, "i": 0.06, ...},
        ...
      },
      "n_observations": int,
    }

This is the inverse-confusion lookup: given Tesseract emitted `tess`,
what's the most likely true character? Useful for targeted single-char
corrections during post-processing.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EN_DASH = "–"
EM_DASH = "—"
SOFT_HYPHEN = "­"
MIN_OBS = 5  # ignore (tess, gold) pairs with fewer observations


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.replace(SOFT_HYPHEN, "").replace(EN_DASH, EM_DASH))


def _align_chars(a: str, b: str):
    """Wagner-Fischer with alignment trace. Returns list of (a_char|None, b_char|None)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub)
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                out.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((a[i - 1], None)); i -= 1; continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            out.append((None, b[j - 1])); j -= 1; continue
        break
    out.reverse()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/mira_pairs")
    ap.add_argument("--out", default="data/tess_confusion.json")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    # counts[tess_ch][gold_ch] = n
    counts: dict = defaultdict(lambda: defaultdict(int))
    tess_totals: dict = defaultdict(int)
    n_pairs = 0

    # Limit to L0+L1 (single-line content) where Wagner-Fischer cost is
    # manageable (~80 chars max per pair). Diacritic confusions are
    # length-independent so the matrix still captures the relevant signal.
    skip_limit = 200  # cap per-pair length to keep WF fast
    for jp in sorted(data_dir.rglob("shard_*.jsonl")):
        bucket = jp.parent.name
        if bucket not in ("L0", "L1", "L2"):
            continue
        with jp.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                gold = _norm(r.get("gold", ""))[:skip_limit]
                tess = _norm(r.get("tesseract", ""))[:skip_limit]
                if not gold or not tess:
                    continue
                n_pairs += 1
                for g, t in _align_chars(gold, tess):
                    if t is None or g is None:
                        continue
                    counts[t][g] += 1
                    tess_totals[t] += 1

    by_tess_char: dict = {}
    for t, g_dict in counts.items():
        total = tess_totals[t]
        if total < MIN_OBS:
            continue
        # Keep top-10 candidates by frequency.
        items = sorted(g_dict.items(), key=lambda kv: -kv[1])[:10]
        by_tess_char[t] = {g: round(c / total, 5) for g, c in items}

    out_path = ROOT / args.out
    out_path.write_text(json.dumps({
        "n_pairs": n_pairs,
        "n_observations": sum(tess_totals.values()),
        "by_tess_char": by_tess_char,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[confusion] pairs scanned: {n_pairs}")
    print(f"[confusion] chars observed: {sum(tess_totals.values())}")
    print(f"[confusion] tess chars with >={MIN_OBS} obs: {len(by_tess_char)}")
    # Show a few interesting entries
    interesting = ["c", "C", "g", "G", "h", "H", "z", "Z", "I", "l", "i"]
    for ch in interesting:
        if ch in by_tess_char:
            top3 = list(by_tess_char[ch].items())[:3]
            print(f"  tess={ch!r:5s}: {top3}")
    print(f"[confusion] -> {out_path}")


if __name__ == "__main__":
    main()
