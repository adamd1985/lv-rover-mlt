"""Forensic analysis of joiner round-trip failures on shard_0002.

Loads every sample in the shard, replays the joiner round-trip exactly as
scripts/validate_shard.py does, captures the full failure list (not just the
first 5), classifies each into a failure type, and prints worked examples.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.joiner import join_lines

SOFT_HYPHEN = "­"
EN_DASH = "–"
EM_DASH = "—"
ASCII_HYPHEN = "-"

NUMBERED_BULLET = re.compile(r"^\s*\d+(\.\d+)+\s")
LEADING_DASH = re.compile(r"^\s*[-–—]\s")
STRUCTURAL_PREFIX = re.compile(r"\b(il|id|in|im|ir|is|it|ix|iz|i[ċġż]|l|ta'|fis|tas|tal|ta|bil|bħal|għal|fl|min|fis|fit|fil|fid|fir|tas|tan|tat|tax|taz)-", re.IGNORECASE)


def gold_paragraph(parts: List[str]) -> str:
    return unicodedata.normalize("NFC", "".join(parts).replace(SOFT_HYPHEN, ""))


def joined_via_joiner(parts: List[str]) -> str:
    joined = join_lines(list(parts))
    return unicodedata.normalize("NFC", joined.replace(SOFT_HYPHEN, ""))


def diff_index(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def classify(parts: List[str], joined: str, gold: str) -> Tuple[str, str]:
    """Return (type_code, rationale).

    The dominant pattern (Type G): a corpus token containing an ASCII
    hyphen mid-word (e.g. `Marie-Louise`, `notice-board`, `eks-registratur`)
    happened to land such that the line wrap split it at the hyphen. The
    label keeps `parts[i].endswith('-')` and `parts[i+1]` begins with the
    second compound member. malti.RBLineJoiner correctly identifies this as
    a line-break hyphen (the last word is not in the structural-clitic
    denylist) and removes the hyphen, producing `MarieLouise`. The gold
    side `''.join(parts)` keeps the hyphen. Both sides are internally
    consistent; the disagreement is structural to the rendered-corpus
    pipeline, not a joiner bug.

    Other types still recognised below for completeness.
    """
    j2 = " ".join(joined.split()).replace(" ", "")
    g2 = " ".join(gold.split()).replace(" ", "")
    idx = diff_index(j2, g2)
    around_j = j2[max(0, idx - 40):idx + 40]
    around_g = g2[max(0, idx - 40):idx + 40]

    # Find every line-break wrap point that ends in ASCII hyphen and the
    # previous char is alphabetic. That is the joiner's rejoin trigger.
    wrap_points: List[Tuple[int, str, str]] = []
    for i, p in enumerate(parts[:-1]):
        s = p.rstrip()
        if not s.endswith(ASCII_HYPHEN):
            continue
        if len(s) < 2 or not s[-2].isalpha():
            continue
        if SOFT_HYPHEN in p:
            continue
        nxt = parts[i + 1].lstrip()
        # Determine the joined "word" formed by removing the hyphen.
        # If the original corpus actually had `X-Y` as a compound (with the
        # hyphen as structural), and the wrap broke on it, this is Type G.
        prev_word = s.rstrip(ASCII_HYPHEN).split()[-1] if s.rstrip(ASCII_HYPHEN).split() else ""
        next_word = nxt.split()[0] if nxt.split() else ""
        wrap_points.append((i, prev_word, next_word))

    if wrap_points:
        # Check whether any of these wrap points are reflected in the gold
        # as a preserved ASCII hyphen between prev_word and next_word.
        for (i, pw, nw) in wrap_points:
            needle_gold = (pw + "-" + nw).replace(" ", "")
            needle_join = (pw + nw).replace(" ", "")
            if needle_gold and needle_gold in g2 and needle_join in j2:
                return ("G", f"corpus compound hyphen at wrap line {i}: {pw!r}-{nw!r}")

    # Type D: dash glyph identity differs between joined and gold.
    dash_chars = {ASCII_HYPHEN, EN_DASH, EM_DASH}
    if idx < len(j2) and idx < len(g2):
        if j2[idx] in dash_chars and g2[idx] in dash_chars and j2[idx] != g2[idx]:
            return ("D", f"dash-triplet swap at {idx}: {j2[idx]!r} vs {g2[idx]!r}")

    # Type C: structural Maltese prefix misjoined.
    if idx < len(g2):
        seg_g = g2[max(0, idx - 6):idx + 4]
        if STRUCTURAL_PREFIX.search(seg_g) and ASCII_HYPHEN in seg_g and ASCII_HYPHEN not in j2[max(0, idx - 6):idx + 4]:
            return ("C", f"structural Maltese prefix-hyphen lost near {seg_g!r}")

    # Type A: numbered bullet at a continuation line start.
    for i, p in enumerate(parts[1:], start=1):
        if NUMBERED_BULLET.match(p):
            return ("A", f"numbered bullet at line {i}: {p[:30]!r}")

    # Type B: leading dash on a continuation line.
    for i, p in enumerate(parts[1:], start=1):
        if LEADING_DASH.match(p):
            return ("B", f"leading-dash continuation at line {i}: {p[:30]!r}")

    # Type E: soft-hyphen present, recovery failed for a non-trivial reason.
    if any(SOFT_HYPHEN in p for p in parts):
        return ("E", "soft-hyphen present; non-trivial mismatch")

    return ("F", f"unclassified; divergence at {idx}: j={around_j!r} g={around_g!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    samples = sorted(p for p in out_dir.glob("*.json") if not p.name.startswith("_"))
    print(f"[forensic] scanning {len(samples)} samples in {out_dir}")

    failures: List[Dict] = []
    type_counts: Counter = Counter()
    examples_by_type: Dict[str, List[Dict]] = defaultdict(list)

    for p in samples:
        r = json.loads(p.read_text(encoding="utf-8"))
        parts = r["label_parts"]
        if len(parts) <= 1:
            continue
        joined = joined_via_joiner(parts)
        gold = gold_paragraph(parts)
        j2 = " ".join(joined.split()).replace(" ", "")
        g2 = " ".join(gold.split()).replace(" ", "")
        if j2 == g2:
            continue
        tcode, rationale = classify(parts, joined, gold)
        type_counts[tcode] += 1
        idx = diff_index(j2, g2)
        rec = {
            "file": p.name,
            "type": tcode,
            "rationale": rationale,
            "diff_index": idx,
            "joined_window": j2[max(0, idx - 40):idx + 40],
            "gold_window": g2[max(0, idx - 40):idx + 40],
            "n_lines": len(parts),
            "first_lines": parts[:3],
        }
        failures.append(rec)
        if len(examples_by_type[tcode]) < 3:
            examples_by_type[tcode].append(rec)

    print(f"\n[forensic] total failures: {len(failures)}")
    print(f"[forensic] type counts: {dict(type_counts)}")

    for tcode in sorted(examples_by_type):
        print(f"\n=== Type {tcode} ({type_counts[tcode]} cases) ===")
        for ex in examples_by_type[tcode]:
            print(f"  file={ex['file']} n_lines={ex['n_lines']}")
            print(f"  rationale: {ex['rationale']}")
            print(f"  joined: ...{ex['joined_window']}...")
            print(f"  gold:   ...{ex['gold_window']}...")
            print(f"  parts[0..2]: {ex['first_lines']}")
            print()

    if args.write_report:
        report = {
            "n_failures": len(failures),
            "type_counts": dict(type_counts),
            "failures": failures,
        }
        out_path = out_dir / "_failure_forensic.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[forensic] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
