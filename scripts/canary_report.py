"""Canary monitor for the Maltese diacritic letters (ċ ġ ħ ż + caps).

Two checks, both required before any submission:

1. Per-character canary confusion: for each diacritic letter, how often the
   gold occurrence is recovered vs demoted to its ASCII base vs spurious.
   The pairs ċ/c, ġ/g, ħ/h, ż/z are the canaries for tokenizer and font
   issues.

2. Charset conformance: every character the transcriber emits must be in
   competition_files/char_set.json (117 chars). A model that emits a
   diacritic outside the inventory - or strips one - is the disqualification
   risk at the encoder stage.

Importable (`canary_table`, `charset_conformance`) so the audit scripts can
print it every run; also runnable as a CLI on a preds jsonl + gold jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

CANARIES: List[Tuple[str, str]] = [
    ("ċ", "c"), ("Ċ", "C"), ("ġ", "g"), ("Ġ", "G"),
    ("ħ", "h"), ("Ħ", "H"), ("ż", "z"), ("Ż", "Z"),
]
_CHARSET_PATH = "competition_files/char_set.json"


def _align(a: str, b: str):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            out.append((a[i - 1], b[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    out.reverse()
    return out


def canary_table(gold: Dict[str, str], pred: Dict[str, str]) -> str:
    keys = sorted(set(gold) & set(pred))
    stats = {d: {"gold": 0, "ok": 0, "ascii": 0, "other": 0, "spurious": 0}
             for d, _ in CANARIES}
    for k in keys:
        for g, p in _align(gold[k], pred[k]):
            for d, a in CANARIES:
                if g == d:
                    stats[d]["gold"] += 1
                    if p == d:
                        stats[d]["ok"] += 1
                    elif p == a:
                        stats[d]["ascii"] += 1
                    else:
                        stats[d]["other"] += 1
                if p == d and g != d:
                    stats[d]["spurious"] += 1
    lines = ["canary  gold#  recall  ->ascii  spurious"]
    tg = tok = ta = tsp = 0
    for d, a in CANARIES:
        s = stats[d]
        if s["gold"] == 0 and s["spurious"] == 0:
            continue
        rec = s["ok"] / s["gold"] if s["gold"] else float("nan")
        lines.append(f"  {d}/{a}  {s['gold']:>5d}  {rec:>5.1%}  {s['ascii']:>6d}  {s['spurious']:>7d}")
        tg += s["gold"]; tok += s["ok"]; ta += s["ascii"]; tsp += s["spurious"]
    rec = tok / tg if tg else float("nan")
    lines.append(f"  TOTAL  {tg:>5d}  {rec:>5.1%}  {ta:>6d}  {tsp:>7d}")
    return "\n".join(lines)


def charset_conformance(pred: Dict[str, str], charset_path: str = _CHARSET_PATH) -> Tuple[int, Dict[str, int]]:
    cs = json.loads(Path(charset_path).read_text())
    allowed = set(cs if isinstance(cs, list) else cs.keys())
    allowed |= {"\n", " ", "\t"}
    out_of_set: Dict[str, int] = {}
    for t in pred.values():
        for ch in t:
            if ch not in allowed:
                out_of_set[ch] = out_of_set.get(ch, 0) + 1
    return len(out_of_set), out_of_set


def print_report(gold: Dict[str, str], pred: Dict[str, str]) -> None:
    print("\n--- canary confusion ---")
    print(canary_table(gold, pred))
    n_oos, oos = charset_conformance(pred)
    print("\n--- charset conformance ---")
    if n_oos == 0:
        print("  PASS: all emitted chars are in char_set.json")
    else:
        top = sorted(oos.items(), key=lambda x: -x[1])[:15]
        print(f"  WARN: {n_oos} distinct out-of-inventory chars emitted:")
        for ch, n in top:
            print(f"    U+{ord(ch):04X} {ch!r}: {n}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="outputs/campaign/dev_gold.jsonl")
    ap.add_argument("--pred", default="outputs/audit_v12/dev_v12_preds.jsonl")
    args = ap.parse_args()
    g = {json.loads(l)["id"]: json.loads(l).get("gold", json.loads(l).get("text"))
         for l in Path(args.gold).read_text().splitlines()}
    p = {json.loads(l)["id"]: json.loads(l)["pred"]
         for l in Path(args.pred).read_text().splitlines()}
    print_report(g, p)


if __name__ == "__main__":
    main()
