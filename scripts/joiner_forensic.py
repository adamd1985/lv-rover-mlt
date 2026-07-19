"""Joiner forensic on synth-val set.

Samples a 500-paragraph stratified slice from a synth shard, simulates a
perfect line-OCR by replaying `label_parts` through the joiner, and reports
per-bucket CER plus categorised failure counts. Reproducible after the
organiser dev set lands by pointing `--shard` at the dev shard once it ships
with `label_parts` (or a derived line split).

Categories of failure (observed in the joiner audit):

  L  line-break hyphen NOT removed (soft-hyphen marker round-trip failed)
  S  structural hyphen WRONGLY removed (e.g. `fis-seħħ` -> `fisseħħ`)
  U  U+2013 / U+2014 mishandled (dash glyph lost, swapped, or space lost)
  P  il-/is-/id- (article paradigm) prefix joined incorrectly
  G  corpus-internal compound hyphen at a wrap point (renderer artefact;
     not a joiner bug, but reported separately)
  C  corpus-spacing artefact: source paragraph contained `<clitic>- foo`
     with a stray space; joiner correctly emitted `<clitic>-foo`; gold
     reconstruction kept the stray space. Not a joiner bug.
  B  leading-bullet `- ` at paragraph start lost its trailing space.
  O  other / unclassified
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.buckets import (
    bucket_il_prefix,
    bucket_language,
    bucket_length,
    bucket_line_hyphen,
    bucket_em_dash,
    compute_quartiles,
)
from src.eval.cer import cer_aggregate, cer_per_bucket
from src.joiner import join_lines

SOFT_HYPHEN = "­"
EN_DASH = "–"
EM_DASH = "—"
ASCII_HYPHEN = "-"
DASH_CHARS = {ASCII_HYPHEN, EN_DASH, EM_DASH}

ARTICLE_FORMS = (
    "il", "id", "is", "it", "in", "ir", "ix", "iz",
    "l", "fil", "fis", "fit", "fid", "fl",
    "bil", "bis", "bit", "bid",
    "tal", "tas", "tat", "tad", "tan", "tar", "tax", "taż",
    "min", "mil", "mis", "mit",
    "mal", "mas", "mat", "mad",
    "mill", "miss", "mitt", "miż",
    "maż", "biż",
    "lil", "lill", "lis", "lid", "lit",
    "għal", "għall", "għas", "għat", "għad", "għaż", "għar", "għax",
    "bħal", "bħall",
)

# Wider list used only for the clitic-space classifier; includes the single
# consonant elision forms that surface mid-text (`d-`, `t-`, `n-`, etc.).
CLITIC_FORMS_WIDE = ARTICLE_FORMS + ("d", "t", "n", "s", "x", "z", "r")
ARTICLE_RE = re.compile(
    r"\b(?:" + "|".join(ARTICLE_FORMS) + r")-[A-Za-zàèìòùċĊġĠħĦżŻ]+",
    re.IGNORECASE,
)

RANDOM_SEED = 42


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def collapse_ws(s: str) -> str:
    return " ".join(s.split())


def gold_from_parts(parts: Sequence[str]) -> str:
    """Reconstruct the gold paragraph per the MalteseParagraph contract:
    rejoin soft-hyphen wrapped words (drop U+00AD with no space), join the
    remaining lines with a single space, NFC, collapse whitespace.

    This differs from validate_shard's `"".join(parts)` reduction. That
    reduction was suitable for round-trip validation (it compared a packed
    string against a packed string) but is not the gold an OCR system is
    expected to produce. The renderer docstring states: 'strip U+00AD and
    collapse newlines/whitespace'. Newlines become spaces, soft hyphens
    vanish with no space.
    """
    joined = " ".join(p or "" for p in parts).replace(SOFT_HYPHEN, "")
    return collapse_ws(nfc(joined))


def joined_from_parts(parts: Sequence[str]) -> str:
    """Replay parts through the joiner exactly as the OCR pipeline will at
    inference time (one line per OCR call, joined post-hoc)."""
    return collapse_ws(nfc(join_lines(list(parts)).replace(SOFT_HYPHEN, "")))


# ------------------------------ tokens & CER -------------------------------


def hyphen_token_cer(refs: Sequence[str], hyps: Sequence[str]) -> Tuple[float, int]:
    """CER restricted to whitespace tokens that contain at least one of
    U+002D, U+2013, U+2014. Per-pair CER averaged over pairs that contribute
    at least one such token. Returns (cer, n_pairs_used).

    Implementation: pull dash-bearing tokens from the ref, find the matching
    span in the hyp by best-effort token alignment using positional order.
    CER is computed via jiwer on the concatenated dash-token strings so the
    aggregate matches the rest of the harness.
    """
    import jiwer

    r_acc: List[str] = []
    h_acc: List[str] = []
    n_pairs = 0
    for r, h in zip(refs, hyps):
        r_toks = [t for t in r.split() if any(c in DASH_CHARS for c in t)]
        if not r_toks:
            continue
        h_toks_all = h.split()
        h_dash = [t for t in h_toks_all if any(c in DASH_CHARS for c in t)]
        # positional alignment: pair by index, pad missing with empty string
        m = max(len(r_toks), len(h_dash))
        r_seg = " ".join(r_toks + [""] * (m - len(r_toks)))
        h_seg = " ".join(h_dash + [""] * (m - len(h_dash)))
        r_acc.append(r_seg)
        h_acc.append(h_seg)
        n_pairs += 1
    if not r_acc:
        return 0.0, 0
    return float(jiwer.cer(r_acc, h_acc)), n_pairs


# ------------------------------ classification ----------------------------


def _wrap_points(parts: Sequence[str]) -> List[Tuple[int, str, str, str]]:
    """Return (i, last_word_with_dash, next_first_word, dash_char) for every
    non-final line whose stripped form ends in -, en-dash, or em-dash."""
    out: List[Tuple[int, str, str, str]] = []
    for i, p in enumerate(parts[:-1]):
        s = (p or "").rstrip()
        if not s:
            continue
        last = s[-1]
        if last not in DASH_CHARS:
            continue
        if len(s) < 2 or not s[-2].isalpha():
            continue
        nxt = (parts[i + 1] or "").lstrip()
        prev_word = s.split()[-1] if s.split() else ""
        next_word = nxt.split()[0] if nxt.split() else ""
        out.append((i, prev_word, next_word, last))
    return out


_CLITIC_SPACE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(set(CLITIC_FORMS_WIDE), key=lambda x: -len(x))) + r")-\s+[0-9A-Za-zàèìòùċĊġĠħĦżŻ]",
    re.IGNORECASE,
)


def classify_failure(parts: Sequence[str], gold: str, hyp: str) -> Tuple[str, str]:
    """Return (code, rationale). Codes: L, S, U, P, G, C, O."""
    # C: gold has one or more `<clitic>- foo` (stray space) sites; the joiner
    # collapsed at least one to `<clitic>-foo`. Treat the failure as C if
    # squashing every clitic-space site in the gold yields the hyp.
    if _CLITIC_SPACE_RE.search(gold):
        squashed = _CLITIC_SPACE_RE.sub(
            lambda m: m.group(0).replace("- ", "-").replace("-\t", "-"),
            gold,
        )
        squashed = collapse_ws(squashed)
        if squashed == hyp:
            return ("C", "corpus stray-space after clitic(s); joiner emitted correct form")
        # the joiner may have collapsed only some sites (it depends on line
        # wrapping). If at least one collapse explains the divergence and the
        # remainder of the diff is also clitic-space, still call it C.
        n = min(len(squashed), len(hyp))
        idx = next((k for k in range(n) if squashed[k] != hyp[k]), n)
        if idx == n and abs(len(squashed) - len(hyp)) <= 2:
            return ("C", "corpus stray-space after clitic(s); near-match")
        # Partial: at least one clitic-space site exists and the hyp has the
        # corresponding squashed form. Common when the joiner only fired on
        # the cross-line clitic.
        for m in _CLITIC_SPACE_RE.finditer(gold):
            site = m.group(0)
            squashed_site = site.replace("- ", "-")
            if squashed_site in hyp and gold.count(site) > hyp.count(site):
                return ("C", f"corpus stray-space after clitic: {site!r}")
    # First, check if the wrap-point compound exists in the joined string
    # as a hyphenless concatenation while the gold keeps the hyphen.
    # This is the rendered-corpus compound failure (Type G).
    g_packed = gold.replace(" ", "")
    h_packed = hyp.replace(" ", "")
    for (i, pw, nw, dash) in _wrap_points(parts):
        needle_gold = (pw.rstrip(dash) + dash + nw)
        needle_join = (pw.rstrip(dash) + nw)
        if needle_gold and needle_gold.replace(" ", "") in g_packed and needle_join.replace(" ", "") in h_packed:
            return ("G", f"corpus compound at wrap line {i}: {pw!r}{dash}{nw!r}")

    # L: line-break hyphen survived (soft-hyphen never collapsed). The gold
    # has the joined word, the hyp keeps a hyphen mid-token.
    soft_lines = [(i, p) for i, p in enumerate(parts) if SOFT_HYPHEN in (p or "")]
    if soft_lines:
        for (i, p) in soft_lines:
            head_tail = p.split(SOFT_HYPHEN, 1)
            if len(head_tail) != 2:
                continue
            head = head_tail[0].split()[-1] if head_tail[0].split() else ""
            tail_seg = head_tail[1]
            tail = tail_seg.split()[0] if tail_seg.split() else ""
            if i + 1 < len(parts) and not tail:
                tail = (parts[i + 1] or "").split()[0] if (parts[i + 1] or "").split() else ""
            target = (head + tail).replace(" ", "")
            broken = (head + "-" + tail).replace(" ", "")
            if target and target in g_packed and broken in h_packed:
                return ("L", f"soft-hyphen survived as `-` at line {i}: {head!r}-{tail!r}")

    # S: structural hyphen removed. Look for `<article>-X` in gold but not in
    # hyp at the same position.
    g_arts = set(m.group(0).lower() for m in ARTICLE_RE.finditer(gold))
    h_arts = set(m.group(0).lower() for m in ARTICLE_RE.finditer(hyp))
    missing = g_arts - h_arts
    if missing:
        a = next(iter(missing))
        # Confirm the de-hyphenated form is in hyp
        squashed = a.replace("-", "")
        if squashed in hyp.lower().replace(" ", ""):
            return ("S", f"structural prefix dropped: {a!r} -> {squashed!r}")

    # B: leading-bullet dash at paragraph start dropped its trailing space.
    # Renderer emitted ` - <text>` as a leading bullet on line 0; the joiner
    # joined to give `-<text>`. The space after the leading dash is part of
    # the gold.
    if gold[:2] == "- " and hyp[:1] == "-" and hyp[:2] != "- ":
        return ("B", "leading-bullet dash lost trailing space")

    # U-em: em-dash followed by space dropped (joiner ran two lines together
    # at an em-dash boundary without keeping the space).
    if EM_DASH in gold:
        # find em-dash positions in gold and check if the following space is
        # missing in hyp.
        ok = True
        for m in re.finditer(EM_DASH + r"\s", gold):
            pos = m.start()
            # locate the same em-dash in hyp by counting occurrences
            occ = gold[:pos].count(EM_DASH)
            hyp_positions = [i for i, c in enumerate(hyp) if c == EM_DASH]
            if occ < len(hyp_positions):
                hpos = hyp_positions[occ]
                if hpos + 1 < len(hyp) and not hyp[hpos + 1].isspace():
                    return ("U", f"em-dash space lost at hyp pos {hpos}")
        del ok

    # U: dash glyph identity differs at the first divergence point.
    n = min(len(gold), len(hyp))
    idx = next((k for k in range(n) if gold[k] != hyp[k]), n)
    if idx < len(gold) and idx < len(hyp):
        gc, hc = gold[idx], hyp[idx]
        if gc in DASH_CHARS and hc in DASH_CHARS and gc != hc:
            return ("U", f"dash swap at {idx}: gold={gc!r} hyp={hc!r}")
        if gc in (EN_DASH, EM_DASH) and hc not in DASH_CHARS:
            return ("U", f"unicode dash lost at {idx}: {gc!r}")

    # P: article prefix presence agrees but case/space around it diverges.
    # E.g. "il- kelb" with stray space in hyp, but `il-kelb` in gold.
    for a in g_arts:
        spaced = a.replace("-", "- ")
        if spaced in hyp.lower():
            return ("P", f"article prefix has stray space: {a!r}")

    return ("O", f"unclassified divergence at char {idx}")


# ------------------------------ sampling ----------------------------------


def load_shard(shard_dir: Path) -> List[Dict]:
    samples = []
    for p in sorted(shard_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "label_parts" not in r:
            continue
        r["_file"] = p.name
        samples.append(r)
    return samples


def stratified_sample(records: List[Dict], n: int, seed: int) -> List[Dict]:
    """Stratified 500-paragraph sample. Each paragraph is tagged by its five
    bucket labels; we draw round-robin from the rarest buckets first so each
    bucket gets at least ceil(n / n_buckets) paragraphs where possible."""
    refs = [gold_from_parts(r["label_parts"]) for r in records]
    qs = compute_quartiles(refs)

    tagged: List[Tuple[Dict, List[str]]] = []
    for r, g in zip(records, refs):
        parts = r["label_parts"]
        tags = [
            bucket_length(g, qs),
            bucket_language(g),
            bucket_il_prefix(g),
            bucket_line_hyphen(parts),
            bucket_em_dash(g),
        ]
        tagged.append((r, tags))

    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for i, (_, tags) in enumerate(tagged):
        for t in tags:
            by_bucket[t].append(i)

    rng = random.Random(seed)
    for v in by_bucket.values():
        rng.shuffle(v)

    chosen: set = set()
    buckets_sorted = sorted(by_bucket, key=lambda k: len(by_bucket[k]))
    while len(chosen) < n:
        progressed = False
        for b in buckets_sorted:
            if len(chosen) >= n:
                break
            for idx in by_bucket[b]:
                if idx not in chosen:
                    chosen.add(idx)
                    progressed = True
                    break
        if not progressed:
            break

    out = [tagged[i][0] for i in sorted(chosen)]
    if len(out) < n:
        extras = [r for r in records if r["_file"] not in {x["_file"] for x in out}]
        rng.shuffle(extras)
        out += extras[: n - len(out)]
    return out[:n], qs


# ------------------------------ main --------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="path to a synth shard dir")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", required=True, help="output JSON report path")
    args = ap.parse_args()

    shard_dir = Path(args.shard)
    records = load_shard(shard_dir)
    print(f"[forensic] loaded {len(records)} records from {shard_dir}")
    if not records:
        return 2

    sample, qs = stratified_sample(records, args.n, args.seed)
    print(f"[forensic] sampled {len(sample)} paragraphs (quartiles={qs})")

    refs: List[str] = []
    hyps: List[str] = []
    tag_lists: List[List[str]] = []
    failures: List[Dict] = []
    type_counts: Counter = Counter()
    n_total_failures = 0

    for r in sample:
        parts = r["label_parts"]
        gold = gold_from_parts(parts)
        hyp = joined_from_parts(parts)
        refs.append(gold)
        hyps.append(hyp)
        tags = [
            bucket_length(gold, qs),
            bucket_language(gold),
            bucket_il_prefix(gold),
            bucket_line_hyphen(parts),
            bucket_em_dash(gold),
        ]
        tag_lists.append(tags)
        if gold != hyp:
            code, rationale = classify_failure(parts, gold, hyp)
            type_counts[code] += 1
            n_total_failures += 1
            # Locate the first divergence point and window the report around
            # it so the entry's top-20 table is informative even on long paras.
            n = min(len(gold), len(hyp))
            diff_idx = next((k for k in range(n) if gold[k] != hyp[k]), n)
            lo = max(0, diff_idx - 60)
            hi = diff_idx + 80
            failures.append({
                "file": r["_file"],
                "type": code,
                "rationale": rationale,
                "buckets": tags,
                "diff_idx": diff_idx,
                "gold_window": gold[lo:hi],
                "hyp_window": hyp[lo:hi],
                "gold_len": len(gold),
                "hyp_len": len(hyp),
                "n_lines": len(parts),
                "lang": r.get("lang"),
            })

    global_cer = cer_aggregate(refs, hyps)
    per_bucket = cer_per_bucket(refs, hyps, tag_lists, min_n=10)
    hcer, hpairs = hyphen_token_cer(refs, hyps)

    report = {
        "shard": str(shard_dir),
        "n_sampled": len(sample),
        "seed": args.seed,
        "quartiles": list(qs),
        "global_cer": global_cer,
        "hyphen_token_cer": hcer,
        "hyphen_token_pairs": hpairs,
        "per_bucket_cer": per_bucket,
        "failure_count": n_total_failures,
        "failure_type_counts": dict(type_counts),
        "failures_top20": failures[:20],
        "failures_all": failures,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[forensic] global CER = {global_cer:.5f}")
    print(f"[forensic] hyphen-token CER = {hcer:.5f} over {hpairs} pairs")
    print(f"[forensic] failures = {n_total_failures} / {len(sample)}")
    print(f"[forensic] type counts: {dict(type_counts)}")
    print(f"[forensic] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
