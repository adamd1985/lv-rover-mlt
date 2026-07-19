"""Variant audit orchestrator.

Compares a wave-2 variant against the Track A baseline on the same paired
predictions and emits a KEEP / DROP / INSIDE NOISE verdict.

Decisions are taken on the normalised CER delta (clitic-space-normalised, see
Raw CER is reported alongside for cross-check with the organiser
leaderboard but never drives the verdict. The pair-bootstrap CI on the delta
is the primary signal; the shuffle null and CV give corroboration.

Bucket and canary tables run their own paired bootstrap on the subsets and
are joint-FDR-corrected so a long table cannot wallpaper false positives.

Entry point: :func:`audit_variant`. Thin CLI under ``python -m
src.eval.audit_runner``.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import audit as _audit
from . import buckets as _buckets
from . import cer as _cer
from .audit import (
    _bh_fdr,
    _cer_from_counts,
    _percentile,
    pair_bootstrap_delta,
    shuffle_test,
)
from .cer import _edit_count, clitic_space_normalise, compute_cer_dual, normalise


CANARY_CHARS_AUDIT: Tuple[str, ...] = (
    "Ċ", "ċ", "Ġ", "ġ", "Ħ", "ħ", "Ż", "ż", "–", "—",
)


@dataclass
class AuditReport:
    variant_name: str
    n_pairs: int
    baseline_cer: Dict[str, float]
    variant_cer: Dict[str, float]
    delta_normalised: float
    delta_ci: Tuple[float, float]
    shuffle_p: float
    cv_baseline: Dict[str, object]
    cv_variant: Dict[str, object]
    per_bucket: Dict[str, Dict[str, float]]
    per_canary: Dict[str, Dict[str, float]]
    verdict: str
    md_path: Optional[Path] = None
    json_path: Optional[Path] = None
    extras: Dict[str, object] = field(default_factory=dict)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _align(
    baseline_rows: Sequence[dict],
    variant_rows: Sequence[dict],
    gold_rows: Sequence[dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[List[str]]]:
    base = {r["id"]: r for r in baseline_rows}
    var = {r["id"]: r for r in variant_rows}
    gold = {r["id"]: r for r in gold_rows}
    ids = [i for i in gold if i in base and i in var]
    refs = [gold[i].get("gold", gold[i].get("paragraph", "")) for i in ids]
    hyps_b = [base[i].get("pred", base[i].get("hypothesis", "")) for i in ids]
    hyps_v = [var[i].get("pred", var[i].get("hypothesis", "")) for i in ids]
    line_lists = [list(gold[i].get("lines") or []) for i in ids]
    return ids, refs, hyps_b, hyps_v, line_lists


def _paired_delta_on_normalised(
    refs: Sequence[str],
    hyps_b: Sequence[str],
    hyps_v: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Pair-bootstrap CI of CER(variant) - CER(baseline) on clitic-normalised
    strings. Sign: negative delta means variant is better than baseline.
    """
    refs_n = [clitic_space_normalise(r) for r in refs]
    hyps_b_n = [clitic_space_normalise(h) for h in hyps_b]
    hyps_v_n = [clitic_space_normalise(h) for h in hyps_v]
    return pair_bootstrap_delta(refs_n, hyps_v_n, hyps_b_n, n_boot=n_boot, seed=seed)


def _shuffle_null_on_delta(
    refs: Sequence[str],
    hyps_b: Sequence[str],
    hyps_v: Sequence[str],
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    """Null: variant and baseline labels on each pair are exchangeable. We
    flip the (variant, baseline) assignment per pair with p=0.5 and recompute
    the delta. p = (#|null_delta| >= |obs| + 1) / (n_perm + 1).
    """
    refs_n = [clitic_space_normalise(r) for r in refs]
    triples: List[Tuple[int, int, int]] = []
    for r, hb, hv in zip(
        refs_n,
        [clitic_space_normalise(h) for h in hyps_b],
        [clitic_space_normalise(h) for h in hyps_v],
    ):
        eb, nb = _edit_count(r, hb)
        ev, nv = _edit_count(r, hv)
        if nb == 0 or nv == 0 or nb != nv:
            continue
        triples.append((ev, eb, nb))
    if not triples:
        return {"observed": 0.0, "p_value": 1.0}
    obs_num_v = sum(t[0] for t in triples)
    obs_num_b = sum(t[1] for t in triples)
    den = sum(t[2] for t in triples)
    obs = (obs_num_v - obs_num_b) / den if den else 0.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        nv = 0
        nb = 0
        for ev, eb, _n in triples:
            if rng.random() < 0.5:
                nv += eb
                nb += ev
            else:
                nv += ev
                nb += eb
        null = (nv - nb) / den
        if abs(null) >= abs(obs):
            hits += 1
    return {"observed": obs, "p_value": (hits + 1) / (n_perm + 1)}


def _cv_grouped(
    ids: Sequence[str],
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    k: int,
    seed: int,
) -> Dict[str, object]:
    """k-fold CV that splits on the paragraph id, not by row index. With one
    prediction per id this is equivalent to a row split, but the API is the
    one we want once a paragraph can appear in multiple rows (e.g. TTA)."""
    refs_n = [clitic_space_normalise(r) for r in refs]
    hyps_n = [clitic_space_normalise(h) for h in hyps]
    unique_ids = list(dict.fromkeys(ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    folds_ids: List[List[str]] = [[] for _ in range(k)]
    for i, pid in enumerate(unique_ids):
        folds_ids[i % k].append(pid)
    id_to_fold = {pid: f for f, pids in enumerate(folds_ids) for pid in pids}
    fold_rows: List[List[Tuple[str, str]]] = [[] for _ in range(k)]
    for pid, r, h in zip(ids, refs_n, hyps_n):
        fold_rows[id_to_fold[pid]].append((r, h))
    fold_cers: List[float] = []
    for rows in fold_rows:
        if not rows:
            fold_cers.append(float("nan"))
            continue
        pair_counts: List[Tuple[int, int]] = []
        for r, h in rows:
            ed, n = _edit_count(r, h)
            if n:
                pair_counts.append((ed, n))
        fold_cers.append(_cer_from_counts(pair_counts))
    finite = [x for x in fold_cers if x == x]
    mean = sum(finite) / len(finite) if finite else 0.0
    var = sum((x - mean) ** 2 for x in finite) / max(len(finite) - 1, 1)
    return {"fold_cers": fold_cers, "mean": mean, "std": var ** 0.5}


def _per_bucket_paired_delta(
    refs: Sequence[str],
    hyps_b: Sequence[str],
    hyps_v: Sequence[str],
    line_lists: Sequence[Sequence[str]],
    *,
    n_boot: int,
    fdr_alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Per-bucket pair-bootstrap delta + BH-FDR over buckets jointly."""
    refs_n_for_bucket = refs  # bucket tagging works on raw ref text
    quartiles = _buckets.compute_quartiles(refs_n_for_bucket)
    tags = _buckets.tag_corpus(refs_n_for_bucket, line_lists, quartiles)
    by_bucket_idx: Dict[str, List[int]] = defaultdict(list)
    for i, tag_set in enumerate(tags):
        for t in tag_set:
            by_bucket_idx[t].append(i)
    out: Dict[str, Dict[str, float]] = {}
    pvals: List[float] = []
    keys: List[str] = []
    refs_norm = [clitic_space_normalise(r) for r in refs]
    hyps_b_norm = [clitic_space_normalise(h) for h in hyps_b]
    hyps_v_norm = [clitic_space_normalise(h) for h in hyps_v]
    for bucket, idxs in sorted(by_bucket_idx.items()):
        sub_r = [refs_norm[i] for i in idxs]
        sub_b = [hyps_b_norm[i] for i in idxs]
        sub_v = [hyps_v_norm[i] for i in idxs]
        d = pair_bootstrap_delta(sub_r, sub_v, sub_b, n_boot=n_boot, seed=seed)
        triples: List[Tuple[int, int, int]] = []
        for r, hb, hv in zip(sub_r, sub_b, sub_v):
            eb, nb = _edit_count(r, hb)
            ev, nv = _edit_count(r, hv)
            if nb and nv and nb == nv:
                triples.append((ev, eb, nb))
        if triples:
            rng = random.Random(seed + 7)
            n_t = len(triples)
            draws: List[float] = []
            for _ in range(n_boot):
                sample = [triples[rng.randrange(n_t)] for _ in range(n_t)]
                a = sum(t[0] for t in sample)
                b = sum(t[1] for t in sample)
                dn = sum(t[2] for t in sample)
                draws.append((a - b) / dn if dn else 0.0)
            cnt = sum(1 for x in draws if x * d["delta"] <= 0)
            p = (cnt + 1) / (n_boot + 1)
        else:
            p = 1.0
        out[bucket] = {**d, "p_value": p, "significant": False}
        keys.append(bucket)
        pvals.append(p)
    sig = _bh_fdr(pvals, alpha=fdr_alpha)
    for k, s in zip(keys, sig):
        out[k]["significant"] = bool(s)
    return out


def _per_canary_paired(
    refs: Sequence[str],
    hyps_b: Sequence[str],
    hyps_v: Sequence[str],
    *,
    chars: Sequence[str],
    n_boot: int,
    fdr_alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    refs_n = [clitic_space_normalise(r) for r in refs]
    hyps_b_n = [clitic_space_normalise(h) for h in hyps_b]
    hyps_v_n = [clitic_space_normalise(h) for h in hyps_v]
    out: Dict[str, Dict[str, float]] = {}
    pvals: List[float] = []
    keys: List[str] = []
    for ch in chars:
        sub_idx = [i for i, r in enumerate(refs_n) if ch in r]
        sub_r = [refs_n[i] for i in sub_idx]
        sub_b = [hyps_b_n[i] for i in sub_idx]
        sub_v = [hyps_v_n[i] for i in sub_idx]
        if not sub_r:
            out[ch] = {"delta": float("nan"), "ci_lo": float("nan"),
                       "ci_hi": float("nan"), "n": 0, "p_value": 1.0,
                       "significant": False}
            keys.append(ch)
            pvals.append(1.0)
            continue
        d = pair_bootstrap_delta(sub_r, sub_v, sub_b, n_boot=n_boot, seed=seed)
        triples: List[Tuple[int, int, int]] = []
        for r, hb, hv in zip(sub_r, sub_b, sub_v):
            eb, nb = _edit_count(r, hb)
            ev, nv = _edit_count(r, hv)
            if nb and nv and nb == nv:
                triples.append((ev, eb, nb))
        if triples:
            rng = random.Random(seed + 11)
            n_t = len(triples)
            draws: List[float] = []
            for _ in range(n_boot):
                sample = [triples[rng.randrange(n_t)] for _ in range(n_t)]
                a = sum(t[0] for t in sample)
                b = sum(t[1] for t in sample)
                dn = sum(t[2] for t in sample)
                draws.append((a - b) / dn if dn else 0.0)
            cnt = sum(1 for x in draws if x * d["delta"] <= 0)
            p = (cnt + 1) / (n_boot + 1)
        else:
            p = 1.0
        out[ch] = {**d, "p_value": p, "significant": False}
        keys.append(ch)
        pvals.append(p)
    sig = _bh_fdr(pvals, alpha=fdr_alpha)
    for k, s in zip(keys, sig):
        out[k]["significant"] = bool(s)
    return out


def _decide(delta: float, ci_lo: float, ci_hi: float) -> str:
    """Verdict on the normalised-CER delta. Sign convention: delta = variant -
    baseline, so negative is better.
    """
    if ci_lo <= 0.0 <= ci_hi:
        return "INSIDE NOISE"
    if ci_hi < 0.0:
        return "KEEP"
    return "DROP"


def _fmt(x: object) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.4f}"
    return str(x)


def _render_markdown(rep: AuditReport) -> str:
    lines: List[str] = []
    lines.append(f"# Audit: {rep.variant_name}")
    lines.append("")
    lines.append(f"n_pairs: {rep.n_pairs}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    b = rep.baseline_cer
    v = rep.variant_cer
    lines.append(
        f"baseline CER: raw {_fmt(b.get('raw_cer'))} / normalised "
        f"{_fmt(b.get('normalised_cer'))}"
    )
    lines.append(
        f"variant  CER: raw {_fmt(v.get('raw_cer'))} / normalised "
        f"{_fmt(v.get('normalised_cer'))}"
    )
    lo, hi = rep.delta_ci
    lines.append(
        f"delta (normalised, variant - baseline): {_fmt(rep.delta_normalised)} "
        f"(95% CI {_fmt(lo)} - {_fmt(hi)})"
    )
    lines.append(f"shuffle null p-value: {_fmt(rep.shuffle_p)}")
    lines.append("")
    lines.append(f"**Verdict: {rep.verdict}**")
    lines.append("")
    lines.append("## CV (k-fold, grouped by paragraph id)")
    lines.append("")
    lines.append("| side | fold CERs | mean | std |")
    lines.append("|---|---|---:|---:|")
    cv_b = rep.cv_baseline
    cv_v = rep.cv_variant
    lines.append(
        f"| baseline | {', '.join(_fmt(x) for x in cv_b['fold_cers'])} | "
        f"{_fmt(cv_b['mean'])} | {_fmt(cv_b['std'])} |"
    )
    lines.append(
        f"| variant | {', '.join(_fmt(x) for x in cv_v['fold_cers'])} | "
        f"{_fmt(cv_v['mean'])} | {_fmt(cv_v['std'])} |"
    )
    lines.append("")
    lines.append("## Per-bucket delta (BH-FDR over buckets)")
    lines.append("")
    lines.append("| bucket | n | delta | 95% CI | p | sig |")
    lines.append("|---|---:|---:|---|---:|---|")
    for k in sorted(rep.per_bucket.keys()):
        row = rep.per_bucket[k]
        ci = f"{_fmt(row.get('ci_lo'))} - {_fmt(row.get('ci_hi'))}"
        sig = "yes" if row.get("significant") else ""
        lines.append(
            f"| {k} | {row.get('n')} | {_fmt(row.get('delta'))} | {ci} | "
            f"{_fmt(row.get('p_value'))} | {sig} |"
        )
    lines.append("")
    lines.append("## Canary delta (BH-FDR over canaries)")
    lines.append("")
    lines.append("| char | n | delta | 95% CI | p | sig |")
    lines.append("|---|---:|---:|---|---:|---|")
    for ch, row in rep.per_canary.items():
        ci = f"{_fmt(row.get('ci_lo'))} - {_fmt(row.get('ci_hi'))}"
        sig = "yes" if row.get("significant") else ""
        lines.append(
            f"| `{ch}` (U+{ord(ch):04X}) | {row.get('n')} | "
            f"{_fmt(row.get('delta'))} | {ci} | {_fmt(row.get('p_value'))} | "
            f"{sig} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _to_json_dict(rep: AuditReport) -> dict:
    return {
        "variant_name": rep.variant_name,
        "n_pairs": rep.n_pairs,
        "baseline_cer": rep.baseline_cer,
        "variant_cer": rep.variant_cer,
        "delta_normalised": rep.delta_normalised,
        "delta_ci": list(rep.delta_ci),
        "shuffle_p": rep.shuffle_p,
        "cv_baseline": rep.cv_baseline,
        "cv_variant": rep.cv_variant,
        "per_bucket": rep.per_bucket,
        "per_canary": rep.per_canary,
        "verdict": rep.verdict,
    }


def audit_variant(
    baseline_preds_path: Path,
    variant_preds_path: Path,
    gold_path: Path,
    out_dir: Path,
    *,
    variant_name: str,
    n_bootstrap: int = 1000,
    n_shuffle: int = 1000,
    cv_folds: int = 5,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> AuditReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _read_jsonl(Path(baseline_preds_path))
    var_rows = _read_jsonl(Path(variant_preds_path))
    gold_rows = _read_jsonl(Path(gold_path))
    ids, refs, hyps_b, hyps_v, line_lists = _align(base_rows, var_rows, gold_rows)
    if not ids:
        raise RuntimeError("no overlap between baseline, variant, gold")

    base_dual = compute_cer_dual(refs, hyps_b)
    var_dual = compute_cer_dual(refs, hyps_v)

    d = _paired_delta_on_normalised(
        refs, hyps_b, hyps_v, n_boot=n_bootstrap, seed=seed
    )
    shuf = _shuffle_null_on_delta(
        refs, hyps_b, hyps_v, n_perm=n_shuffle, seed=seed
    )
    cv_b = _cv_grouped(ids, refs, hyps_b, k=cv_folds, seed=seed)
    cv_v = _cv_grouped(ids, refs, hyps_v, k=cv_folds, seed=seed)
    per_bucket = _per_bucket_paired_delta(
        refs, hyps_b, hyps_v, line_lists,
        n_boot=n_bootstrap, fdr_alpha=fdr_alpha, seed=seed,
    )
    per_canary = _per_canary_paired(
        refs, hyps_b, hyps_v,
        chars=CANARY_CHARS_AUDIT,
        n_boot=n_bootstrap, fdr_alpha=fdr_alpha, seed=seed,
    )
    verdict = _decide(d["delta"], d["ci_lo"], d["ci_hi"])

    rep = AuditReport(
        variant_name=variant_name,
        n_pairs=len(ids),
        baseline_cer=base_dual,
        variant_cer=var_dual,
        delta_normalised=d["delta"],
        delta_ci=(d["ci_lo"], d["ci_hi"]),
        shuffle_p=shuf["p_value"],
        cv_baseline=cv_b,
        cv_variant=cv_v,
        per_bucket=per_bucket,
        per_canary=per_canary,
        verdict=verdict,
    )

    md_path = out_dir / f"_audit_{variant_name}.md"
    json_path = out_dir / f"_audit_{variant_name}.json"
    md_path.write_text(_render_markdown(rep), encoding="utf-8")
    json_path.write_text(
        json.dumps(_to_json_dict(rep), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rep.md_path = md_path
    rep.json_path = json_path
    return rep


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="audit_runner")
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--variant", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-shuffle", type=int, default=1000)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    rep = audit_variant(
        a.baseline, a.variant, a.gold, a.out,
        variant_name=a.name,
        n_bootstrap=a.n_bootstrap,
        n_shuffle=a.n_shuffle,
        cv_folds=a.cv_folds,
        fdr_alpha=a.fdr_alpha,
        seed=a.seed,
    )
    print(f"{rep.variant_name}: delta={rep.delta_normalised:.4f} "
          f"CI=({rep.delta_ci[0]:.4f}, {rep.delta_ci[1]:.4f}) "
          f"verdict={rep.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
