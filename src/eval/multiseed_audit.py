"""Multi-seed audit implementing the protocol from arXiv 2511.19794.

Runs a candidate at >= 3 seeds, computes paired per-paragraph CER delta vs
the anchor per seed, aggregates with BCa bootstrap (bias-corrected and
accelerated), paired permutation test across seeds, and applies a
sign-consistency gate.

Verdict KEEP requires ALL of:
  1. multi-seed BCa CI excludes 0
  2. paired permutation p < 0.05
  3. sign-consistent across all seeds (all deltas same sign as aggregate)
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .cer import _edit_count, clitic_space_normalise, cer_organiser
from .audit import _cer_from_counts, _percentile, pair_bootstrap_delta


@dataclass
class MultiseedAuditReport:
    candidate_name: str
    n_seeds: int
    n_pairs: int
    anchor_cer: float
    per_seed_cer: List[float]
    per_seed_delta: List[float]
    aggregate_delta: float
    bca_ci: Tuple[float, float]
    permutation_p: float
    sign_consistent: bool
    verdict: str
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


def _align_two(
    anchor_rows: Sequence[dict],
    cand_rows: Sequence[dict],
    gold_rows: Sequence[dict],
) -> Tuple[List[str], List[str], List[str]]:
    anc = {r["id"]: r.get("pred", "") for r in anchor_rows}
    cand = {r["id"]: r.get("pred", "") for r in cand_rows}
    gold = {r["id"]: r for r in gold_rows}
    ids = [i for i in gold if i in anc and i in cand]
    refs = [gold[i].get("gold", gold[i].get("paragraph", "")) for i in ids]
    hyps_a = [anc[i] for i in ids]
    hyps_c = [cand[i] for i in ids]
    return refs, hyps_a, hyps_c


def _delta_from_normalised(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_c: Sequence[str],
) -> float:
    """CER(candidate) - CER(anchor) on clitic-normalised strings."""
    refs_n = [clitic_space_normalise(r) for r in refs]
    hyps_a_n = [clitic_space_normalise(h) for h in hyps_a]
    hyps_c_n = [clitic_space_normalise(h) for h in hyps_c]
    triples = []
    for r, ha, hc in zip(refs_n, hyps_a_n, hyps_c_n):
        ea, na = _edit_count(r, ha)
        ec, nc = _edit_count(r, hc)
        if na and nc and na == nc:
            triples.append((ec, ea, na))
    if not triples:
        return 0.0
    num_c = sum(t[0] for t in triples)
    num_a = sum(t[1] for t in triples)
    den = sum(t[2] for t in triples)
    return (num_c - num_a) / den if den else 0.0


def _bca_bootstrap(
    triples_per_seed: List[List[Tuple[int, int, int]]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """BCa bootstrap CI of the aggregate CER(candidate)-CER(anchor) delta.

    triples_per_seed: one list per seed of (ed_cand, ed_anc, n) tuples.
    Returns (aggregate_delta, ci_lo, ci_hi).
    """
    all_triples: List[Tuple[int, int, int]] = []
    for ts in triples_per_seed:
        all_triples.extend(ts)
    if not all_triples:
        return 0.0, 0.0, 0.0

    def _delta(sample: Sequence[Tuple[int, int, int]]) -> float:
        nc = sum(t[0] for t in sample)
        na = sum(t[1] for t in sample)
        dn = sum(t[2] for t in sample)
        return (nc - na) / dn if dn else 0.0

    observed = _delta(all_triples)
    n = len(all_triples)
    rng = random.Random(seed)
    draws = [_delta([all_triples[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]

    # bias correction z0
    below = sum(1 for d in draws if d < observed)
    z0_arg = max(1e-9, min(below / n_boot, 1 - 1e-9))
    z0 = _norm_ppf(z0_arg)

    # acceleration a via jackknife
    jk_deltas = []
    for i in range(n):
        jk = [all_triples[j] for j in range(n) if j != i]
        jk_deltas.append(_delta(jk))
    jk_mean = sum(jk_deltas) / n
    num_a = sum((jk_mean - x) ** 3 for x in jk_deltas)
    den_a = sum((jk_mean - x) ** 2 for x in jk_deltas)
    accel = num_a / (6.0 * (den_a ** 1.5 + 1e-30))

    z_alpha_lo = _norm_ppf(alpha / 2.0)
    z_alpha_hi = _norm_ppf(1.0 - alpha / 2.0)

    def _adj(z_alpha: float) -> float:
        denom = 1.0 - accel * (z0 + z_alpha)
        if abs(denom) < 1e-9:
            return z_alpha
        return z0 + (z0 + z_alpha) / denom

    p_lo = _norm_cdf(_adj(z_alpha_lo))
    p_hi = _norm_cdf(_adj(z_alpha_hi))

    ci_lo = _percentile(sorted(draws), p_lo)
    ci_hi = _percentile(sorted(draws), p_hi)
    return observed, ci_lo, ci_hi


def _norm_ppf(p: float) -> float:
    """Rational approximation to the standard normal quantile (Abramowitz & Stegun)."""
    p = max(1e-12, min(p, 1 - 1e-12))
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2.0 * math.log(1 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return t - num / den


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _permutation_test(
    triples_per_seed: List[List[Tuple[int, int, int]]],
    *,
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """Paired permutation test across seeds.

    For each seed, we have a list of (ed_cand, ed_anc, n) pairs. We randomly
    flip each pair's (cand, anc) assignment with p=0.5, recompute the pooled
    delta, and count how often |null| >= |observed|.
    """
    all_triples: List[Tuple[int, int, int]] = []
    for ts in triples_per_seed:
        all_triples.extend(ts)
    if not all_triples:
        return 1.0

    def _delta(triples: Sequence[Tuple[int, int, int]]) -> float:
        nc = sum(t[0] for t in triples)
        na = sum(t[1] for t in triples)
        dn = sum(t[2] for t in triples)
        return (nc - na) / dn if dn else 0.0

    observed = _delta(all_triples)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        null_c = 0
        null_a = 0
        null_n = 0
        for ec, ea, n in all_triples:
            if rng.random() < 0.5:
                null_c += ea
                null_a += ec
            else:
                null_c += ec
                null_a += ea
            null_n += n
        null_delta = (null_c - null_a) / null_n if null_n else 0.0
        if abs(null_delta) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def multiseed_audit(
    anchor_preds_path: Path,
    candidate_preds_paths: Sequence[Path],
    gold_path: Path,
    candidate_name: str,
    *,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
) -> MultiseedAuditReport:
    """Run the multi-seed audit protocol.

    candidate_preds_paths: one JSONL per seed (>= 3).
    Each JSONL has rows {id, pred}.
    """
    anchor_rows = _read_jsonl(Path(anchor_preds_path))
    gold_rows = _read_jsonl(Path(gold_path))

    anchor_preds_list = [r.get("pred", "") for r in anchor_rows]
    gold_list = [r.get("gold", r.get("paragraph", "")) for r in gold_rows]
    anchor_cer = cer_organiser(gold_list, anchor_preds_list)

    per_seed_cer: List[float] = []
    per_seed_delta: List[float] = []
    triples_per_seed: List[List[Tuple[int, int, int]]] = []
    n_pairs = 0

    for preds_path in candidate_preds_paths:
        cand_rows = _read_jsonl(Path(preds_path))
        refs, hyps_a, hyps_c = _align_two(anchor_rows, cand_rows, gold_rows)
        n_pairs = max(n_pairs, len(refs))

        cand_preds_list = [r.get("pred", "") for r in cand_rows]
        gold_aligned = [r.get("gold", r.get("paragraph", "")) for r in gold_rows
                       if r["id"] in {cr["id"] for cr in cand_rows}]
        cand_cer = cer_organiser(refs, hyps_c)
        per_seed_cer.append(cand_cer)

        delta = _delta_from_normalised(refs, hyps_a, hyps_c)
        per_seed_delta.append(delta)

        refs_n = [clitic_space_normalise(r) for r in refs]
        hyps_a_n = [clitic_space_normalise(h) for h in hyps_a]
        hyps_c_n = [clitic_space_normalise(h) for h in hyps_c]
        triples: List[Tuple[int, int, int]] = []
        for r, ha, hc in zip(refs_n, hyps_a_n, hyps_c_n):
            ea, na = _edit_count(r, ha)
            ec, nc = _edit_count(r, hc)
            if na and nc and na == nc:
                triples.append((ec, ea, na))
        triples_per_seed.append(triples)

    aggregate_delta, bca_lo, bca_hi = _bca_bootstrap(
        triples_per_seed, n_boot=n_boot, seed=seed
    )
    perm_p = _permutation_test(triples_per_seed, n_perm=n_perm, seed=seed)

    sign_consistent = all(
        d * aggregate_delta >= 0 for d in per_seed_delta
    ) if per_seed_delta else False

    if bca_lo <= 0.0 <= bca_hi:
        verdict = "INSIDE NOISE"
    elif aggregate_delta < 0.0 and bca_hi < 0.0 and perm_p < 0.05 and sign_consistent:
        verdict = "KEEP"
    else:
        verdict = "DROP"

    return MultiseedAuditReport(
        candidate_name=candidate_name,
        n_seeds=len(candidate_preds_paths),
        n_pairs=n_pairs,
        anchor_cer=anchor_cer,
        per_seed_cer=per_seed_cer,
        per_seed_delta=per_seed_delta,
        aggregate_delta=aggregate_delta,
        bca_ci=(bca_lo, bca_hi),
        permutation_p=perm_p,
        sign_consistent=sign_consistent,
        verdict=verdict,
    )
