"""Shared paired bootstrap CI + permutation test helpers.

Extracted from scripts/audit_bootstrap_full_chain.py so the Hungarian and
Luxembourgish cross-language experiments (entries 224/225) can run the same
statistical audit the Maltese post-processing delta got, per the project's
audit-gate standard. `bootstrap_ci` below is a plain
percentile bootstrap - it sorts resampled deltas and reads off the 2.5th/97.5th
percentiles with no bias-correction or acceleration step, so it is not a BCa
interval; do not describe it as one in comments, docs, or the paper.
"""
from __future__ import annotations

import random

import jiwer

N_RESAMPLES = 1000
N_PERMUTATIONS = 10000


def edit_distance(a: str, b: str) -> int:
    """RAW jiwer character edit distance - the same backend every script that
    reports a CER number in the paper must use (competition_evaluator.py,
    audit_postproc_overfit.py). This used to be a hand-rolled Levenshtein
    that silently drifted from the organiser's numbers (it fed the
    anchor-to-ensemble CI, the oracle diagnostics, and both cross-language
    audits through a second, uncalibrated backend). Do not reimplement
    Levenshtein in this file or any caller; import jiwer.
    """
    out = jiwer.process_characters([a], [b])
    return int(out.substitutions + out.deletions + out.insertions)


def aggregate_cer(pairs: list) -> float:
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return num / den if den else 0.0


def bootstrap_ci(per_item: list, seed: int) -> tuple:
    """per_item: list of (ed_a, len_a, ed_b, len_b). Returns (delta, lo, hi)."""
    rng = random.Random(seed)
    n = len(per_item)
    cer_a = aggregate_cer([(p[0], p[1]) for p in per_item])
    cer_b = aggregate_cer([(p[2], p[3]) for p in per_item])
    point_delta = cer_a - cer_b

    deltas = []
    for _ in range(N_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        resample = [per_item[j] for j in idx]
        c_a = aggregate_cer([(p[0], p[1]) for p in resample])
        c_b = aggregate_cer([(p[2], p[3]) for p in resample])
        deltas.append(c_a - c_b)
    deltas.sort()
    lo = deltas[int(0.025 * N_RESAMPLES)]
    hi = deltas[int(0.975 * N_RESAMPLES)]
    return cer_a, cer_b, point_delta, lo, hi


def permutation_test(per_item: list, observed_delta: float, seed: int) -> tuple:
    rng = random.Random(seed)
    null_deltas = []
    for _ in range(N_PERMUTATIONS):
        num_a = num_b = 0
        den = 0
        for ed_a, len_a, ed_b, len_b in per_item:
            den += len_a
            if rng.random() < 0.5:
                num_a += ed_a
                num_b += ed_b
            else:
                num_a += ed_b
                num_b += ed_a
        c_a = num_a / den if den else 0.0
        c_b = num_b / den if den else 0.0
        null_deltas.append(c_a - c_b)

    abs_null = sorted(abs(d) for d in null_deltas)
    abs_observed = abs(observed_delta)
    n_ge = sum(1 for d in abs_null if d >= abs_observed)
    # Add-one (Laplace) correction: the true null could exceed the observed
    # delta on a draw this simulation never sampled, so raw n_ge/N can read
    # p=0 when it should read a finite-resolution bound. Never report p=0.
    p_value = (n_ge + 1) / (N_PERMUTATIONS + 1)
    critical_value = abs_null[int(0.95 * N_PERMUTATIONS)]
    return p_value, critical_value
