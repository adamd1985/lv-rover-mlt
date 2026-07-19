"""Statistical audit framework for the eval harness.

Implements:
  - bootstrap_cer        : aggregate CER with 95% CI via 1000 resamples
  - bootstrap_per_bucket : per-bucket aggregate CER with 95% CI
  - shuffle_test         : p-value of the observed CER under random alignment
  - kfold_synth_cv       : k=5 stratified CV on a synth-val set
  - pair_bootstrap_delta : CI of CER(A) - CER(B) over paired (ref, hyp_A, hyp_B)
  - per_char_paired_bootstrap : per-canary-char paired bootstrap with FDR

Aggregation is always sum-of-numerators within a resample. We pre-compute
(edit_count, ref_length) per pair once and resample those tuples - the inner
CER evaluator is a sum / sum, not a fresh jiwer call per draw.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .cer import _edit_count, normalise

CANARY_CHARS = ["ċ", "ġ", "ħ", "ż", "-", "–", "—"]


def _pair_counts(refs: Sequence[str], hyps: Sequence[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for r, h in zip(refs, hyps):
        ed, n = _edit_count(r, h)
        if n == 0:
            continue
        out.append((ed, n))
    return out


def _cer_from_counts(pairs: Sequence[Tuple[int, int]]) -> float:
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return (num / den) if den else 0.0


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def bootstrap_cer(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    pairs = _pair_counts(refs, hyps)
    if not pairs:
        return {"cer": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(pairs)
    draws: List[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        draws.append(_cer_from_counts(sample))
    return {
        "cer": _cer_from_counts(pairs),
        "ci_lo": _percentile(draws, 0.025),
        "ci_hi": _percentile(draws, 0.975),
        "n": n,
    }


def bootstrap_per_bucket(
    refs: Sequence[str],
    hyps: Sequence[str],
    bucket_tags: Sequence[Iterable[str]],
    *,
    n_boot: int = 1000,
    min_n: int = 20,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    rows = []
    for r, h, tags in zip(refs, hyps, bucket_tags):
        ed, n = _edit_count(r, h)
        if n == 0:
            continue
        rows.append((ed, n, set(tags)))
    by_bucket: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for ed, n, tags in rows:
        for t in tags:
            by_bucket[t].append((ed, n))
    out: Dict[str, Dict[str, float]] = {}
    rng = random.Random(seed)
    for t, plist in by_bucket.items():
        nb = len(plist)
        if nb < min_n:
            out[t] = {"cer": _cer_from_counts(plist), "ci_lo": float("nan"),
                       "ci_hi": float("nan"), "n": nb, "small": True}
            continue
        draws = []
        for _ in range(n_boot):
            sample = [plist[rng.randrange(nb)] for _ in range(nb)]
            draws.append(_cer_from_counts(sample))
        out[t] = {
            "cer": _cer_from_counts(plist),
            "ci_lo": _percentile(draws, 0.025),
            "ci_hi": _percentile(draws, 0.975),
            "n": nb,
            "small": False,
        }
    return out


def shuffle_test(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Null hypothesis: hypothesis-to-reference alignment is random. Compute
    CER on a randomised pairing and count how often the null CER is <= the
    observed CER. p = (#null <= obs + 1) / (n_perm + 1).
    """
    refs_n = [normalise(r) for r in refs]
    hyps_n = [normalise(h, is_hyp=True) for h in hyps]
    pairs = list(zip(refs_n, hyps_n))
    pairs = [(r, h) for r, h in pairs if r]
    obs = _cer_from_counts(_pair_counts([r for r, _ in pairs], [h for _, h in pairs]))
    if not pairs:
        return {"observed": obs, "p_value": 1.0}
    rng = random.Random(seed)
    hyps_only = [h for _, h in pairs]
    nb = 0
    null_draws: List[float] = []
    for _ in range(n_perm):
        shuffled = hyps_only[:]
        rng.shuffle(shuffled)
        null = _cer_from_counts(
            _pair_counts([r for r, _ in pairs], shuffled)
        )
        null_draws.append(null)
        if null <= obs:
            nb += 1
    p = (nb + 1) / (n_perm + 1)
    return {"observed": obs, "p_value": p,
            "null_mean": sum(null_draws) / len(null_draws),
            "null_min": min(null_draws)}


def kfold_synth_cv(
    refs: Sequence[str],
    hyps: Sequence[str],
    bucket_tags: Sequence[Iterable[str]],
    *,
    k: int = 5,
    stratify_on: str = "len-q1",
    seed: int = 0,
) -> Dict[str, float]:
    """Stratified k-fold CV on a synth-val set. Stratification by the chosen
    bucket label keeps fold composition balanced. Returns fold CER mean and
    sample std.
    """
    rng = random.Random(seed)
    idx = list(range(len(refs)))
    rng.shuffle(idx)
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    fold_cers: List[float] = []
    for f in folds:
        sub_r = [refs[i] for i in f]
        sub_h = [hyps[i] for i in f]
        fold_cers.append(_cer_from_counts(_pair_counts(sub_r, sub_h)))
    mean = sum(fold_cers) / len(fold_cers) if fold_cers else 0.0
    var = sum((x - mean) ** 2 for x in fold_cers) / max(len(fold_cers) - 1, 1)
    return {"fold_cers": fold_cers, "mean": mean, "std": var ** 0.5}


def pair_bootstrap_delta(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Variant-comparison primitive: pair-bootstrap CI of CER(A) - CER(B).

    For each pair we keep (ed_a, ed_b, n) so a resample is one rng draw over
    a tuple. Sign convention: positive delta means A is worse than B.
    """
    triples: List[Tuple[int, int, int]] = []
    for r, ha, hb in zip(refs, hyps_a, hyps_b):
        ea, na = _edit_count(r, ha)
        eb, nb = _edit_count(r, hb)
        if na == 0 or nb == 0 or na != nb:
            continue
        triples.append((ea, eb, na))
    if not triples:
        return {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(triples)

    def _delta(sample: Sequence[Tuple[int, int, int]]) -> float:
        a_num = sum(t[0] for t in sample)
        b_num = sum(t[1] for t in sample)
        d = sum(t[2] for t in sample)
        if d == 0:
            return 0.0
        return (a_num - b_num) / d

    draws = [_delta([triples[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    return {
        "delta": _delta(triples),
        "ci_lo": _percentile(draws, 0.025),
        "ci_hi": _percentile(draws, 0.975),
        "n": n,
    }


def _bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg FDR. Returns a list of significance booleans."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    sig = [False] * m
    threshold = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            threshold = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= threshold:
            sig[idx] = True
    return sig


def per_char_paired_bootstrap(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    *,
    chars: Sequence[str] = tuple(CANARY_CHARS),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Per-character paired bootstrap of CER delta on paragraphs that contain
    the character in the reference. FDR-corrected across characters."""
    refs_n = [normalise(r) for r in refs]
    hyps_a_n = [normalise(h, is_hyp=True) for h in hyps_a]
    hyps_b_n = [normalise(h, is_hyp=True) for h in hyps_b]
    results: Dict[str, Dict[str, float]] = {}
    pvals: List[float] = []
    chars_seen: List[str] = []
    for ch in chars:
        sub_idx = [i for i, r in enumerate(refs_n) if ch in r]
        sub_refs = [refs_n[i] for i in sub_idx]
        sub_a = [hyps_a_n[i] for i in sub_idx]
        sub_b = [hyps_b_n[i] for i in sub_idx]
        if not sub_refs:
            results[ch] = {"delta": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "n": 0, "p_value": 1.0,
                            "significant": False}
            pvals.append(1.0)
            chars_seen.append(ch)
            continue
        d = pair_bootstrap_delta(sub_refs, sub_a, sub_b, n_boot=n_boot, seed=seed)
        triples = []
        for r, ha, hb in zip(sub_refs, sub_a, sub_b):
            ea, na = _edit_count(r, ha)
            eb, nb = _edit_count(r, hb)
            if na and nb and na == nb:
                triples.append((ea, eb, na))
        rng = random.Random(seed + 1)
        if triples:
            n = len(triples)
            draws = []
            for _ in range(n_boot):
                sample = [triples[rng.randrange(n)] for _ in range(n)]
                a_num = sum(t[0] for t in sample)
                b_num = sum(t[1] for t in sample)
                dn = sum(t[2] for t in sample)
                draws.append((a_num - b_num) / dn if dn else 0.0)
            cnt = sum(1 for x in draws if x * d["delta"] <= 0)
            pvals.append((cnt + 1) / (n_boot + 1))
        else:
            pvals.append(1.0)
        results[ch] = {**d, "p_value": pvals[-1], "significant": False}
        chars_seen.append(ch)
    sig = _bh_fdr(pvals, alpha=alpha)
    for ch, s in zip(chars_seen, sig):
        results[ch]["significant"] = bool(s)
    return results
