"""Paired bootstrap CI plus permutation test on the combined post-processing
delta (recognition-only CER 0.01317 -> full-pipeline CER 0.00700), as one
block.

The paper's audit harness (Section 5.4) bootstraps the 3-to-5-stream
ensemble-expansion decision but never the combined post-processing delta -
this closes that gap, reusing the exact recognition/replay logic from
scripts/audit_postproc_overfit.py rather than reimplementing it.

Method, per the project's own audit-gate standard (bootstrap CI plus
permutation p < 0.05, sign-consistency required for a KEEP verdict):

1. 1,000 paired bootstrap resamples over the 422 dev paragraphs. For each
   resample, recompute aggregate CER (sum edit distance / sum ref length,
   the paper's own formula, Section 5.3) for both conditions, take the delta.
   Report the 95% CI of that delta distribution.
2. A paired permutation test: for each of 10,000 permutations, independently
   and randomly swap the (recognition-only, full-pipeline) label on each
   paragraph's pair with probability 0.5, recompute the aggregate delta under
   that relabeling. The p-value is the fraction of permuted deltas whose
   magnitude meets or exceeds the observed delta's magnitude (two-sided).
   This tests the null that post-processing has no systematic effect, i.e.
   that the observed delta arises from an arbitrary, no-effect labeling of
   the same 844 (422 x 2) CER contributions.

Usage:
    PYTHONPATH=. python scripts/audit_bootstrap_full_chain.py
    PYTHONPATH=. python scripts/audit_bootstrap_full_chain.py --limit 20  # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import jiwer
import PIL.Image
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import competition_transcriber as ct
from scripts.audit_postproc_overfit import (
    _base_select,
    _load_corrector_router,
    load_dev,
    replay,
    streams_for,
)

RANDOM_SEED = 42
N_RESAMPLES = 1000
N_PERMUTATIONS = 10000


def _edit_distance(a: str, b: str) -> int:
    """RAW jiwer character edit distance on unnormalised strings - the same
    backend and the same no-normalisation contract as the organiser's scorer
    (competition_evaluator.py) and scripts/audit_postproc_overfit.py.

    A prior version of this function reimplemented Levenshtein by hand
    instead of importing jiwer. That second backend silently drifted the
    audit's point estimate (0.00503) away from the frozen headline delta
    (0.00617) computed via jiwer. Every
    script that recomputes CER for a claim reported in the paper must call
    jiwer (or a thin wrapper around it) and nothing else; do not add a third
    edit-distance implementation to this repo.
    """
    out = jiwer.process_characters([a], [b])
    return int(out.substitutions + out.deletions + out.insertions)


def recognition_only(streams: dict, corrector, router) -> str:
    """Same as replay()'s vote step, stopping before any v16-v20 rule."""
    present = [s for s in (streams["mlt"], streams["ita"], streams["romance"],
                           streams["stock"], streams["up"]) if s]
    if not present:
        return ""
    base = _base_select(streams)
    joined = base
    if corrector is not None and len(joined) >= ct._CORRECTOR_LEN_THR:
        joined = corrector.correct(joined)
    candidates = []
    for cand in (streams["mlt"], streams["ita"], streams["romance"],
                 streams["stock"], streams["up"]):
        if cand and cand != base and cand not in candidates:
            candidates.append(cand)
    if candidates:
        joined = router.combine_lv(joined, candidates)
    return joined


def aggregate_cer(pairs: list) -> float:
    num = sum(p[0] for p in pairs)
    den = sum(p[1] for p in pairs)
    return num / den if den else 0.0


def permutation_test(per_para: list, observed_delta: float, rng: random.Random) -> tuple:
    """Paired sign-flip permutation test on the recognition-only vs full-pipeline delta.

    Each paragraph independently keeps or swaps its (recog, full) assignment
    with probability 0.5, under the null that post-processing has no
    systematic per-paragraph effect. Returns (p_value, critical_value) where
    critical_value is the 95th-percentile magnitude of the null distribution -
    the two-sided rejection threshold at alpha=0.05.
    """
    null_deltas = []
    for _ in range(N_PERMUTATIONS):
        num_recog = num_full = 0
        den = 0
        for ed_recog, len_recog, ed_full, len_full in per_para:
            den += len_recog
            if rng.random() < 0.5:
                num_recog += ed_recog
                num_full += ed_full
            else:
                num_recog += ed_full
                num_full += ed_recog
        c_recog = num_recog / den if den else 0.0
        c_full = num_full / den if den else 0.0
        null_deltas.append(c_recog - c_full)

    null_deltas.sort()
    abs_null = sorted(abs(d) for d in null_deltas)
    abs_observed = abs(observed_delta)
    n_ge = sum(1 for d in abs_null if d >= abs_observed)
    p_value = (n_ge + 1) / (N_PERMUTATIONS + 1)
    critical_value = abs_null[int(0.95 * N_PERMUTATIONS)]
    return p_value, critical_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corrector, router = _load_corrector_router()
    from malti.line_joiner import RBLineJoiner
    joiner = RBLineJoiner()

    items = load_dev(args.limit)
    print(f"scoring {len(items)} real dev paragraphs (5 Tesseract streams each)...")

    per_para = []  # (ed_recog, len_recog, ed_full, len_full) per paragraph
    t0 = time.time()
    for i, it in enumerate(items):
        img = PIL.Image.open(it["image"]).convert("RGB")
        st = streams_for(joiner, img)
        gold = it["gold"]
        hyp_recog = recognition_only(st, corrector, router)
        hyp_full = replay(st, corrector, router, off=None)
        per_para.append((
            _edit_distance(gold, hyp_recog), len(gold),
            _edit_distance(gold, hyp_full), len(gold),
        ))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(items)} scored, {time.time()-t0:.0f}s elapsed")

    cer_recog = aggregate_cer([(p[0], p[1]) for p in per_para])
    cer_full = aggregate_cer([(p[2], p[3]) for p in per_para])
    print(f"point estimate: recognition-only CER={cer_recog:.5f}, full-pipeline CER={cer_full:.5f}")
    print(f"point delta: {cer_recog - cer_full:.5f}")

    rng = random.Random(RANDOM_SEED)
    n = len(per_para)
    deltas = []
    for _ in range(N_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        resample = [per_para[j] for j in idx]
        c_recog = aggregate_cer([(p[0], p[1]) for p in resample])
        c_full = aggregate_cer([(p[2], p[3]) for p in resample])
        deltas.append(c_recog - c_full)

    deltas.sort()
    lo = deltas[int(0.025 * N_RESAMPLES)]
    hi = deltas[int(0.975 * N_RESAMPLES)]
    excludes_zero = lo > 0 or hi < 0
    print(f"95% CI of delta (recognition-only minus full-pipeline CER): [{lo:.5f}, {hi:.5f}]")
    print(f"excludes zero: {excludes_zero}")

    perm_rng = random.Random(RANDOM_SEED + 1)
    p_value, critical_value = permutation_test(per_para, cer_recog - cer_full, perm_rng)
    significant = p_value < 0.05
    print(f"permutation test ({N_PERMUTATIONS} perms): p={p_value:.5f}, "
          f"critical value (alpha=0.05, two-sided)={critical_value:.5f}")
    print(f"significant at alpha=0.05: {significant}")

    with open("outputs/campaign/bootstrap_full_chain.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_paragraphs": n,
            "cer_recognition_only": cer_recog,
            "cer_full_pipeline": cer_full,
            "point_delta": cer_recog - cer_full,
            "ci_95_lo": lo,
            "ci_95_hi": hi,
            "excludes_zero": excludes_zero,
            "n_resamples": N_RESAMPLES,
            "permutation_p_value": p_value,
            "permutation_critical_value_alpha05": critical_value,
            "permutation_n": N_PERMUTATIONS,
            "significant_alpha05": significant,
            "seed": RANDOM_SEED,
        }, f, indent=2)
    print("wrote outputs/campaign/bootstrap_full_chain.json")


if __name__ == "__main__":
    main()
