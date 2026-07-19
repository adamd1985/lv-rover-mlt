"""Cross-shard distribution analysis for shard_0001 / shard_0002_korpus / shard_0003.

Computes per-shard distributions and pairwise divergences (baseline = shard_0001).
Writes outputs/analyses/cross_shard_distribution.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.buckets import (  # noqa: E402
    bucket_il_prefix,
    bucket_language,
    bucket_length,
    bucket_line_hyphen,
    bucket_em_dash,
    compute_quartiles,
)

import numpy as np  # noqa: E402
from scipy.stats import chisquare, ks_2samp  # noqa: E402

SHARDS = ["shard_0001", "shard_0002_korpus", "shard_0003"]
CANARY = list("ĊċĠġĦħŻż")
GRAVES = list("àèìòù")
EN_DASH = "–"
EM_DASH = "—"


def load_shard_manifests(shard_dir: Path) -> Iterator[dict]:
    """Yield per-sample dicts lazily. Drops corrupt JSONs silently (count is the
    survivor count). `hyphen_kinds` is included only when present."""
    for jpath in sorted(shard_dir.glob("*.json")):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        label = raw.get("label", "") or ""
        label_parts = raw.get("label_parts") or []
        out = {
            "id": jpath.stem,
            "label": label,
            "label_parts": label_parts,
            "lang_meta": raw.get("lang"),
        }
        if "hyphen_kinds" in raw:
            out["hyphen_kinds"] = raw["hyphen_kinds"]
        yield out


def per_shard_stats(shard_dir: Path) -> dict:
    samples = list(load_shard_manifests(shard_dir))
    n = len(samples)
    refs = [s["label"] for s in samples]
    quartiles = compute_quartiles(refs)

    bucket_counts: dict[str, Counter] = {
        "length": Counter(),
        "language": Counter(),
        "il_prefix": Counter(),
        "line_hyphen": Counter(),
        "unicode_dash": Counter(),
    }
    char_counter: Counter = Counter()
    canary_total = 0
    grave_total = 0
    en_dash_total = 0
    em_dash_total = 0
    char_lens: list[int] = []
    line_lens: list[int] = []
    hyphen_kind_counter: Counter | None = None

    for s in samples:
        ref = s["label"]
        parts = s["label_parts"] or [ref]
        bucket_counts["length"][bucket_length(ref, quartiles)] += 1
        bucket_counts["language"][bucket_language(ref)] += 1
        bucket_counts["il_prefix"][bucket_il_prefix(ref)] += 1
        bucket_counts["line_hyphen"][bucket_line_hyphen(parts)] += 1
        bucket_counts["em_dash"][bucket_em_dash(ref)] += 1
        char_counter.update(ref)
        canary_total += sum(1 for c in ref if c in CANARY)
        grave_total += sum(1 for c in ref if c in GRAVES)
        en_dash_total += ref.count(EN_DASH)
        em_dash_total += ref.count(EM_DASH)
        char_lens.append(len(ref))
        line_lens.append(len(parts))
        if "hyphen_kinds" in s:
            if hyphen_kind_counter is None:
                hyphen_kind_counter = Counter()
            for hk in s["hyphen_kinds"]:
                hyphen_kind_counter[hk] += 1

    top30 = char_counter.most_common(30)

    def length_stats(xs: list[int]) -> dict:
        if not xs:
            return {"min": 0, "median": 0, "mean": 0.0, "p95": 0, "max": 0}
        arr = np.array(xs)
        return {
            "min": int(arr.min()),
            "median": float(median(arr)),
            "mean": float(mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
        }

    return {
        "shard": shard_dir.name,
        "sample_count": n,
        "quartiles_chars": list(quartiles),
        "bucket_counts": {k: dict(v) for k, v in bucket_counts.items()},
        "char_inventory": {
            "top30": [[c, n_] for c, n_ in top30],
            "canary_total": canary_total,
            "grave_total": grave_total,
            "en_dash_total": en_dash_total,
            "em_dash_total": em_dash_total,
            "vocab_size": len(char_counter),
            "total_chars": sum(char_counter.values()),
        },
        "char_length_stats": length_stats(char_lens),
        "line_length_stats": length_stats(line_lens),
        "hyphen_kind_distribution": (
            dict(hyphen_kind_counter) if hyphen_kind_counter is not None else "schema_field_absent"
        ),
        # carried forward for divergence calcs
        "_internal": {
            "char_counter": dict(char_counter),
            "char_lens": char_lens,
        },
    }


def js_divergence(p_counts: dict, q_counts: dict) -> float:
    keys = sorted(set(p_counts) | set(q_counts))
    p_total = sum(p_counts.values()) or 1
    q_total = sum(q_counts.values()) or 1
    p = np.array([p_counts.get(k, 0) / p_total for k in keys])
    q = np.array([q_counts.get(k, 0) / q_total for k in keys])
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def chi_squared_bucket(p_counts: dict, q_counts: dict) -> dict:
    """Chi-square goodness of fit: q observed against p expected proportions."""
    keys = sorted(set(p_counts) | set(q_counts))
    p_total = sum(p_counts.values()) or 1
    q_total = sum(q_counts.values()) or 1
    expected = np.array([max(p_counts.get(k, 0) / p_total * q_total, 1e-9) for k in keys])
    observed = np.array([q_counts.get(k, 0) for k in keys], dtype=float)
    # rescale expected to match observed sum (chisquare requirement)
    expected *= observed.sum() / expected.sum()
    try:
        chi, p = chisquare(f_obs=observed, f_exp=expected)
        return {"chi2": float(chi), "p_value": float(p), "categories": keys}
    except ValueError as e:
        return {"error": str(e), "categories": keys}


def pairwise_divergences(stats_by_shard: dict) -> dict:
    base = "shard_0001"
    base_stats = stats_by_shard[base]
    out: dict = {}
    for shard in SHARDS:
        if shard == base:
            continue
        cur = stats_by_shard[shard]
        # bucket chi-squared per dimension
        bucket_chi = {}
        for dim in base_stats["bucket_counts"]:
            bucket_chi[dim] = chi_squared_bucket(
                base_stats["bucket_counts"][dim], cur["bucket_counts"][dim]
            )
        # JS divergence on char unigram
        js = js_divergence(
            base_stats["_internal"]["char_counter"], cur["_internal"]["char_counter"]
        )
        # KS on char-length distributions
        ks = ks_2samp(base_stats["_internal"]["char_lens"], cur["_internal"]["char_lens"])
        out[f"{base}_vs_{shard}"] = {
            "bucket_chi_squared": bucket_chi,
            "char_unigram_js_divergence_bits": js,
            "char_length_ks": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
        }
    return out


def main() -> None:
    stats_by_shard: dict = {}
    for shard in SHARDS:
        shard_dir = REPO / "data" / "synth" / shard
        print(f"[scan] {shard} ...", flush=True)
        stats_by_shard[shard] = per_shard_stats(shard_dir)
        print(f"  -> {stats_by_shard[shard]['sample_count']} samples", flush=True)

    divergences = pairwise_divergences(stats_by_shard)

    # strip internal before writing
    clean = {}
    for shard, s in stats_by_shard.items():
        s.pop("_internal", None)
        clean[shard] = s

    out_path = REPO / "outputs" / "analyses" / "cross_shard_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"shards": clean, "pairwise_divergences": divergences},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
