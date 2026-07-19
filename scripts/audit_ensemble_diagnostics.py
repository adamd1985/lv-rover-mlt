"""Peer-review diagnostics: Task 1 (attribution CI) + Task 2 (stream independence)
for the LV-ROVER-MLT 5-stream Maltese OCR ensemble.

All inputs are cached per-paragraph artifacts on the 422-image dev set:
  anchor   outputs/tesseract/audit/finetune_50k_preds.jsonl   (fine-tuned Tesseract, reported CER 0.01605)
  gold     outputs/tesseract/audit/gold.jsonl
  stream mlt      outputs/tess_lang_chain/mlt.jsonl
  stream ita      outputs/tess_lang_chain/mlt_ita.jsonl        (mlt+ita)
  stream romance  outputs/tess_lang_chain/mlt_ita_fra.jsonl    (mlt+ita+fra)
  stream stock    outputs/audit_v12/dev_mltstock.jsonl
  stream up       outputs/campaign/up_stream_dev.jsonl         (2x-upscale mlt+ita, regenerated)

TASK 1: paired bootstrap of the per-paragraph CER delta between the anchor and
the 5-stream recognition-only ensemble, seed 42, 1000-resample percentile CI +
10000-permutation test (bootstrap_stats.py, the project's own audit helpers).
The recognition-only ensemble is reconstructed by the exact executable
recognition_only() path from scripts/audit_bootstrap_full_chain.py (base select
-> confusion corrector -> LV-ROVER vote), stopping before any v16-v20 rule.

NOTE (disclosed, the claim audit): the executable recognition_only()
reproduction yields ~0.01218, not the paper headline 0.01317 (the v15 wrapper
output under jiwer). We report the executable number and label it; we do not
fabricate the 0.01317 pairing.

TASK 2: from the 5 individual per-paragraph streams,
  (a) mean pairwise word-disagreement rate over aligned positions (10 pairs),
  (b) oracle-vote CER = per-paragraph pick the single stream with the lowest
      edit distance to gold (a lower bound on any voting scheme),
  compared to the actual 5-stream recognition-only CER from Task 1.

Usage:
    PYTHONPATH=. python scripts/audit_ensemble_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bootstrap_stats import (
    aggregate_cer, bootstrap_ci, edit_distance, permutation_test,
)
from scripts.audit_bootstrap_full_chain import recognition_only
from scripts.audit_postproc_overfit import _load_corrector_router
import competition_transcriber as ct

SEED = 42
GOLD = ROOT / "outputs/tesseract/audit/gold.jsonl"
ANCHOR = ROOT / "outputs/tesseract/audit/finetune_50k_preds.jsonl"
STREAM_PATHS = {
    "mlt": ROOT / "outputs/tess_lang_chain/mlt.jsonl",
    "ita": ROOT / "outputs/tess_lang_chain/mlt_ita.jsonl",
    "romance": ROOT / "outputs/tess_lang_chain/mlt_ita_fra.jsonl",
    "stock": ROOT / "outputs/audit_v12/dev_mltstock.jsonl",
    "up": ROOT / "outputs/campaign/up_stream_dev.jsonl",
}
REPORT = ROOT / "outputs/campaign/ensemble_diagnostics.json"


def load(path: Path, key: str) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r.get(key, "")
    return out


def words(s: str) -> list:
    return [w for w in ct._WS_SPLIT.split(s.replace("\n", " ").strip()) if w]


def pairwise_disagreement(sa: str, sb: str) -> tuple:
    """Return (n_disagree, n_aligned) over aligned word positions of two streams."""
    aw, bw = words(sa), words(sb)
    al = ct._align_word_seqs(aw, bw)
    n_dis = n_al = 0
    for a, b in al:
        if a is None or b is None:
            continue
        n_al += 1
        if a != b:
            n_dis += 1
    return n_dis, n_al


def main() -> None:
    gold = load(GOLD, "gold")
    anchor = load(ANCHOR, "pred")
    streams = {name: load(p, "pred") for name, p in STREAM_PATHS.items()}

    ids = sorted(set(gold) & set(anchor) & set.intersection(*[set(s) for s in streams.values()]))
    n = len(ids)
    print(f"[diag] {n} paragraphs common across gold+anchor+5 streams", flush=True)
    for name, s in streams.items():
        missing = len(gold) - len(s)
        print(f"[diag] stream {name}: {len(s)} rows", flush=True)

    corrector, router = _load_corrector_router()

    # ----- Task 1: anchor vs 5-stream recognition-only -----
    per_item = []          # (ed_anchor, len_gold, ed_ensemble, len_gold) for bootstrap
    ensemble_pairs = []    # (ed_ensemble, len_gold)
    anchor_pairs = []      # (ed_anchor, len_gold)
    stream_ed = {name: [] for name in streams}   # per-stream (ed, len_gold) for standalone CER
    oracle_pairs = []      # (ed_best, len_gold)
    per_para_streams = []  # for pairwise disagreement

    for i in ids:
        g = gold[i]
        lg = len(g)
        sdict = {name: streams[name][i] for name in streams}
        hyp_ens = recognition_only(sdict, corrector, router)
        ed_ens = edit_distance(g, hyp_ens)
        ed_anc = edit_distance(g, anchor[i])
        per_item.append((ed_anc, lg, ed_ens, lg))
        anchor_pairs.append((ed_anc, lg))
        ensemble_pairs.append((ed_ens, lg))
        # per-stream standalone + oracle
        best_ed = None
        for name in streams:
            ed_s = edit_distance(g, sdict[name])
            stream_ed[name].append((ed_s, lg))
            if best_ed is None or ed_s < best_ed:
                best_ed = ed_s
        oracle_pairs.append((best_ed, lg))
        per_para_streams.append(sdict)

    cer_anchor = aggregate_cer(anchor_pairs)
    cer_ens = aggregate_cer(ensemble_pairs)
    cer_a, cer_b, point_delta, lo, hi = bootstrap_ci(per_item, SEED)
    p_value, crit = permutation_test(per_item, point_delta, SEED + 1)

    print(f"\n=== TASK 1: anchor vs 5-stream recognition-only ===", flush=True)
    print(f"CER_a (anchor)             = {cer_anchor:.5f}", flush=True)
    print(f"CER_b (ensemble recog-only)= {cer_ens:.5f}", flush=True)
    print(f"delta (anchor - ensemble)  = {point_delta:+.5f}", flush=True)
    print(f"95% CI of delta            = [{lo:.5f}, {hi:.5f}]  excludes_zero={lo > 0 or hi < 0}", flush=True)
    print(f"permutation p (10000)      = {p_value:.5f}  crit={crit:.5f}", flush=True)

    # ----- Task 2: stream independence -----
    # (a) mean pairwise disagreement over the 10 stream pairs
    names = list(streams.keys())
    pair_rates = {}
    for a, b in combinations(names, 2):
        tot_dis = tot_al = 0
        for sd in per_para_streams:
            d, al = pairwise_disagreement(sd[a], sd[b])
            tot_dis += d
            tot_al += al
        pair_rates[f"{a}|{b}"] = tot_dis / tot_al if tot_al else 0.0
    mean_pair_rate = sum(pair_rates.values()) / len(pair_rates)

    # (b) oracle-vote CER + per-stream standalone
    cer_oracle = aggregate_cer(oracle_pairs)
    standalone = {name: aggregate_cer(stream_ed[name]) for name in streams}

    print(f"\n=== TASK 2: stream independence ===", flush=True)
    print("per-stream standalone CER:", flush=True)
    for name in names:
        print(f"  {name:8s} {standalone[name]:.5f}", flush=True)
    print("pairwise word-disagreement rate:", flush=True)
    for k, v in sorted(pair_rates.items()):
        print(f"  {k:20s} {v:.4f}", flush=True)
    print(f"mean pairwise disagreement = {mean_pair_rate:.4f}", flush=True)
    print(f"oracle-vote CER (best stream/para) = {cer_oracle:.5f}", flush=True)
    print(f"actual 5-stream recog-only CER     = {cer_ens:.5f}", flush=True)
    print(f"actual - oracle                    = {cer_ens - cer_oracle:+.5f}", flush=True)
    best_single = min(standalone, key=standalone.get)
    print(f"best single stream = {best_single} @ {standalone[best_single]:.5f}", flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_paragraphs": n,
        "task1": {
            "cer_anchor": cer_anchor,
            "cer_ensemble_recognition_only": cer_ens,
            "delta_anchor_minus_ensemble": point_delta,
            "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": lo > 0 or hi < 0,
            "permutation_p": p_value,
            "permutation_critical": crit,
            "seed": SEED,
            "note": "ensemble via executable recognition_only(); reproduces ~0.01218 "
                    "(disclosed the claim audit), not the paper headline 0.01317",
        },
        "task2": {
            "standalone_cer": standalone,
            "best_single_stream": best_single,
            "best_single_cer": standalone[best_single],
            "pairwise_disagreement": pair_rates,
            "mean_pairwise_disagreement": mean_pair_rate,
            "oracle_vote_cer": cer_oracle,
            "actual_ensemble_cer": cer_ens,
            "actual_minus_oracle": cer_ens - cer_oracle,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[diag] wrote -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
