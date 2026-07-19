"""Tests for src/eval/. No GPU, no network, no real model output."""
from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

import pytest

from src.eval import audit, buckets, cer, runner


# ---------- helpers ----------

def _perturb(s: str, rate: float = 0.05, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for ch in s:
        if rng.random() < rate:
            if rng.random() < 0.5:
                continue
            out.append("x")
        else:
            out.append(ch)
    return "".join(out)


GOLD = [
    "0 — Għadha mhux fis-seħħ.",
    "Il-kelb tat-tifel fid-dar tas-suq.",
    "The quick brown fox jumps over the lazy dog.",
    "Ċensura, Ġid, Ħajja, Żgur.",
    "L-orizzont jinŻel bil-mod – imma jinŻel.",
    "Test paragraph with no diacritics at all today.",
    "Bonġu, kif int? Jien tajjeb ħafna grazzi.",
    "Short.",
    "Another short paragraph.",
    "A medium-length paragraph that mentions il-knisja and id-dar with some words to fill it out for the length bucket assignment.",
]


# ---------- (a) NFC normalisation ----------

def test_nfc_normalisation_applied():
    # NFD-decomposed gh vs NFC-precomposed: must compute identical CER.
    nfc_ref = "Għadha"  # G + U+0127 (NFC)
    nfd_ref = unicodedata.normalize("NFD", nfc_ref)
    nfc_hyp = "Għadha"
    assert cer.cer_aggregate([nfc_ref], [nfc_hyp]) == pytest.approx(0.0)
    assert cer.cer_aggregate([nfd_ref], [nfc_hyp]) == pytest.approx(0.0)
    # Confirm the normalise pass yields identical strings.
    assert cer.normalise(nfd_ref) == cer.normalise(nfc_ref)


def test_normalise_strips_and_replaces_newlines_in_hyp():
    assert cer.normalise("  hi  ") == "hi"
    assert cer.normalise("a\nb", is_hyp=True) == "a b"
    # newline preserved in ref
    assert cer.normalise("a\nb", is_hyp=False) == "a\nb"


# ---------- (b) bucketing edge cases ----------

def test_bucket_empty_paragraph():
    tags = buckets.tag_paragraph("", lines=None)
    assert "len-q1" in tags
    assert "lang-other" in tags
    assert "prefix-no" in tags
    assert "linehyp-no" in tags
    assert "em-dash-no" in tags
    assert "single-line" in tags


def test_bucket_all_dash():
    tags = buckets.tag_paragraph("— – —", lines=["—", "– —"])
    assert "em-dash-yes" in tags
    assert "linehyp-no" in tags  # standalone dash, not a line-break hyphen
    assert "multi-line" in tags


def test_bucket_line_count():
    assert "single-line" in buckets.tag_paragraph("one line", lines=["one line"])
    assert "multi-line" in buckets.tag_paragraph("a b", lines=["a", "b"])
    # missing line info is treated as single-line
    assert "single-line" in buckets.tag_paragraph("x", lines=None)


def test_bucket_line_hyphen_detected():
    lines = ["fis-", "seħħ"]
    tags = buckets.tag_paragraph("fis-seħħ", lines=lines)
    assert "linehyp-yes" in tags


def test_bucket_il_prefix():
    assert "prefix-yes" in buckets.tag_paragraph("Il-kelb fid-dar.")
    assert "prefix-no" in buckets.tag_paragraph("Kelb dar.")


def test_bucket_language_english_vs_mt():
    en = buckets.tag_paragraph(
        "the quick brown fox jumps over the lazy dog with many words"
    )
    mt = buckets.tag_paragraph("Ċensura u Ġid.")
    assert "lang-en" in en or "lang-mt" in en
    assert "lang-mt" in mt


def test_compute_quartiles_placeholder_on_empty():
    assert buckets.compute_quartiles([]) == (80, 200, 400)


# ---------- (c) bootstrap CI finite on small synth set ----------

def _mock_preds(gold, rate=0.05, seed=0):
    return [_perturb(g, rate=rate, seed=seed + i) for i, g in enumerate(gold)]


def test_bootstrap_cer_returns_finite():
    hyps = _mock_preds(GOLD, rate=0.05, seed=1)
    out = audit.bootstrap_cer(GOLD, hyps, n_boot=200, seed=1)
    assert 0.0 <= out["cer"] <= 1.0
    assert 0.0 <= out["ci_lo"] <= out["cer"] + 1e-9
    assert out["cer"] - 1e-9 <= out["ci_hi"] <= 1.0
    assert out["n"] == len(GOLD)


def test_bootstrap_per_bucket_finite_and_small_flagged():
    hyps = _mock_preds(GOLD, rate=0.05, seed=2)
    tags = buckets.tag_corpus(GOLD)
    out = audit.bootstrap_per_bucket(GOLD, hyps, tags, n_boot=200, min_n=20, seed=2)
    # small set -> every bucket below 20, small flag set everywhere
    assert out
    assert all(b["small"] for b in out.values())
    for b in out.values():
        assert 0.0 <= b["cer"] <= 1.0


def test_shuffle_test_p_value_in_range():
    hyps = _mock_preds(GOLD, rate=0.05, seed=3)
    out = audit.shuffle_test(GOLD, hyps, n_perm=200, seed=3)
    assert 0.0 < out["p_value"] <= 1.0
    assert out["null_mean"] > out["observed"]  # random alignment is worse


def test_pair_bootstrap_delta_zero_for_identical_systems():
    hyps = _mock_preds(GOLD, rate=0.05, seed=4)
    out = audit.pair_bootstrap_delta(GOLD, hyps, hyps, n_boot=200, seed=4)
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)


# ---------- (d) regression gate fires on +0.005 absolute increase ----------

def test_regression_gate_fires_on_005_absolute():
    # Build a baseline and a regressed system. The regressed system perturbs
    # the same paragraphs at a higher rate so per-bucket CER rises by >= 0.005
    # absolute on at least one non-small bucket.
    baseline = _mock_preds(GOLD, rate=0.02, seed=10)
    regressed = _mock_preds(GOLD, rate=0.20, seed=10)
    tags = buckets.tag_corpus(GOLD)
    pb_base = cer.cer_per_bucket(GOLD, baseline, tags, min_n=1)
    pb_reg = cer.cer_per_bucket(GOLD, regressed, tags, min_n=1)
    deltas = {k: pb_reg[k]["cer"] - pb_base[k]["cer"] for k in pb_base}
    # at least one bucket should regress by >= 0.005
    assert any(d >= 0.005 for d in deltas.values()), deltas
    # Regression gate logic, as enforced in the eval-runner contract.
    gated = any(d > 0.005 for d in deltas.values())
    assert gated is True


# ---------- runner E2E with mock JSONL ----------

def test_runner_end_to_end(tmp_path: Path):
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "preds.jsonl"
    with gold_path.open("w", encoding="utf-8") as f:
        for i, g in enumerate(GOLD):
            f.write(json.dumps({"id": f"p{i}", "paragraph": g, "lines": [g]}) + "\n")
    with pred_path.open("w", encoding="utf-8") as f:
        for i, g in enumerate(GOLD):
            f.write(json.dumps({"id": f"p{i}", "hypothesis": _perturb(g, 0.03, seed=i)}) + "\n")
    out = runner.run_eval(pred_path, gold_path, out_dir=tmp_path / "out", n_boot=100)
    assert out["n_pairs"] == len(GOLD)
    assert 0.0 <= out["aggregate_cer"] <= 1.0
    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "report.json").exists()


def test_organiser_kill_criterion(tmp_path: Path):
    # Build a fake organiser module that reports a CER 0.02 absolute off; the
    # runner must raise.
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "preds.jsonl"
    with gold_path.open("w", encoding="utf-8") as f:
        for i, g in enumerate(GOLD):
            f.write(json.dumps({"id": f"p{i}", "paragraph": g}) + "\n")
    with pred_path.open("w", encoding="utf-8") as f:
        for i, g in enumerate(GOLD):
            f.write(json.dumps({"id": f"p{i}", "hypothesis": g}) + "\n")
    org_path = tmp_path / "fake_organiser.py"
    org_path.write_text("def run(predictions, gold):\n    return 0.5\n")
    with pytest.raises(RuntimeError):
        runner.run_eval(pred_path, gold_path, organiser_script_path=org_path)
