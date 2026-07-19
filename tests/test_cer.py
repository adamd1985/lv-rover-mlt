"""Tests for the dual-CER reporting path.

Covers the clitic-space normaliser and `compute_cer_dual`. The raw CER path
is exercised by tests/test_eval.py; this module focuses on the normalised
side and its invariants.
"""
from __future__ import annotations

import pytest

from src.eval import cer


def test_clitic_space_normalise_collapses_common_articles():
    assert cer.clitic_space_normalise("il- foo") == "il-foo"
    assert cer.clitic_space_normalise("tal- Kummissjoni") == "tal-Kummissjoni"
    assert cer.clitic_space_normalise("għall- finijiet") == "għall-finijiet"
    assert cer.clitic_space_normalise("l- artiklu") == "l-artiklu"
    assert cer.clitic_space_normalise("fis- seħħ") == "fis-seħħ"
    assert cer.clitic_space_normalise("biż- żmien") == "biż-żmien"


def test_clitic_space_normalise_case_insensitive_preserves_case():
    assert cer.clitic_space_normalise("Il- Gvernatur") == "Il-Gvernatur"
    assert cer.clitic_space_normalise("L- Akkwist") == "L-Akkwist"
    assert cer.clitic_space_normalise("GĦALL- finijiet") == "GĦALL-finijiet"


def test_clitic_space_normalise_idempotent():
    samples = [
        "il-kelb fid-dar tas-suq",
        "0 — Għadha mhux fis-seħħ.",
        "il- foo and tal- bar and l- baz",
        "no clitics here at all",
        "",
    ]
    for s in samples:
        once = cer.clitic_space_normalise(s)
        twice = cer.clitic_space_normalise(once)
        assert once == twice, f"not idempotent on {s!r}: {once!r} -> {twice!r}"


def test_clitic_space_normalise_preserves_structural_hyphen():
    text = "Il-kelb tat-tifel fid-dar tas-suq."
    assert cer.clitic_space_normalise(text) == text


def test_clitic_space_normalise_preserves_unicode_dashes():
    en_dash = "0 – Għadha mhux fis-seħħ"
    em_dash = "fil-Parlament — jipproponi"
    assert cer.clitic_space_normalise(en_dash) == en_dash
    assert cer.clitic_space_normalise(em_dash) == em_dash
    en_with_clitic_noise = "0 – il- foo"
    assert cer.clitic_space_normalise(en_with_clitic_noise) == "0 – il-foo"


def test_clitic_space_normalise_does_not_touch_non_clitic_hyphens():
    assert cer.clitic_space_normalise("Marie- Louise") == "Marie- Louise"
    assert cer.clitic_space_normalise("oxy- / oxi-") == "oxy- / oxi-"
    assert cer.clitic_space_normalise("foo- bar") == "foo- bar"


def test_compute_cer_dual_zero_on_identical():
    refs = ["il-kelb", "id-dar", "fis-seħħ"]
    out = cer.compute_cer_dual(refs, refs)
    assert out["raw_cer"] == pytest.approx(0.0)
    assert out["normalised_cer"] == pytest.approx(0.0)
    assert out["delta"] == pytest.approx(0.0)


def test_compute_cer_dual_delta_zero_on_already_normalised():
    refs = ["il-foo bar", "tal-Kummissjoni hawn"]
    hyps = ["il-foo bar", "tal-Kummissjoni hawn"]
    out = cer.compute_cer_dual(refs, hyps)
    assert out["delta"] == pytest.approx(0.0)


def test_compute_cer_dual_positive_delta_on_gold_side_noise():
    refs = ["il- foo bar baz qux"]
    hyps = ["il-foo bar baz qux"]
    out = cer.compute_cer_dual(refs, hyps)
    assert out["raw_cer"] > 0.0
    assert out["normalised_cer"] == pytest.approx(0.0)
    assert out["delta"] > 0.0


def test_compute_cer_dual_shape_keys():
    out = cer.compute_cer_dual(["a"], ["a"])
    assert set(out.keys()) == {"raw_cer", "normalised_cer", "delta"}
