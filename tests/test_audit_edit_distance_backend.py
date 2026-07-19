"""Regression guard for a hand-rolled Levenshtein reimplementation
silently drifted an audit's point estimate away from the frozen headline CER
(0.00503 vs 0.00617), and the same function had been copy-pasted into
scripts/bootstrap_stats.py, feeding the anchor-to-ensemble CI, the oracle
diagnostics, and both cross-language audits.

Every script that recomputes a CER number reported in the paper must call
jiwer (directly, or through one of the two functions tested here) and never
reimplement edit distance. This module pins that contract two ways: a
numeric parity check against jiwer on a battery of pairs, and a source-level
guard that fails if either function stops calling jiwer.
"""
from __future__ import annotations

import inspect

import jiwer
import pytest

from scripts.audit_bootstrap_full_chain import _edit_distance as full_chain_edit_distance
from scripts.bootstrap_stats import edit_distance as shared_edit_distance

PAIRS = [
    ("il-kelb fid-dar tas-suq", "il-kelb fid-dar tas-suq"),  # identical
    ("Ħrejjef, stejjer u kitba oħra", "Hrejjef, stejjer u kitba ohra"),  # diacritic drop
    ("Ġesù jħobb", "Gesu jhobb"),  # diacritic + grave drop
    ("0 — Għadha mhux fis-seħħ", "0 - Ghadha mhux fis-sehh"),  # dash + diacritics
    ("", "abc"),  # empty ref
    ("abc", ""),  # empty hyp
    ("", ""),  # both empty
    ("a", "b"),  # single substitution
    ("abcdef", "abcdf"),  # single deletion
    ("abcdef", "abcxdef"),  # single insertion
    ("Curlew Sandpiper", "Curlew  Sandpiper"),  # spacing noise
]


@pytest.mark.parametrize("ref,hyp", PAIRS)
def test_shared_edit_distance_matches_jiwer(ref: str, hyp: str) -> None:
    if not ref:
        pytest.skip("jiwer.cer is undefined on an empty reference")
    expected = jiwer.cer([ref], [hyp])
    got = shared_edit_distance(ref, hyp) / len(ref)
    assert got == pytest.approx(expected), (
        f"bootstrap_stats.edit_distance({ref!r}, {hyp!r}) diverged from jiwer.cer "
        f"({got} vs {expected}) - do not reimplement Levenshtein here"
    )


@pytest.mark.parametrize("ref,hyp", PAIRS)
def test_full_chain_edit_distance_matches_jiwer(ref: str, hyp: str) -> None:
    if not ref:
        pytest.skip("jiwer.cer is undefined on an empty reference")
    expected = jiwer.cer([ref], [hyp])
    got = full_chain_edit_distance(ref, hyp) / len(ref)
    assert got == pytest.approx(expected), (
        f"audit_bootstrap_full_chain._edit_distance({ref!r}, {hyp!r}) diverged from "
        f"jiwer.cer ({got} vs {expected}) - do not reimplement Levenshtein here"
    )


def test_shared_edit_distance_calls_jiwer_not_a_hand_rolled_dp() -> None:
    """Source-level guard: a copy-paste of the old DP table reintroduces the
    bug even if it is numerically correct in isolation, because a future
    edit can silently diverge it again. Require the jiwer call to stay.
    """
    src = inspect.getsource(shared_edit_distance)
    assert "jiwer" in src, "edit_distance() must call jiwer, not reimplement Levenshtein"
    assert "prev = list(range" not in src, (
        "a hand-rolled Levenshtein DP table reappeared in bootstrap_stats.edit_distance"
    )


def test_full_chain_edit_distance_calls_jiwer_not_a_hand_rolled_dp() -> None:
    src = inspect.getsource(full_chain_edit_distance)
    assert "jiwer" in src, "_edit_distance() must call jiwer, not reimplement Levenshtein"
    assert "prev = list(range" not in src, (
        "a hand-rolled Levenshtein DP table reappeared in audit_bootstrap_full_chain._edit_distance"
    )
