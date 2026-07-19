"""Disqualification guard: the deliverable must never mangle a Maltese
diacritic (ċ ġ ħ ż għ). A model whose tokenizer mangles these is
disqualified at the encoder stage.

Our deliverable has no learned tokenizer - Tesseract emits raw Unicode
codepoints - so the only mangling surface is our own post-processing. These
tests prove each post-processing step is diacritic-preserving, the curly-quote
fixes never touch a letter, NFC keeps the canaries precomposed and in the
117-char inventory, and `_strip_diac` (used only for comparison) never reaches
the output path.
"""
from __future__ import annotations

import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ct", str(ROOT / "competition_transcriber.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ct = _load_module()
CANARIES = "ċġħżĊĠĦŻ"
SAMPLES = [
    "fis-seħħ id-dar il-kelb",
    "Ġesù u l-orizzont",
    "x'ġara fiż-żmien tal-għaġeb",
    "Stħarriġ dwar iċ-ċittadini ż-żgħar",
    "0 - Għadha mhux fis-seħħ",
    "tiegħu, żwieġ, perċettivi",
]


def _diac_multiset(s: str) -> dict:
    return {c: s.count(c) for c in CANARIES if c in s}


@pytest.mark.parametrize("text", SAMPLES)
def test_normalise_preserves_diacritics(text):
    out = ct._normalise(text)
    assert _diac_multiset(out) == _diac_multiset(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_lead_marker_preserves_diacritics(text):
    out = ct._fix_lead_marker(text)
    assert _diac_multiset(out) == _diac_multiset(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_apostrophe_fix_preserves_diacritics(text):
    out = ct._fix_apostrophe(text)
    # the curly-quote fix only ever changes U+0027; no letter is touched
    assert _diac_multiset(out) == _diac_multiset(text)
    assert out.replace("’", "'").replace("‘", "'") == text


def test_canaries_are_precomposed_and_in_inventory():
    _org = ROOT / "competition_files" / "char_set.json"
    _path = _org if _org.exists() else ROOT / "data" / "char_set.json"
    cs = json.loads(_path.read_text(encoding="utf-8"))
    allowed = set(cs if isinstance(cs, list) else cs.keys())
    for c in CANARIES:
        assert c in allowed, f"{c!r} missing from char_set.json"
        assert unicodedata.normalize("NFC", c) == c, f"{c!r} not NFC-stable"
        assert len(unicodedata.normalize("NFC", c)) == 1


def test_router_diacritic_branch_only_adds_never_strips():
    lex = {w.lower() for w in ["żwieġ", "ċerti", "fis-seħħ", "iżda", "certi", "zwieg"]}
    r = ct._CrossEngineRouter(lex, max_swap_dist=2)
    # restores when a candidate has the diacritics
    assert r._decide("zwieg", "żwieġ") == "żwieġ"
    assert r._decide("fis-sehh", "fis-seħħ") == "fis-seħħ"  # hyphen preserved
    # never strips diacritics back to ASCII
    assert r._decide("żwieġ", "zwieg") == "żwieġ"
    # never substitutes a different word
    assert r._decide("kelb", "xqsd") == "kelb"


def test_strip_diac_never_in_output_path():
    # _strip_diac is a comparison-only helper; the transcribe output path must
    # not call it. Guard against accidental use by checking the source.
    src = (ROOT / "competition_transcriber.py").read_text()
    transcribe_src = src.split("def transcribe(", 1)[1]
    assert "_strip_diac" not in transcribe_src, \
        "_strip_diac must not be applied to output - comparison only"


def test_full_pipeline_preserves_diacritics_on_samples():
    # _normalise -> _fix_lead_marker -> _fix_apostrophe is the output tail.
    for text in SAMPLES:
        out = ct._fix_apostrophe(ct._fix_lead_marker(ct._normalise(text)))
        assert _diac_multiset(out) == _diac_multiset(text), text
