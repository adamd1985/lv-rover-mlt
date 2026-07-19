"""Focused tests for the arbitration/gating logic in competition_transcriber.py.

Unlike test_competition_transcriber.py, these exercise pure-Python logic
(no Tesseract binary or model files needed) and always run.
"""
from __future__ import annotations

import competition_transcriber as ct


def test_no_easyocr_attribute_or_reference():
    assert not hasattr(ct.CompetitionTranscriber, "easyocr")
    assert not hasattr(ct.CompetitionTranscriber, "_try_load_easyocr")
    assert not hasattr(ct.CompetitionTranscriber, "_easyocr_paragraph")
    assert "easyocr" not in open(ct.__file__, encoding="utf-8").read().lower()


def test_exactly_five_tesseract_streams():
    # mlt, mlt+ita (anchor), mlt+ita+fra, stock, and mlt+ita on a 2x-upscaled
    # image - five structurally distinct streams, no sixth candidate.
    langs = {ct._TESS_LANG_PRIMARY, ct._TESS_LANG_AUGMENTED,
             ct._TESS_LANG_ROMANCE, ct._TESS_LANG_STOCK}
    assert langs == {"mlt", "mlt+ita", "mlt+ita+fra", "mltstock"}
    # The fifth stream reuses _TESS_LANG_AUGMENTED at 2x scale (image-level
    # variation, not a distinct language chain), per transcribe()'s up_out.


def test_anchor_fallback_ratio_constant():
    assert ct._ITA_FALLBACK_RATIO == 0.6


def test_plurality_no_quorum_single_stream_wins():
    lexicon = {"għadha", "ghadha"}
    router = ct._CrossEngineRouter(lexicon)
    # Anchor is out-of-lexicon; only one candidate stream proposes an
    # eligible in-lexicon replacement. No quorum is required for it to win.
    result = router.combine_lv("Xhadha", ["Għadha", "Xhadha", "Xhadha"])
    assert result == "Għadha"


def test_duplicate_hypothesis_collapse_tie_break():
    lexicon = {"ghadha", "vadha"}
    router = ct._CrossEngineRouter(lexicon)
    # Two streams propose the same swap ("Ghadha"), two propose a different
    # swap ("Vadha"). Duplicates must collapse into one Counter entry each
    # (count 2 apiece) rather than each occurrence acting as its own
    # candidate; the tie then breaks on first-seen order, not raw stream
    # index, so "Ghadha" (proposed first) wins over the equally-voted
    # "Vadha".
    result = router.combine_lv("Xhadha", ["Ghadha", "Ghadha", "Vadha", "Vadha"])
    assert result == "Ghadha"


def test_diacritic_restoration_beats_already_valid_anchor():
    lexicon = {"ghadha", "għadha"}
    router = ct._CrossEngineRouter(lexicon)
    # Anchor "Ghadha" is itself lexicon-valid (diacritic-stripped form), but
    # a candidate with strictly more canary diacritics and equal validity
    # should still be adopted via the diacritic-restoration gate.
    decision = router._decide("Ghadha", "Għadha")
    assert decision == "Għadha"


def test_diacritic_restoration_does_not_fire_without_richer_diacritics():
    lexicon = {"ghadha"}
    router = ct._CrossEngineRouter(lexicon)
    decision = router._decide("Ghadha", "Ghadha")
    assert decision == "Ghadha"


def test_label_normalization_unchanged():
    assert ct._fix_lead_marker("0 - Something") == "0 — Something"
    assert ct._fix_apostrophe("l-Ewwel 'darba'") == "l-Ewwel ‘darba’"
    assert ct._fix_doublequote('He said "hello"') == "He said “hello”"
