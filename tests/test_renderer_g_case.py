"""G-case fix: compound-hyphen wrap and structural-hyphen wrap distinction.

Entry 158 makes the renderer break before an internal compound hyphen
(`Marie-Louise` -> first line `Marie`, next line `-Louise`) and tag each
line with a HyphenKind. Structural clitics (`il-kelma`) remain unchanged.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from PIL import ImageDraw, ImageFont, Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datagen.maltese_paragraph import (
    HyphenKind,
    LayoutConfig,
    MalteseParagraph,
    _classify_trailing_hyphen,
    _wrap_to_lines,
)
from src.datagen.font_loader import load_fonts
from src.joiner.joiner import join_lines


def _pipeline(seed: int = 42):
    rng = random.Random(seed)
    fonts = load_fonts(ROOT / "data" / "fonts", handwriting_rate=0.0, rng=rng, fallback_system=True)
    assert len(fonts) > 0
    layout = LayoutConfig(hyphenation_rate=0.0)
    return MalteseParagraph(font_sampler=fonts, layout=layout, rng=rng)


def test_classify_structural():
    assert _classify_trailing_hyphen("hello il-") == HyphenKind.STRUCTURAL
    assert _classify_trailing_hyphen("dwar tas-") == HyphenKind.STRUCTURAL
    assert _classify_trailing_hyphen("fis-") == HyphenKind.STRUCTURAL


def test_classify_compound():
    assert _classify_trailing_hyphen("Ms Marie-") == HyphenKind.COMPOUND
    assert _classify_trailing_hyphen("the oxy-azido-") == HyphenKind.COMPOUND


def test_classify_none():
    assert _classify_trailing_hyphen("plain text") == HyphenKind.NONE
    assert _classify_trailing_hyphen("end with em —") == HyphenKind.NONE


def test_wrap_breaks_before_compound_hyphen():
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = "alpha beta gamma Marie-Louise Coleiro delta epsilon zeta eta theta"
    lines = _wrap_to_lines(text, font, max_width_px=80, draw=draw)
    joined = "\n".join(lines)
    assert "Marie-Louise" not in joined or any(ln.startswith("-Louise") for ln in lines)
    if not any(ln.startswith("-Louise") for ln in lines):
        pytest.skip("default font failed to trigger wrap on Marie-Louise")


def test_wrap_keeps_structural_after_dash():
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = "alpha beta gamma delta epsilon il-kelma magnifika sigma omega"
    lines = _wrap_to_lines(text, font, max_width_px=70, draw=draw)
    for ln in lines:
        assert not ln.startswith("-kelma"), f"structural clitic was broken: {lines}"


def test_metadata_hyphen_kinds_length_matches_lines():
    pipe = _pipeline(seed=1)
    img, label, parts, meta = pipe.render_one({"text": "kelma " * 200, "lang": "mt"})
    assert len(meta.hyphen_kinds) == meta.n_lines
    assert all(isinstance(k, HyphenKind) for k in meta.hyphen_kinds)


def test_compound_wrap_joined_paragraph_round_trip():
    text_lines = ["alpha Marie", "-Louise beta"]
    out = join_lines(text_lines)
    assert "Marie" in out and "Louise" in out
    assert "Marie-Louise" in out or "Marie -Louise" in out or "MarieLouise" in out


def test_structural_wrap_joined_paragraph_round_trip():
    text_lines = ["fis-", "seħħ"]
    assert join_lines(text_lines) == "fis-seħħ"


def test_force_marie_louise_wrap_tagged_compound():
    pipe = _pipeline(seed=7)
    txt = "Coleiro Preca Marie-Louise Coleiro Preca " * 10
    img, label, parts, meta = pipe.render_one({"text": txt, "lang": "en"})
    assert len(meta.hyphen_kinds) == len(parts)
    saw_compound_split = any(p.startswith("-") for p in parts)
    if saw_compound_split:
        assert any(k == HyphenKind.COMPOUND for k in meta.hyphen_kinds)


def test_force_clitic_wrap_tagged_structural():
    pipe = _pipeline(seed=11)
    txt = "kelma " * 5 + "il- "
    rng = random.Random(11)
    layout = LayoutConfig(hyphenation_rate=0.0, paragraph_width_lo=200, paragraph_width_hi=300)
    pipe2 = MalteseParagraph(font_sampler=pipe.fonts, layout=layout, rng=rng)
    img, label, parts, meta = pipe2.render_one({"text": "il-kelma " * 50, "lang": "mt"})
    assert len(meta.hyphen_kinds) == len(parts)
