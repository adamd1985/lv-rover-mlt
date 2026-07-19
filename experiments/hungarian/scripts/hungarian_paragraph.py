"""Hungarian paragraph renderer. Reuses the generic PIL render/justify/augment
primitives from the Maltese pipeline unchanged; drops only the Maltese-specific
clitic/structural-hyphen classification, which Hungarian does not need (its
hyphenation is a plain typographic convention - see plan). Same method
(SynthTIGER-style render -> augment -> Tesseract fine-tune), different
language, per the explicit "no new architecture" constraint.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import pyphen
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/adamd1985/doceng2026")
sys.path.insert(0, str(ROOT))

from src.datagen.maltese_paragraph import _render_with_pil  # generic, reused as-is

HU_CANARY = "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"
_DIC = pyphen.Pyphen(lang="hu")


@dataclass
class HuLayoutConfig:
    # DPI/JPEG-quality calibration matches the corrected Maltese config
    # (post-hoc fix): real crop resolution, not naive 300 DPI.
    dpi: int = 200
    font_pt_lo: int = 8
    font_pt_hi: int = 14
    line_spacing_lo: float = 1.0
    line_spacing_hi: float = 1.6
    paragraph_width_lo: int = 400
    paragraph_width_hi: int = 1200
    pad_x: int = 24
    pad_y: int = 18
    justify_p: float = 0.45
    hyphenation_rate: float = 0.06
    max_lines: int = 8


def _wrap_to_lines(text: str, font, max_width_px: int, draw) -> List[str]:
    """Plain greedy word-wrap - no clitic-hyphen special-casing needed for Hungarian."""
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if draw.textlength(trial, font=font) <= max_width_px or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _apply_hyphenation(lines: List[str], rate: float, rng: random.Random) -> List[str]:
    """Synthetic line-break hyphenation using Pyphen's real syllable points,
    matching how Pyphen will also be used at inference for line-join repair."""
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        has_next = i + 1 < len(lines)
        words = ln.split()
        if has_next and rate > 0 and rng.random() < rate and len(words) >= 2 and len(words[-1]) >= 6:
            last = words[-1]
            points = _DIC.positions(last)
            cut = points[len(points) // 2] if points else len(last) // 2
            head, tail = last[:cut], last[cut:]
            printed_line = " ".join(words[:-1] + [head + "-"])
            next_line_words = lines[i + 1].split()
            next_printed = " ".join([tail] + next_line_words)
            out.append(printed_line)
            out.append(next_printed)
            i += 2
        else:
            out.append(ln)
            i += 1
    return out


class HungarianParagraph:
    def __init__(self, font_sampler, layout: HuLayoutConfig = None, rng: random.Random = None):
        self.fonts = font_sampler
        self.layout = layout or HuLayoutConfig()
        self.rng = rng or random.Random()

    def render_one(self, text: str) -> Tuple[Image.Image, str]:
        img, _, _ = self.render_with_lines(text)
        return img, text

    def render_with_lines(self, text: str) -> Tuple[Image.Image, List[str], List[Tuple[int, int, int, int]]]:
        """Returns (paragraph image, printed lines, per-line bboxes) so callers
        can crop individual line images - Tesseract fine-tuning needs line
        crops, matching how the Maltese LSTM was fine-tuned on line crops cut
        from paragraph shards, not whole paragraph images."""
        L = self.layout
        face = self.fonts.sample()
        pt = self.rng.randint(L.font_pt_lo, L.font_pt_hi)
        px = int(round(pt * L.dpi / 72))
        font = ImageFont.truetype(str(face.path), px)
        width_px = self.rng.randint(L.paragraph_width_lo, L.paragraph_width_hi)
        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)
        lines = _wrap_to_lines(text, font, width_px - 2 * L.pad_x, draw)
        if L.max_lines and len(lines) > L.max_lines:
            lines = lines[: L.max_lines]
        printed = _apply_hyphenation(lines, L.hyphenation_rate, self.rng)
        spacing = self.rng.uniform(L.line_spacing_lo, L.line_spacing_hi)
        justify = self.rng.random() < L.justify_p
        img, bboxes = _render_with_pil(printed, font, spacing, L.pad_x, L.pad_y, width_px, justify)
        return img, printed, bboxes

    def __call__(self, corpus_iter: Iterable[str]) -> Iterator[Tuple[Image.Image, str]]:
        for text in corpus_iter:
            yield self.render_one(text)
