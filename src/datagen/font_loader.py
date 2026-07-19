"""Font loader and weighted sampler for the synth pipeline.

Scans `data/fonts/printed/` and `data/fonts/handwriting/`. Files left in the
root are treated as legacy printed. Every face is gated by a cmap canary on
the Maltese diacritic set; failing faces are dropped with a warning. The
returned sampler draws 95 percent from printed, 5 percent from handwriting
(configurable). Output schema matches what `font-checker` consumes.
"""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# Canary glyphs a font must cover: Maltese diacritics plus the lowercase
# grave-accented vowels in the gold inventory (à ì ò ù; see char_set.json).
CANARY = "ĊċĠġĦħŻżàìòù"
# Dashes a font should render. En-dash is image-only; we still want fonts that
# can draw it, since the renderer uses it as a visual-to-canonical signal.
DASHES = "-–—"


@dataclass(frozen=True)
class FontFace:
    path: Path
    bucket: str
    family: str


def _cmap_ok(path: Path, required: str) -> bool:
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(str(path), lazy=True)
    except Exception:
        return False
    cmap = f.getBestCmap()
    return all(ord(c) in cmap for c in required)


def _family_name(path: Path) -> str:
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(str(path), lazy=True)
        for record in f["name"].names:
            if record.nameID == 1:
                try:
                    return record.toUnicode()
                except Exception:
                    continue
    except Exception:
        pass
    return path.stem


def _ensure_layout(root: Path) -> None:
    (root / "printed").mkdir(parents=True, exist_ok=True)
    (root / "handwriting").mkdir(parents=True, exist_ok=True)


def _scan_bucket(root: Path, bucket: str, canary: str) -> List[FontFace]:
    found: List[FontFace] = []
    for ext in ("*.ttf", "*.otf"):
        for p in sorted(root.glob(ext)):
            if not _cmap_ok(p, canary):
                warnings.warn(f"font {p.name} fails canary, skipping")
                continue
            found.append(FontFace(path=p, bucket=bucket, family=_family_name(p)))
    return found


class FontSampler:
    def __init__(
        self,
        printed: Sequence[FontFace],
        handwriting: Sequence[FontFace],
        handwriting_rate: float = 0.05,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.printed = list(printed)
        self.handwriting = list(handwriting)
        self.handwriting_rate = handwriting_rate
        self.rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self.printed) + len(self.handwriting)

    def sample(self) -> FontFace:
        if not self.printed and not self.handwriting:
            raise RuntimeError("no fonts available; populate data/fonts/")
        if self.handwriting and self.rng.random() < self.handwriting_rate:
            return self.rng.choice(self.handwriting)
        if not self.printed:
            return self.rng.choice(self.handwriting)
        return self.rng.choice(self.printed)

    def all_faces(self) -> List[FontFace]:
        return list(self.printed) + list(self.handwriting)


def load_fonts(
    fonts_dir: Path,
    handwriting_rate: float = 0.05,
    rng: Optional[random.Random] = None,
    fallback_system: bool = False,
) -> FontSampler:
    fonts_dir = Path(fonts_dir)
    _ensure_layout(fonts_dir)

    printed = _scan_bucket(fonts_dir / "printed", "printed", CANARY)
    legacy = _scan_bucket(fonts_dir, "printed", CANARY)
    printed = printed + [f for f in legacy if f.path.parent == fonts_dir]
    handwriting = _scan_bucket(fonts_dir / "handwriting", "handwriting", CANARY)

    if not printed and fallback_system:
        printed = _system_fallback(CANARY)

    return FontSampler(printed, handwriting, handwriting_rate=handwriting_rate, rng=rng)


def _system_fallback(canary: str) -> List[FontFace]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    ]
    out: List[FontFace] = []
    for c in candidates:
        p = Path(c)
        if p.exists() and _cmap_ok(p, canary):
            out.append(FontFace(path=p, bucket="printed", family=_family_name(p)))
    return out
