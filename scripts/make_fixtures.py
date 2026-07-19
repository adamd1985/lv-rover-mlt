"""Render a handful of Maltese paragraph crops as an offline smoke-test set.

The competition development set is the organisers' to distribute, so it is not
mirrored here. These fixtures stand in for it: same JSON shape
(``[{"image", "text", "as_lines"}, ...]``), same crop style, our own text.

They are small enough to keep in git, so ``run.sh`` and the tests work with no
network and no organiser data. Regenerate with::

    python scripts/make_fixtures.py

Uses the validated font pool under ``data/fonts/`` when present, otherwise a
system DejaVu face - both cover the Maltese canary glyphs (ċ ġ ħ ż).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "dev"

# Public-domain / self-written Maltese lines exercising the features the
# pipeline is built around: canary diacritics, the clitic article hyphen,
# a soft line-break hyphen, an em-dash clause marker, and code-switched English.
SAMPLES: list[list[str]] = [
    ["Il-ktieb tal-istorja ta' Malta jinsab fil-librerija", "nazzjonali ta' Beltna."],
    ["0 — Għadha mhux fis-", "seħħ."],
    ["Iż-żewġ aħwa marru lejn ir-raħal biex jaraw", "iċ-ċimiterju l-antik."],
    ["The Maltese Language Resource Server hosts", "Korpus Malti for research use."],
    ["Ġużeppi qal li x-xogħol tal-ġurnata kien", "iebes imma sabiħ."],
]

FONT_CANDIDATES = [
    "data/fonts/printed/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    root = Path(__file__).resolve().parents[1]
    for cand in FONT_CANDIDATES:
        p = Path(cand)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise SystemExit(
        "no usable font found; install DejaVu (fonts-dejavu) or populate data/fonts/"
    )


def render(lines: list[str], font: ImageFont.FreeTypeFont, pad: int = 12) -> Image.Image:
    probe = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(probe)
    widths, heights = [], []
    for ln in lines:
        box = d.textbbox((0, 0), ln, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    leading = int(max(heights) * 1.6)
    img = Image.new("RGB", (max(widths) + 2 * pad, leading * len(lines) + 2 * pad), "white")
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((pad, pad + i * leading), ln, fill="black", font=font)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _load_font(28)
    manifest = []
    for i, lines in enumerate(SAMPLES, start=1):
        name = f"{i:03d}.png"
        # paragraph gold: join lines, repair the soft hyphen, keep clitic hyphens
        text = ""
        for j, ln in enumerate(lines):
            if j == 0:
                text = ln
            elif text.endswith("-"):
                text += ln
            else:
                text += " " + ln
        render(lines, font).save(OUT_DIR / name, optimize=True)
        manifest.append({"image": name, "text": text, "as_lines": lines})
    (OUT_DIR / "texts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifest)} fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
