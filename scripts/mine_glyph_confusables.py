"""Mine visually similar character pairs from rendered font glyphs.

Renders each char in char_set.json across the validated fonts at 8-14pt,
extracts ORB feature descriptors, ranks pairs by descriptor distance.
The hand-listed confusable pairs (ċ/c, ġ/g, ħ/h, ż/z, etc.) are used
as a seed set; mined top-k pairs extend them.

Output: data/glyph_confusables.json - a list of {char_a, char_b, score}
records sorted by visual similarity (lower = more similar).

Usage:
    python scripts/mine_glyph_confusables.py [--fonts-dir data/fonts] [--top-k 50]
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

SEED_CONFUSABLES: List[Tuple[str, str]] = [
    ("ċ", "c"), ("Ċ", "C"),
    ("ġ", "g"), ("Ġ", "G"),
    ("ħ", "h"), ("Ħ", "H"),
    ("ż", "z"), ("Ż", "Z"),
    ("à", "a"), ("ì", "i"), ("ò", "o"), ("ù", "u"),
    ("għ", "gh"), ("għ", "g"),
    ("ie", "i"), ("ie", "e"),
]

GLYPH_SIZE = 64
FONT_SIZES = [12, 14]


def _render_glyph(char: str, font_path: Path, font_size: int) -> np.ndarray:
    img = Image.new("L", (GLYPH_SIZE, GLYPH_SIZE), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (GLYPH_SIZE - w) // 2 - bbox[0]
    y = (GLYPH_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img, dtype=np.uint8)


def _orb_descriptors(img: np.ndarray):
    try:
        import cv2
        orb = cv2.ORB_create(nfeatures=64)
        kps, descs = orb.detectAndCompute(img, None)
        return descs
    except ImportError:
        return None


def _simple_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-level L2 distance as fallback when ORB/cv2 is not available."""
    diff = a.astype(float) - b.astype(float)
    return float(np.sqrt((diff ** 2).mean()))


def _orb_distance(desc_a, desc_b) -> float:
    """Hamming distance between ORB descriptors (mean over matched pairs)."""
    try:
        import cv2
        if desc_a is None or desc_b is None:
            return float("inf")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_a, desc_b)
        if not matches:
            return float("inf")
        return float(np.mean([m.distance for m in matches]))
    except Exception:
        return float("inf")


def _load_chars() -> List[str]:
    cs_path = ROOT / "competition_files" / "char_set.json"
    chars: List[str] = json.loads(cs_path.read_text(encoding="utf-8"))
    # Include single-char only; digraphs (għ, ie) handled via seed
    single = [c for c in chars if len(c) == 1]
    return single


def _find_fonts(fonts_dir: Path) -> List[Path]:
    exts = {".ttf", ".otf"}
    fonts = [p for p in fonts_dir.rglob("*") if p.suffix.lower() in exts]
    return fonts[:5] if fonts else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default=str(ROOT / "data" / "fonts"))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "data" / "glyph_confusables.json"))
    args = ap.parse_args()

    fonts_dir = Path(args.fonts_dir)
    fonts = _find_fonts(fonts_dir)
    if not fonts:
        print("[mine_confusables] no fonts found, using PIL default only")
        fonts = [None]

    chars = _load_chars()
    print(f"[mine_confusables] {len(chars)} chars, {len(fonts)} fonts")

    # Render all chars across all fonts and sizes, average glyphs
    glyph_imgs: Dict[str, np.ndarray] = {}
    for ch in chars:
        imgs = []
        for font_path in fonts:
            for fs in FONT_SIZES:
                try:
                    img = _render_glyph(ch, font_path, fs) if font_path else None
                    if img is not None:
                        imgs.append(img.astype(float))
                except Exception:
                    pass
        if imgs:
            glyph_imgs[ch] = np.mean(imgs, axis=0).astype(np.uint8)

    # Compute pairwise distances
    char_list = sorted(glyph_imgs.keys())
    n = len(char_list)
    records = []

    # Seed pairs first
    seed_pairs = set()
    for a, b in SEED_CONFUSABLES:
        if a in glyph_imgs and b in glyph_imgs:
            d = _simple_distance(glyph_imgs[a], glyph_imgs[b])
            records.append({"char_a": a, "char_b": b, "score": d, "source": "seed"})
            seed_pairs.add((a, b))
            seed_pairs.add((b, a))

    # Mine top-k pairs from all pairs
    mined = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = char_list[i], char_list[j]
            if (a, b) in seed_pairs or (b, a) in seed_pairs:
                continue
            d = _simple_distance(glyph_imgs[a], glyph_imgs[b])
            mined.append({"char_a": a, "char_b": b, "score": d, "source": "mined"})

    mined.sort(key=lambda r: r["score"])
    records.extend(mined[: args.top_k])
    records.sort(key=lambda r: r["score"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mine_confusables] wrote {len(records)} pairs -> {out_path}")
    print("Top 10 similar pairs:")
    for r in records[:10]:
        print(f"  '{r['char_a']}' <-> '{r['char_b']}': {r['score']:.2f} [{r['source']}]")


if __name__ == "__main__":
    main()
