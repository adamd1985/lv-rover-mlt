"""Tesseract `tesstrain`-style training-data exporter.

Emits one single-line image per training sample plus a matching `.gt.txt`
ground-truth file, then builds the `.lstmf` feature file Tesseract LSTM
training consumes. Only the OUTPUT format is new: corpus, fonts, and the
augmentation primitives are the existing synth pipeline.

Per shard dir (sharded, deterministic from RANDOM_SEED):
    <out-root>/<shard>/
        000000.tif  000000.gt.txt  000000.box  000000.lstmf
        ...
        <shard>.training_files.txt   one .lstmf path per line
        _summary.json

lstmf generation: a single-line image is paired with a WordStr `.box` file
carrying the whole-line transcription, then `tesseract ... lstm.train` builds
the `.lstmf`. The box coordinates span the image; Tesseract re-aligns chars
internally during LSTM training.

Charset coverage: korpus_malti and the embedded fixture do not exercise the
full 117-char inventory in competition_files/char_set.json - symbols, curly
quotes, currency, and test-only glyphs (`= ô diamond` etc). COVERAGE_LINES
mixes natural-context lines with exhaustive space-separated packs so every
codepoint appears at least once when all coverage lines are emitted.

Augmentations follow the narrow Tesseract-safe set: Gaussian
blur, brightness/contrast jitter, low-density salt-and-pepper, slight
rotation, JPEG artefacts, and colored backgrounds. Perspective warp and
elastic distortion are dropped - they break Tesseract's height normaliser.
Colored backgrounds (near-white, light grey, light blue, light yellow) are
the highest-leverage augmentation for Maltese PDFs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datagen.augmentations import (
    _blur,
    _jitter,
    _jpeg_recompress,
    _rotate,
    _salt_pepper,
)
from src.datagen.corpus_loader import iter_paragraphs
from src.datagen.font_loader import load_fonts
from src.datagen.maltese_paragraph import EN_DASH, EM_DASH, SOFT_HYPHEN

RANDOM_SEED = 42

# The 117-character label inventory. Prefers the organisers' copy when their
# asset package is present, otherwise the vendored list under data/.
_ORGANISER_CHAR_SET = ROOT / "competition_files" / "char_set.json"
CHAR_SET_PATH = (
    _ORGANISER_CHAR_SET if _ORGANISER_CHAR_SET.exists() else ROOT / "data" / "char_set.json"
)

# Light backgrounds typical of Maltese PDF scans. RGB.
BACKGROUNDS: Tuple[Tuple[int, int, int], ...] = (
    (255, 255, 255),  # near-white
    (248, 248, 246),  # off-white
    (236, 236, 236),  # light grey
    (232, 238, 246),  # light blue
    (250, 247, 230),  # light yellow
)

# Natural-context lines that exercise the symbol/quote/currency glyphs
# korpus_malti rarely produces, so the LSTM sees them in plausible context.
_CONTEXT_LINES: Tuple[str, ...] = (
    "Prezz: 12 + 3 = 15 EUR jew €15 [kollox].",
    "Ir-ras qalet “iva” u ‘le’ fl-istess ħin.",
    "Nota⁴ dwar id-dija ² u t-temp ¹ bil-massa.",
    "Il-ktieb © 2024 — awtur maħruf — preżzju €9.",
    "Simboli: • punt, ♢ djamant, Ø vojt, ł Pollakk.",
    "Ismijiet: Għawdex, Ġesù, Cité, naïf, façade, ôtel.",
    "Lingwa oħra: año, über, öko, sólo, más, ríò.",
    "Test rari: ā twil, ỹ vjetnamiż, ú ü ó ò.",
    "Kodiċi: A1_B2 & C3; (x=2) {y} \"kwota\" 'singola'.",
    "Domanda? Tweġiba! Numri 0123456789 + - / : ; .",
)


def _exhaustive_lines(char_set: List[str]) -> Tuple[str, ...]:
    """Space-separated packs that render every non-space char in the
    inventory at least once. The guarantee COVERAGE makes."""
    chars = [c for c in char_set if c != " "]
    return tuple(" ".join(chars[i : i + 22]) for i in range(0, len(chars), 22))


def load_char_set() -> List[str]:
    return json.loads(CHAR_SET_PATH.read_text(encoding="utf-8"))


COVERAGE_LINES: Tuple[str, ...] = _CONTEXT_LINES + _exhaustive_lines(load_char_set())


def _normalise_label(text: str) -> str:
    """Labels carry only ASCII hyphen and em-dash. En-dash is image-only; soft
    hyphen never appears in gold. NFC matches the organiser reference."""
    text = text.replace(SOFT_HYPHEN, "").replace(EN_DASH, EM_DASH)
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _row_text(row: dict) -> str:
    """korpus rows can carry `text` as a list of strings."""
    raw = row.get("text", "")
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    return str(raw)


def _render_line(text: str, font: ImageFont.FreeTypeFont, pad: int, bg: Tuple[int, int, int]) -> Image.Image:
    scratch = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    w = int(scratch.textlength(text, font=font))
    ascent, descent = font.getmetrics()
    h = ascent + descent
    img = Image.new("RGB", (max(w, 1) + 2 * pad, h + 2 * pad), color=bg)
    ImageDraw.Draw(img).text((pad, pad), text, font=font, fill=(0, 0, 0))
    return img


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Tesseract-safe augmentation set. No perspective warp, no elastic."""
    npr = np.random.default_rng(rng.randint(0, 2**31 - 1))
    img = _rotate(img, rng.uniform(-1.5, 1.5))
    img = _blur(img, rng.uniform(0.3, 0.8))
    img = _jitter(img, rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12))
    arr = np.asarray(img.convert("RGB"))
    arr = _salt_pepper(arr, 0.006, npr)
    img = Image.fromarray(arr)
    return _jpeg_recompress(img, rng.randint(60, 92))


def _iter_line_texts(
    count: int, english_frac: float, rng: random.Random, skip_corpus: int = 0
) -> Iterator[str]:
    """Yield `count` line-shaped strings. The first block is the charset
    coverage lines (repeated so each rare glyph is well-sampled), the rest are
    corpus paragraphs split into wrapped-width line pieces.

    `skip_corpus` discards that many corpus-derived lines before yielding, so a
    later shard draws fresh corpus text instead of repeating shard 0. Coverage
    lines are only emitted when `skip_corpus` is 0 (shard 0 carries them)."""
    emitted = 0
    if skip_corpus == 0:
        coverage_budget = min(count, max(len(COVERAGE_LINES) * 6, count // 20))
        while emitted < coverage_budget:
            yield COVERAGE_LINES[emitted % len(COVERAGE_LINES)]
            emitted += 1

    corpus = iter_paragraphs(english_frac=english_frac, use_streaming=True, rng=rng)
    seen = 0  # corpus pieces produced overall, including the skipped prefix
    while emitted < count:
        try:
            row = next(corpus)
        except StopIteration:
            break
        for chunk in _row_text(row).split("\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            words = chunk.split()
            step = rng.randint(6, 14)
            for i in range(0, len(words), step):
                piece = " ".join(words[i : i + step])
                if len(piece) < 3:
                    continue
                seen += 1
                if seen <= skip_corpus:
                    continue
                yield piece
                emitted += 1
                if emitted >= count:
                    return


def _write_box(box_path: Path, label: str, size: Tuple[int, int]) -> None:
    """WordStr box: whole-line transcription spanning the image. Tesseract
    re-aligns characters internally during LSTM training."""
    w, h = size
    box_path.write_text(
        f"WordStr 0 0 {w} {h} 0 #{label}\n\t 0 0 {w} {h} 0\n", encoding="utf-8"
    )


def _make_lstmf(
    tif_path: Path, box_path: Path, out_base: Path, tessdata_dir: Path, lang: str, env: Dict[str, str]
) -> Path | None:
    cmd = [
        "tesseract",
        str(tif_path),
        str(out_base),
        "--tessdata-dir",
        str(tessdata_dir),
        "--psm",
        "7",
        "-l",
        lang,
        "lstm.train",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lstmf = out_base.with_suffix(".lstmf")
    if res.returncode != 0 or not lstmf.is_file():
        print(f"[tesstrain] lstm.train failed for {tif_path.name}: {res.stderr.strip()[:200]}")
        return None
    return lstmf


def export_shard(
    shard_name: str,
    count: int,
    out_root: Path,
    fonts_dir: Path,
    english_frac: float,
    tessdata_dir: Path,
    lang: str,
    make_lstmf: bool,
    clean_frac: float = 0.30,
    index_offset: int = 0,
) -> Dict:
    """Export one shard. `index_offset` shifts the corpus stream and the
    per-sample RNG so shards rendered with the same seed carry distinct
    content. Shard 0 (index_offset 0) also carries the charset coverage lines."""
    out_dir = out_root / shard_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rng = random.Random(RANDOM_SEED + index_offset)
    fonts = load_fonts(fonts_dir, handwriting_rate=0.05, rng=base_rng, fallback_system=True)
    if len(fonts) == 0:
        raise RuntimeError(f"no validated fonts under {fonts_dir}")

    texts = _iter_line_texts(count, english_frac, base_rng, skip_corpus=index_offset)
    charset = set(load_char_set())
    seen_chars: set = set()
    lstmf_paths: List[str] = []
    written = 0
    failures = 0
    env = {**os.environ}

    for local_idx in range(count):
        idx = index_offset + local_idx
        try:
            raw = next(texts)
        except StopIteration:
            break
        label = _normalise_label(raw)
        if not label:
            continue

        sample_rng = random.Random(RANDOM_SEED * 1_000_003 + idx)
        face = fonts.sample()
        pt = sample_rng.randint(20, 40)
        bg = sample_rng.choice(BACKGROUNDS)
        try:
            font = ImageFont.truetype(str(face.path), pt)
            img = _render_line(label, font, sample_rng.randint(6, 16), bg)
            if sample_rng.random() >= clean_frac:
                img = _augment(img, sample_rng)
        except Exception as exc:
            print(f"[tesstrain] render failure idx={idx}: {exc}")
            failures += 1
            continue

        stem = f"{idx:06d}"
        tif_path = out_dir / f"{stem}.tif"
        gt_path = out_dir / f"{stem}.gt.txt"
        box_path = out_dir / f"{stem}.box"
        img.convert("L").save(tif_path, format="TIFF")
        gt_path.write_text(label, encoding="utf-8")
        _write_box(box_path, label, img.size)
        seen_chars.update(label)
        written += 1

        if make_lstmf:
            lstmf = _make_lstmf(tif_path, box_path, out_dir / stem, tessdata_dir, lang, env)
            if lstmf is not None:
                lstmf_paths.append(str(lstmf.resolve()))

    if make_lstmf:
        list_path = out_dir / f"{shard_name}.training_files.txt"
        list_path.write_text("\n".join(lstmf_paths) + "\n", encoding="utf-8")

    missing = sorted(charset - seen_chars)
    summary = {
        "shard": shard_name,
        "count_requested": count,
        "written": written,
        "failures": failures,
        "lstmf_written": len(lstmf_paths),
        "clean_frac": clean_frac,
        "charset_total": len(charset),
        "charset_covered": len(charset & seen_chars),
        "charset_missing": missing,
        "seed": RANDOM_SEED,
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[tesstrain] {shard_name}: wrote {written} pairs, {len(lstmf_paths)} lstmf, "
        f"charset {summary['charset_covered']}/{summary['charset_total']} covered"
    )
    if missing:
        print(f"[tesstrain] WARNING uncovered chars: {missing}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="shard_0001", help="output shard dir name")
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "tesseract" / "lstmf"))
    ap.add_argument("--fonts-dir", default=str(ROOT / "data" / "fonts"))
    ap.add_argument("--english-frac", type=float, default=0.12)
    ap.add_argument("--tessdata-dir", default=str(ROOT / "data" / "tesseract" / "tessdata"))
    ap.add_argument("--lang", default="mlt")
    ap.add_argument("--clean-frac", type=float, default=0.30)
    ap.add_argument("--no-lstmf", action="store_true", help="emit tif/gt/box only, skip .lstmf")
    ap.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="shift corpus stream + per-sample RNG so a later shard carries distinct content",
    )
    args = ap.parse_args()

    export_shard(
        shard_name=args.shard,
        count=args.count,
        out_root=Path(args.out_root),
        fonts_dir=Path(args.fonts_dir),
        english_frac=args.english_frac,
        tessdata_dir=Path(args.tessdata_dir),
        lang=args.lang,
        make_lstmf=not args.no_lstmf,
        clean_frac=args.clean_frac,
        index_offset=args.index_offset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
