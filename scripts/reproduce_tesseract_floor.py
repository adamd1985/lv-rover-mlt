"""Reproduce the stock-Tesseract submission floor on the real dev set.

Runs stock `mlt.traineddata` (tessdata_best) plus the malti rule-based line
joiner over all 422 competition_files/dev images, using the exact transcribe
logic of competition_transcriber.py (PSM 6, RBLineJoiner, en-dash -> em-dash
normalisation). Scores with cer_organiser, the leaderboard-faithful CER.

baselines.docx target: mlt-best + malti joiner = CER 0.02387. This number is
the submission floor and the regression anchor for fine-tuning - it must
reproduce before any fine-tuned checkpoint is scored.

The dev tessdata dir defaults to data/tesseract/tessdata (populated by the
setup step). Override with --tessdata-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import List

import PIL.Image
import pytesseract
from malti.line_joiner import RBLineJoiner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.cer import cer_organiser

_EN_DASH = "–"
_EM_DASH = "—"
_SOFT_HYPHEN = "­"

FLOOR_TARGET = 0.02387


def _normalise(text: str) -> str:
    text = text.replace(_SOFT_HYPHEN, "").replace(_EN_DASH, _EM_DASH)
    return unicodedata.normalize("NFC", text)


def transcribe(image: PIL.Image.Image, joiner: RBLineJoiner, config: str, lang: str) -> str:
    # convert("RGB") forces a full decode. Tesseract output is sensitive to
    # whether the image is materialised vs left as a lazy JPEG handle, so the
    # deliverable's convert() path is the only faithful one - match it here.
    image = image.convert("RGB")
    raw = pytesseract.image_to_string(image, lang=lang, config=config)
    lines: List[str] = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    joined = joiner.join_lines(lines, fix_hyphenated_words=True)
    return _normalise(joined)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", default=str(ROOT / "competition_files" / "dev"))
    ap.add_argument("--tessdata-dir", default=str(ROOT / "data" / "tesseract" / "tessdata"))
    ap.add_argument("--lang", default="mlt")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap images for a smoke run")
    ap.add_argument("--report", default=str(ROOT / "outputs" / "tesseract" / "floor.json"))
    args = ap.parse_args()

    dev_dir = Path(args.dev_dir)
    tessdata_dir = Path(args.tessdata_dir)
    if not (tessdata_dir / f"{args.lang}.traineddata").is_file():
        print(f"[floor] missing {args.lang}.traineddata under {tessdata_dir}", file=sys.stderr)
        return 2

    with open(dev_dir / "texts.json", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    config = f'--tessdata-dir "{tessdata_dir}" --psm {args.psm}'
    joiner = RBLineJoiner()

    refs: List[str] = []
    hyps: List[str] = []
    t0 = time.perf_counter()
    for i, doc in enumerate(data):
        img = PIL.Image.open(dev_dir / doc["image"])
        hyps.append(transcribe(img, joiner, config, args.lang))
        refs.append(doc["text"])
        if (i + 1) % 50 == 0:
            print(f"[floor] {i + 1}/{len(data)}")
    wall = time.perf_counter() - t0

    cer = cer_organiser(refs, hyps)
    delta = cer - FLOOR_TARGET
    # Pass if we match the baseline within tolerance OR beat it. A worse number
    # means the config/joiner/tessdata path drifted from the deliverable.
    ok = delta <= 0.002
    print(f"[floor] images={len(data)} wall={wall:.1f}s")
    print(f"[floor] CER={cer:.5f} target={FLOOR_TARGET:.5f} delta={delta:+.5f}")
    if ok:
        print("[floor] floor confirmed (matches or beats the baseline)")
    else:
        print("[floor] REGRESSION - CER above baseline, debug PSM/config/joiner")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "cer": cer,
                "target": FLOOR_TARGET,
                "delta": delta,
                "n_images": len(data),
                "wall_s": round(wall, 2),
                "lang": args.lang,
                "psm": args.psm,
                "tessdata_dir": str(tessdata_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
