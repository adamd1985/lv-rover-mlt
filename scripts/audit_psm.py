"""Sweep Tesseract page-segmentation modes for the mlt+ita chain.

PSM 6 is the v11 default. Other modes may handle different layouts:
  4 = single column of variable-sized text
  6 = single uniform block of text (default)
  7 = single text line
  11 = sparse text
  13 = raw line (no scripts)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import PIL.Image
import pytesseract
from jiwer import cer
from malti.line_joiner import RBLineJoiner

TESSDATA = "data/tesseract/tessdata"


def normalise(text): return text.replace("–", "—")


def tess(joiner, img, psm, lang="mlt+ita"):
    cfg = f'--tessdata-dir "{TESSDATA}" --psm {psm}'
    raw = pytesseract.image_to_string(img, lang=lang, config=cfg)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    return normalise(joiner.join_lines(lines, fix_hyphenated_words=True)) if lines else ""


def load_hard(): return [{"image": d["hard_image"], "gold": d["gold"]} for d in json.loads(Path("data/synth_val_hard/meta.json").read_text())]


def load_synth(n=120):
    out = []
    for bucket in ["L0", "L1", "L2", "L3"]:
        p = Path(f"data/mira_pairs/{bucket}/shard_000.jsonl")
        if not p.exists(): continue
        for ln in p.read_text().splitlines()[:n // 4]:
            d = json.loads(ln)
            out.append({"image": str(Path("data/mira_pairs") / bucket / d["image"].split("/", 1)[1]), "gold": d["gold"]})
    return out[:n]


def evaluate(joiner, items, psm):
    refs, hyps = [], []
    t0 = time.time()
    for it in items:
        img = PIL.Image.open(it["image"]).convert("RGB")
        hyps.append(tess(joiner, img, psm))
        refs.append(it["gold"])
    return cer(refs, hyps), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psms", type=int, nargs="+", default=[4, 6, 7, 11])
    args = ap.parse_args()

    joiner = RBLineJoiner()
    for name, items in [("synth_val", load_synth()), ("hard_synth", load_hard())]:
        print(f"\n[{name}]  n={len(items)}")
        for psm in args.psms:
            c, t = evaluate(joiner, items, psm)
            print(f"  PSM {psm:2d}  CER={c:.5f}  wall={t:.1f}s")


if __name__ == "__main__":
    main()
