"""Audit image preprocessing variants on top of Tesseract mlt+ita.

Tesseract LSTM is known to benefit from larger glyphs and sharper edges.
Sweep upscale factors + light denoise + sharpening on a small synth val
sample first; promote winners to full audit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import PIL.Image
import PIL.ImageFilter
import pytesseract
from jiwer import cer
from malti.line_joiner import RBLineJoiner

TESSDATA = "data/tesseract/tessdata"
CFG = f'--tessdata-dir "{TESSDATA}" --psm 6'


def normalise(text: str) -> str:
    return text.replace("–", "—")


def tess(joiner, img, lang="mlt+ita"):
    raw = pytesseract.image_to_string(img, lang=lang, config=CFG)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    return normalise(joiner.join_lines(lines, fix_hyphenated_words=True)) if lines else ""


def variant_identity(img): return img


def variant_2x(img):
    w, h = img.size
    return img.resize((w * 2, h * 2), PIL.Image.LANCZOS)


def variant_15x(img):
    w, h = img.size
    return img.resize((int(w * 1.5), int(h * 1.5)), PIL.Image.LANCZOS)


def variant_2x_sharp(img):
    return variant_2x(img).filter(PIL.ImageFilter.UnsharpMask(radius=1.2, percent=80))


def variant_sharp(img):
    return img.filter(PIL.ImageFilter.UnsharpMask(radius=1.0, percent=60))


def variant_2x_smooth(img):
    return variant_2x(img).filter(PIL.ImageFilter.SMOOTH)


VARIANTS: Dict[str, Callable] = {
    "identity": variant_identity,
    "1.5x": variant_15x,
    "2x": variant_2x,
    "2x+sharp": variant_2x_sharp,
    "sharp": variant_sharp,
    "2x+smooth": variant_2x_smooth,
}


def load_hard_synth():
    data = json.loads(Path("data/synth_val_hard/meta.json").read_text())
    return [{"image": d["hard_image"], "gold": d["gold"]} for d in data]


def load_synth_val(n=120):
    out = []
    for bucket in ["L0", "L1", "L2", "L3"]:
        p = Path(f"data/mira_pairs/{bucket}/shard_000.jsonl")
        if not p.exists(): continue
        for ln in p.read_text().splitlines()[:n // 4]:
            d = json.loads(ln)
            out.append({"image": str(Path("data/mira_pairs") / bucket / d["image"].split("/", 1)[1]),
                        "gold": d["gold"]})
    return out[:n]


def load_dev():
    gold = {json.loads(l)["id"]: json.loads(l)["gold"]
            for l in Path("outputs/campaign/dev_gold.jsonl").read_text().splitlines()}
    return [{"image": f"competition_files/dev/{k}", "gold": gold[k]} for k in sorted(gold)]


def eval_variant(joiner, items, fn, lang="mlt+ita"):
    refs, hyps = [], []
    t0 = time.time()
    for it in items:
        img = PIL.Image.open(it["image"]).convert("RGB")
        img = fn(img)
        hyps.append(tess(joiner, img, lang))
        refs.append(it["gold"])
    return cer(refs, hyps), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--sets", nargs="+", default=["synth_val", "hard"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    joiner = RBLineJoiner()
    sets = {}
    if "synth_val" in args.sets:
        sets["synth_val"] = load_synth_val()
    if "hard" in args.sets:
        sets["hard_synth"] = load_hard_synth()
    if "dev" in args.sets:
        sets["dev"] = load_dev()
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}

    for name, items in sets.items():
        print(f"\n[{name}]  n={len(items)}")
        for v in args.variants:
            c, t = eval_variant(joiner, items, VARIANTS[v])
            print(f"  {v:12s}  CER={c:.5f}  wall={t:.1f}s")


if __name__ == "__main__":
    main()
