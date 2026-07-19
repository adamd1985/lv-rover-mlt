"""Audit Tesseract language-chain anchors across synth val + hard synth val + dev.

Compares standalone Tesseract chains (with malti joiner only, no corrector or
router) to v11's full stack. Selection metric: must win or tie on all three
distributions to be eligible for v12 promotion.

Usage:
    PYTHONPATH=. python scripts/audit_anchor_chains.py --chains mlt mlt+ita mlt+ita+fra
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import PIL.Image
import pytesseract
from jiwer import cer
from malti.line_joiner import RBLineJoiner


TESSDATA = "data/tesseract/tessdata"
PSM_CONFIG = f'--tessdata-dir "{TESSDATA}" --psm 6'
FALLBACK_RATIO = 0.6


def normalise(text: str) -> str:
    return text.replace("–", "—")


def tess_paragraph(joiner: RBLineJoiner, image: PIL.Image.Image, lang: str) -> str:
    raw = pytesseract.image_to_string(image, lang=lang, config=PSM_CONFIG)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    return normalise(joiner.join_lines(lines, fix_hyphenated_words=True))


def run_chain_with_safety(joiner: RBLineJoiner, image: PIL.Image.Image,
                           anchor_lang: str, fallback_lang: str = "mlt") -> str:
    """Tesseract with v11's length-ratio safety fallback."""
    primary = tess_paragraph(joiner, image, anchor_lang)
    if anchor_lang == fallback_lang:
        return primary
    fb = tess_paragraph(joiner, image, fallback_lang)
    if len(fb) > 10 and len(primary) < FALLBACK_RATIO * len(fb):
        return fb
    return primary


def eval_dataset(joiner: RBLineJoiner, items: List[dict], anchor_lang: str,
                 fallback_lang: str, image_root: Path) -> Dict[str, float]:
    refs, hyps = [], []
    t0 = time.time()
    for it in items:
        img_path = image_root / it["image"] if not it.get("hard_image") else Path(it["hard_image"])
        img = PIL.Image.open(img_path).convert("RGB")
        pred = run_chain_with_safety(joiner, img, anchor_lang, fallback_lang)
        refs.append(it["gold"])
        hyps.append(pred)
    return {"cer": cer(refs, hyps), "n": len(refs), "wall_s": time.time() - t0}


def load_hard_synth(path="data/synth_val_hard/meta.json") -> List[dict]:
    data = json.loads(Path(path).read_text())
    out = []
    for d in data:
        out.append({"image": d["hard_image"], "gold": d["gold"], "hard_image": d["hard_image"]})
    return out


def load_synth_val(n: int = 120) -> List[dict]:
    out = []
    for bucket in ["L0", "L1", "L2", "L3"]:
        p = Path(f"data/mira_pairs/{bucket}/shard_000.jsonl")
        if not p.exists():
            continue
        for ln in p.read_text().splitlines()[:n // 4]:
            d = json.loads(ln)
            out.append({"image": str(Path("data/mira_pairs") / bucket / d["image"].split("/", 1)[1]),
                        "gold": d["gold"]})
    return out[:n]


def load_dev() -> List[dict]:
    out = []
    gold = {json.loads(l)["id"]: json.loads(l)["gold"]
            for l in Path("outputs/campaign/dev_gold.jsonl").read_text().splitlines()}
    for k in sorted(gold):
        out.append({"image": f"competition_files/dev/{k}", "gold": gold[k]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", required=True)
    ap.add_argument("--fallback", default="mlt")
    ap.add_argument("--sets", nargs="+", default=["synth_val", "hard", "dev"])
    ap.add_argument("--limit", type=int, default=0, help="cap items per set for smoke")
    args = ap.parse_args()

    joiner = RBLineJoiner()
    datasets = {}
    if "synth_val" in args.sets:
        datasets["synth_val"] = load_synth_val(120)
    if "hard" in args.sets:
        datasets["hard_synth"] = load_hard_synth()
    if "dev" in args.sets:
        datasets["dev"] = load_dev()

    if args.limit:
        datasets = {k: v[: args.limit] for k, v in datasets.items()}

    print(f'config: PSM_CONFIG={PSM_CONFIG}, fallback={args.fallback}')
    for name, items in datasets.items():
        print(f"\n[{name}]  n={len(items)}")
        for chain in args.chains:
            res = eval_dataset(joiner, items, chain, args.fallback,
                                image_root=Path("."))
            print(f"  {chain:18s}  CER={res['cer']:.5f}  wall={res['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
