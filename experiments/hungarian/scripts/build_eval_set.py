"""Extract a held-out Hungarian eval sample from HuCCPDF shard 0, filtered to
prose-like pages (excluding tables/forms/technical spec sheets that a single-PSM
Tesseract pass handles poorly and that don't match our paragraph-prose task shape).

Mirrors the role of the 422-paragraph Maltese dev set, but at page granularity:
HuCCPDF ships whole-page images with page-level ground truth text, not
paragraph-cropped images with paragraph bounding boxes, so this is a
page-level comparison, not a like-for-like paragraph one.
"""
import io
import json
import random
import re

import pandas as pd
from PIL import Image

RANDOM_SEED = 42
N_EVAL = 200
N_CANDIDATES = 1200  # oversample pool to filter down from


def prose_score(text: str) -> float:
    """Heuristic prose-likeness in [0, 1]-ish; higher = more likely flowing prose.
    Penalizes high digit density and short average line length (typical of
    tables/forms/technical spec sheets extracted line-by-line)."""
    if not text or len(text.strip()) < 200:
        return -1.0
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return -1.0
    alpha = sum(c.isalpha() for c in non_ws)
    digit = sum(c.isdigit() for c in non_ws)
    alpha_frac = alpha / len(non_ws)
    digit_frac = digit / len(non_ws)

    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
    if not lines:
        return -1.0
    avg_line_len = sum(len(ln) for ln in lines) / len(lines)

    score = alpha_frac - 2.0 * digit_frac
    if avg_line_len < 40:
        score -= 0.3
    return score


df = pd.read_parquet("data/raw/data/train-00000-of-00056.parquet")
random.seed(RANDOM_SEED)
candidate_idx = random.sample(range(len(df)), min(N_CANDIDATES, len(df)))

scored = []
for i in candidate_idx:
    text = df.iloc[i]["text"]
    s = prose_score(text)
    if s > 0.55:
        scored.append((s, i))

scored.sort(reverse=True)
chosen = [i for _, i in scored[:N_EVAL]]
print(f"candidates screened: {len(candidate_idx)}, passed prose filter: {len(scored)}, "
      f"selected: {len(chosen)}")

manifest = []
for n, i in enumerate(chosen):
    row = df.iloc[i]
    text = row["text"]
    img = Image.open(io.BytesIO(row["image"]["bytes"]))
    fname = f"page_{n:04d}.png"
    img.save(f"data/eval_pages/{fname}")
    manifest.append({"file": fname, "text": text, "source_file_name": row["file_name"]})

with open("data/eval_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"eval set: {len(manifest)} prose-filtered pages written to data/eval_pages/")
