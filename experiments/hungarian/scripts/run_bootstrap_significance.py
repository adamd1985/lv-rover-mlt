"""Paired bootstrap CI + permutation test, stock Tesseract hun vs the 5-stream
LV-ROVER ensemble, on the same 200-page Hungarian eval set. Closes the
statistical-rigor gap the peer review flagged (0.7% margin,
no test run)."""
import json
import sys
import unicodedata
from pathlib import Path

import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from bootstrap_stats import bootstrap_ci, edit_distance, permutation_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hu_transcriber import transcribe

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

per_item = []
for i, item in enumerate(manifest):
    img = Image.open(f"data/eval_pages/{item['file']}")
    ref = unicodedata.normalize("NFC", item["text"])
    stock_hyp = unicodedata.normalize(
        "NFC", pytesseract.image_to_string(img, lang="hun", config="--psm 3")
    )
    ens_hyp = unicodedata.normalize("NFC", transcribe(img))
    per_item.append((
        edit_distance(ref, stock_hyp), len(ref),
        edit_distance(ref, ens_hyp), len(ref),
    ))
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(manifest)} scored")

cer_stock, cer_ens, delta, lo, hi = bootstrap_ci(per_item, seed=42)
print(f"stock CER={cer_stock:.5f}, ensemble CER={cer_ens:.5f}, delta={delta:.5f}")
print(f"95% CI of delta (stock minus ensemble): [{lo:.5f}, {hi:.5f}]")
excludes_zero = lo > 0 or hi < 0
print(f"excludes zero: {excludes_zero}")

p_value, critical_value = permutation_test(per_item, delta, seed=43)
significant = p_value < 0.05
print(f"permutation test: p={p_value:.5f}, critical value (alpha=0.05)={critical_value:.5f}")
print(f"significant at alpha=0.05: {significant}")

with open("results/bootstrap_significance.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_pages": len(per_item),
        "cer_stock": cer_stock,
        "cer_ensemble": cer_ens,
        "point_delta": delta,
        "ci_95_lo": lo,
        "ci_95_hi": hi,
        "excludes_zero": excludes_zero,
        "permutation_p_value": p_value,
        "permutation_critical_value_alpha05": critical_value,
        "significant_alpha05": significant,
        "seed": 42,
    }, f, indent=2)
print("wrote results/bootstrap_significance.json")
