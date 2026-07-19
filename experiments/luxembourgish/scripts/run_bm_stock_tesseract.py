"""BM (benchmark/reference): stock Tesseract ltz, single stream, no ensemble,
no post-processing, no fine-tuning. Real BnL line crops, --psm 7 (single
line) since these are pre-cropped lines, not whole pages (Hungarian) or
paragraphs (Maltese) - verified against real crop dimensions before use."""
import json
import unicodedata

import jiwer
import pytesseract
from PIL import Image

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

refs, hyps = [], []
for item in manifest:
    img = Image.open(f"data/antiqua/antiqua/{item['stem']}.png")
    raw = pytesseract.image_to_string(img, lang="ltz", config="--psm 7")
    refs.append(unicodedata.normalize("NFC", item["text"]))
    hyps.append(unicodedata.normalize("NFC", raw))

cer = jiwer.cer(refs, hyps)
print(f"lines scored: {len(refs)}")
print(f"stock Tesseract ltz CER: {cer:.5f}")
with open("results/bm_stock_tesseract.json", "w", encoding="utf-8") as f:
    json.dump({"lines": len(refs), "cer": cer, "lang": "ltz", "config": "--psm 7"}, f, indent=2)
