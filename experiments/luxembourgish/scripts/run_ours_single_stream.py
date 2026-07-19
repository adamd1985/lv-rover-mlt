"""Score the fine-tuned Luxembourgish Tesseract (single stream, no ensemble)
on the 200 real held-out BnL line crops."""
import json
import unicodedata

import jiwer
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/home/adamd1985/miniforge3/envs/sys/bin/tesseract"

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

refs, hyps = [], []
for item in manifest:
    img = Image.open(f"data/antiqua/antiqua/{item['stem']}.png")
    raw = pytesseract.image_to_string(
        img, lang="ltz", config='--tessdata-dir "tessdata_ltz" --psm 7'
    )
    refs.append(unicodedata.normalize("NFC", item["text"]))
    hyps.append(unicodedata.normalize("NFC", raw))

cer = jiwer.cer(refs, hyps)
print(f"lines scored: {len(refs)}")
print(f"fine-tuned single-stream Luxembourgish CER: {cer:.5f}")
with open("results/ours_single_stream.json", "w", encoding="utf-8") as f:
    json.dump({"lines": len(refs), "cer": cer, "lang": "ltz (fine-tuned)", "config": "--psm 7"}, f, indent=2)
