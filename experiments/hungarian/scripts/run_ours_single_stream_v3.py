"""Score the v2 (DPI-corrected) fine-tuned Hungarian Tesseract on the same
200-page eval set."""
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
    img = Image.open(f"data/eval_pages/{item['file']}")
    raw = pytesseract.image_to_string(
        img, lang="hun",
        config='--tessdata-dir "tessdata_hu" --psm 3',
    )
    refs.append(unicodedata.normalize("NFC", item["text"]))
    hyps.append(unicodedata.normalize("NFC", raw))

cer = jiwer.cer(refs, hyps)
print(f"pages scored: {len(refs)}")
print(f"fine-tuned v3 (DPI-corrected) single-stream Hungarian CER: {cer:.5f}")
with open("results/ours_single_stream_v3.json", "w", encoding="utf-8") as f:
    json.dump({"pages": len(refs), "cer": cer, "lang": "hun (fine-tuned v3)", "config": "--psm 3"}, f, indent=2)
