"""BM (benchmark/reference): stock Tesseract `hun`, single stream, no ensemble,
no post-processing, no fine-tuning. Mirrors how the Maltese organiser baseline
was first established. Self-measured reference CER since no published Hungarian
printed-OCR CER exists to cite.
"""
import json
import unicodedata

import jiwer
import pytesseract
from PIL import Image

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

refs, hyps = [], []
for item in manifest:
    img = Image.open(f"data/eval_pages/{item['file']}")
    raw = pytesseract.image_to_string(img, lang="hun", config="--psm 3")
    ref = unicodedata.normalize("NFC", item["text"])
    hyp = unicodedata.normalize("NFC", raw)
    refs.append(ref)
    hyps.append(hyp)

# same sum-of-edit-distances-over-sum-of-lengths CER as the Maltese recipe
cer = jiwer.cer(refs, hyps)
print(f"pages scored: {len(refs)}")
print(f"stock Tesseract hun CER: {cer:.5f}")

with open("results/bm_stock_tesseract.json", "w", encoding="utf-8") as f:
    json.dump({"pages": len(refs), "cer": cer, "lang": "hun", "config": "--psm 3"}, f, indent=2)
