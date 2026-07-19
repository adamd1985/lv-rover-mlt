"""Score the Luxembourgish 5-stream LV-ROVER ensemble on the 200-line real
eval set, same method as competition_transcriber.py, only the language and
PSM (single-line crops) changed."""
import json
import unicodedata

import jiwer
from PIL import Image

from ltz_transcriber import transcribe

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

refs, hyps = [], []
for i, item in enumerate(manifest):
    img = Image.open(f"data/antiqua/antiqua/{item['stem']}.png")
    hyp = transcribe(img)
    refs.append(unicodedata.normalize("NFC", item["text"]))
    hyps.append(unicodedata.normalize("NFC", hyp))
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(manifest)} scored")

cer = jiwer.cer(refs, hyps)
print(f"lines scored: {len(refs)}")
print(f"5-stream LV-ROVER ensemble Luxembourgish CER: {cer:.5f}")
with open("results/ours_ensemble.json", "w", encoding="utf-8") as f:
    json.dump({"lines": len(refs), "cer": cer, "streams": 5, "method": "LV-ROVER"}, f, indent=2)
