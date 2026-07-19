"""Score the Hungarian 5-stream LV-ROVER ensemble on the 200-page eval set,
same method as competition_transcriber.py, only the language changed."""
import json
import unicodedata

import jiwer
from PIL import Image

from hu_transcriber import transcribe

with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

refs, hyps = [], []
for i, item in enumerate(manifest):
    img = Image.open(f"data/eval_pages/{item['file']}")
    hyp = transcribe(img)
    refs.append(unicodedata.normalize("NFC", item["text"]))
    hyps.append(unicodedata.normalize("NFC", hyp))
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(manifest)} scored")

cer = jiwer.cer(refs, hyps)
print(f"pages scored: {len(refs)}")
print(f"5-stream LV-ROVER ensemble Hungarian CER: {cer:.5f}")
with open("results/ours_ensemble.json", "w", encoding="utf-8") as f:
    json.dump({"pages": len(refs), "cer": cer, "streams": 5, "method": "LV-ROVER"}, f, indent=2)
