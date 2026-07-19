"""Generate a synthetic Hungarian training shard: same method as Maltese
(paragraph render -> augment -> write image+label pairs), reusing the
generic FontSampler/augment_cpu primitives, only the layout/hyphenation
logic is simplified (no clitic tagging, Hungarian doesn't need it)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/adamd1985/doceng2026")
sys.path.insert(0, "/home/adamd1985/doceng2026/experiments/hungarian/scripts")

from src.datagen.augmentations import AugConfig, augment_cpu
from src.datagen.font_loader import FontFace, FontSampler
from hungarian_paragraph import HungarianParagraph, HuLayoutConfig

RANDOM_SEED = 42
N_SAMPLES = 3000

with open("data/hu_validated_fonts.txt") as f:
    font_paths = [Path(line.strip()) for line in f if line.strip()]
printed = [FontFace(path=p, bucket="printed", family=p.stem) for p in font_paths]
sampler = FontSampler(printed=printed, handwriting=[], handwriting_rate=0.0,
                       rng=random.Random(RANDOM_SEED))

renderer = HungarianParagraph(sampler, HuLayoutConfig(), rng=random.Random(RANDOM_SEED))
aug_rng = random.Random(RANDOM_SEED + 1)
aug_cfg = AugConfig()

paragraphs = []
with open("data/hu_corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        paragraphs.append(json.loads(line)["text"])
random.Random(RANDOM_SEED).shuffle(paragraphs)
paragraphs = paragraphs[:N_SAMPLES]

out_dir = Path("data/synth_shard")
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []
for i, text in enumerate(paragraphs):
    try:
        img, label = renderer.render_one(text)
        img = augment_cpu(img, aug_cfg, aug_rng)
    except Exception as e:
        continue
    fname = f"{i:06d}.jpg"
    img.save(out_dir / fname, quality=85)
    manifest.append({"file": fname, "label": label})
    if (i + 1) % 500 == 0:
        print(f"{i+1}/{len(paragraphs)} rendered")

with open("data/synth_shard_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"done: {len(manifest)} samples in {out_dir}")
