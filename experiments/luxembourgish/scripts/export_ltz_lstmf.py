"""Cut line crops from synthetic Luxembourgish paragraph renders and build
.lstmf training files, reusing tesstrain_export.py's generic box/lstmf logic
unchanged. Supplements the thin real training set (96 lines after eval
holdout) with synthetic data, DPI calibrated by direct measurement against
real BnL crops (see luxembourgish_paragraph.py docstring)."""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/adamd1985/doceng2026")
sys.path.insert(0, "/home/adamd1985/doceng2026/experiments/luxembourgish/scripts")

from src.datagen.tesstrain_export import _write_box, _make_lstmf
from src.datagen.font_loader import FontFace, FontSampler
from src.datagen.augmentations import AugConfig, augment_cpu
from luxembourgish_paragraph import HungarianParagraph, HuLayoutConfig

RANDOM_SEED = 42
N_PARAGRAPHS = 1500

with open("data/ltz_validated_fonts.txt") as f:
    font_paths = [Path(line.strip()) for line in f if line.strip()]
printed = [FontFace(path=p, bucket="printed", family=p.stem) for p in font_paths]
sampler = FontSampler(printed=printed, handwriting=[], handwriting_rate=0.0,
                       rng=random.Random(RANDOM_SEED))
renderer = HungarianParagraph(sampler, HuLayoutConfig(), rng=random.Random(RANDOM_SEED))
aug_rng = random.Random(RANDOM_SEED + 2)
aug_cfg = AugConfig(
    jpeg_q_lo=80, jpeg_q_hi=95,
    blur_sigma_lo=0.0, blur_sigma_hi=0.3,
    rotation_deg=0.5,
    sp_prob=0.002,
    ink_bleed_p=0.0,
)  # light augmentation per the Hungarian lesson (attempt 3) - real target
   # images are historical print, not modern scans, but not heavily degraded
   # either; err toward less synthetic noise, not more.

with open("data/ltz_corpus.jsonl", encoding="utf-8") as f:
    paragraphs = [json.loads(line)["text"] for line in f]
random.Random(RANDOM_SEED + 3).shuffle(paragraphs)
paragraphs = paragraphs[:N_PARAGRAPHS]

tessdata_dir = Path("/home/adamd1985/miniforge3/envs/sys/share/tessdata")
assert (tessdata_dir / "ltz.traineddata").is_file()

out_dir = Path("data/lstm_train")
out_dir.mkdir(parents=True, exist_ok=True)
lstmf_paths = []
n_lines = 0
for pi, text in enumerate(paragraphs):
    try:
        img, printed_lines, bboxes = renderer.render_with_lines(text)
    except Exception:
        continue
    for li, (line_text, bbox) in enumerate(zip(printed_lines, bboxes)):
        if not line_text.strip():
            continue
        x0, y0, x1, y1 = bbox
        crop = img.crop((x0, y0, x1, min(y1, img.height)))
        crop = augment_cpu(crop, aug_cfg, aug_rng)
        base = out_dir / f"{pi:05d}_{li:02d}"
        tif_path = base.with_suffix(".tif")
        crop.convert("L").save(tif_path)
        box_path = base.with_suffix(".box")
        _write_box(box_path, line_text, crop.size)
        lstmf = _make_lstmf(tif_path, box_path, base, tessdata_dir, "ltz", dict(os.environ))
        if lstmf:
            lstmf_paths.append(str(lstmf))
            n_lines += 1
    if (pi + 1) % 200 == 0:
        print(f"{pi+1}/{len(paragraphs)} paragraphs, {n_lines} lstmf so far")

with open("data/ltz_training_files_synth.txt", "w") as f:
    for p in lstmf_paths:
        f.write(p + "\n")
print(f"done: {n_lines} synthetic lstmf files written")
