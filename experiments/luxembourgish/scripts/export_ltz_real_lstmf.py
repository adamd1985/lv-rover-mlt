"""Convert the 96 real Luxembourgish training line pairs (correctly-scaled
BnL crops, no synthetic rendering needed) into .lstmf files, mirroring the
synthetic export's box/lstmf logic for consistency."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/adamd1985/doceng2026")
from src.datagen.tesstrain_export import _write_box, _make_lstmf
from PIL import Image

with open("data/train_manifest.json", encoding="utf-8") as f:
    train = json.load(f)

tessdata_dir = Path("/home/adamd1985/miniforge3/envs/sys/share/tessdata")
out_dir = Path("data/lstm_train_real")
out_dir.mkdir(parents=True, exist_ok=True)

lstmf_paths = []
for item in train:
    stem, text = item["stem"], item["text"]
    src_png = Path(f"data/antiqua/antiqua/{stem}.png")
    img = Image.open(src_png).convert("L")
    tif_path = out_dir / f"{stem}.tif"
    img.save(tif_path)
    box_path = out_dir / f"{stem}.box"
    _write_box(box_path, text, img.size)
    lstmf = _make_lstmf(tif_path, box_path, out_dir / stem, tessdata_dir, "ltz", dict(os.environ))
    if lstmf:
        lstmf_paths.append(str(lstmf))

with open("data/ltz_training_files_real.txt", "w") as f:
    for p in lstmf_paths:
        f.write(p + "\n")
print(f"real lstmf written: {len(lstmf_paths)} / {len(train)}")
