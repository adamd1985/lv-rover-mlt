"""Build a paragraph-shaped Hungarian text corpus from HuCCPDF, excluding the
pages already reserved for eval. This replaces korpus_malti's role for
Maltese - real text, no streaming/domain-config machinery needed since we
already have real text in hand (per the approved plan's cost note)."""
import json
import re

import pandas as pd

with open("data/eval_manifest.json", encoding="utf-8") as f:
    eval_files = {m["source_file_name"] for m in json.load(f)}

paragraphs = []
for shard in ["data/raw/data/train-00000-of-00056.parquet",
              "data/raw/data/train-00001-of-00056.parquet",
              "data/raw/data/train-00002-of-00056.parquet"]:
    df = pd.read_parquet(shard)
    for _, row in df.iterrows():
        if row["file_name"] in eval_files:
            continue  # keep eval pages held out of training text
        text = row["text"] or ""
        # split on blank-line paragraph breaks; fall back to sentence-ish chunks
        chunks = re.split(r"\n\s*\n", text)
        for c in chunks:
            c = " ".join(c.split())
            if 60 <= len(c) <= 600:
                paragraphs.append(c)

print(f"paragraphs extracted: {len(paragraphs)}")
with open("data/hu_corpus.jsonl", "w", encoding="utf-8") as f:
    for p in paragraphs:
        f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
