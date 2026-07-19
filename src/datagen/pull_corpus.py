"""Pull `MLRS/korpus_malti` v4.2 paragraph chunks.

Important: use a domain-split config so sentence order is preserved. The
shuffled config breaks paragraph coherence and ruins paragraph models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAIN_SPLITS = [
    "news",
    "law",
    "literature",
    "religion",
    "academic",
    "parliament",
    "europarl",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--english-frac", type=float, default=0.12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    # Pull domain-split configs, not the shuffled one. Iterate streaming to
    # keep RAM low and dump to JSONL shards.
    shard_idx = 0
    shard_path = out / f"corpus.{shard_idx:04d}.jsonl"
    f = shard_path.open("w", encoding="utf-8")
    n_in_shard = 0
    target_per_shard = 50_000

    for split in DOMAIN_SPLITS:
        try:
            ds = load_dataset("MLRS/korpus_malti", split, split="train", streaming=True)
        except Exception as e:
            print(f"skip {split}: {e}")
            continue
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            text = text.strip()
            if len(text) < 40:
                continue
            f.write(json.dumps({"text": text, "domain": split}, ensure_ascii=False) + "\n")
            n_in_shard += 1
            if n_in_shard >= target_per_shard:
                f.close()
                shard_idx += 1
                if shard_idx >= args.shards:
                    return 0
                shard_path = out / f"corpus.{shard_idx:04d}.jsonl"
                f = shard_path.open("w", encoding="utf-8")
                n_in_shard = 0
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
