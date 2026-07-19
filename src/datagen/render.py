"""Render synthetic paragraph or line crops from the corpus.

This file is a thin orchestrator. The heavy lifting belongs to SynthDoG
(paragraph mode) or SynthTIGER (line mode). Install those in scripts/setup.sh
and import their public APIs here.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator


def iter_corpus(corpus_dir: Path) -> Iterator[str]:
    for shard in sorted(corpus_dir.glob("corpus.*.jsonl")):
        with shard.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)["text"]
                except Exception:
                    continue


def render_paragraph(text: str, fonts: list, cfg: dict):
    raise NotImplementedError("wire up SynthDoG paragraph render here")


def render_line(text: str, fonts: list, cfg: dict):
    raise NotImplementedError("wire up SynthTIGER line render here")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["line", "paragraph"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--fonts", default="data/fonts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    fonts = sorted(list(Path(args.fonts).glob("*.ttf")) + list(Path(args.fonts).glob("*.otf")))
    if not fonts:
        print("no fonts available, run `make fonts-check` first")
        return 2

    corpus = iter_corpus(Path(args.corpus))
    done = 0
    fn = render_paragraph if args.mode == "paragraph" else render_line
    for text in corpus:
        if done >= args.samples:
            break
        try:
            fn(text, fonts, cfg)
            done += 1
        except NotImplementedError:
            print("render backend not wired in yet; see TODO in render.py")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
