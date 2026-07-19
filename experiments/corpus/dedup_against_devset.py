"""Contamination guard: check corpus paragraph pairs against the competition dev set.

Two independent checks, per the plan's dedup requirement:
1. Text n-gram overlap: any corpus paragraph whose normalized text is a
   near-duplicate of a dev-set text (word-level Jaccard over 5-grams) is
   flagged. Catches exact and near-exact text leakage.
2. Perceptual image hash: any corpus crop whose image is near-identical to
   a dev crop is flagged. Catches image leakage even if text differs (e.g.
   same document photographed differently) - unlikely for this corpus
   (different sources entirely) but cheap to check and honest to report.

This does not delete anything - it reports matches above threshold so a
human can review before any HF publish. Designed to run over the
`paragraphs/manifest.json` files produced by align_pdf_paragraphs.py.

Usage:
    python dedup_against_devset.py --corpus-root data/wikipedia_mt_batch1 \
        --dev-texts competition_files/dev/texts.json \
        --dev-images competition_files/dev
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

NGRAM_N = 5
JACCARD_THRESHOLD = 0.5
PHASH_HAMMING_THRESHOLD = 4  # out of 64 bits; <=4 is a strong near-duplicate signal for dHash


def normalize(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return s.split()


def ngrams(words: list[str], n: int) -> set[tuple]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (dHash) - compares adjacent-pixel gradients.

    Average-hash (aHash) was tried first and rejected: these are thin
    text-line crops, mostly white background, so aHash's mean-brightness
    threshold produces near-identical bit patterns across *unrelated*
    documents (807/39668 false-positive "duplicates" on a corpus with zero
    real overlap with the dev set - caught by that implausible count, not
    assumed). dHash compares each pixel to its neighbor rather than to a
    global mean, which is far less sensitive to a shared white background.
    """
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = 0
    for row in range(hash_size):
        row_pixels = pixels[row * (hash_size + 1):(row + 1) * (hash_size + 1)]
        for col in range(hash_size):
            bits = (bits << 1) | (1 if row_pixels[col] > row_pixels[col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_dev(dev_texts_path: Path, dev_images_dir: Path) -> tuple[list[dict], list[int]]:
    dev = json.loads(dev_texts_path.read_text(encoding="utf-8"))
    dev_ngram_sets = [ngrams(normalize(d["text"]), NGRAM_N) for d in dev]
    dev_phashes = []
    for d in dev:
        img_path = dev_images_dir / d["image"]
        if img_path.exists():
            dev_phashes.append(phash(Image.open(img_path)))
        else:
            dev_phashes.append(None)
    return [{"text": d["text"], "ngrams": ng} for d, ng in zip(dev, dev_ngram_sets)], dev_phashes


def scan_corpus(corpus_root: Path, dev_entries: list[dict], dev_phashes: list) -> dict:
    manifests = list(corpus_root.rglob("manifest.json"))
    total_pairs = 0
    text_flags = []
    image_flags = []

    for mf in manifests:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        for pair in manifest.get("pairs", []):
            total_pairs += 1
            words = normalize(pair["text"])
            pg = ngrams(words, NGRAM_N)
            for i, dev_e in enumerate(dev_entries):
                score = jaccard(pg, dev_e["ngrams"])
                if score >= JACCARD_THRESHOLD:
                    text_flags.append({
                        "manifest": str(mf), "crop": pair["crop"], "text": pair["text"][:120],
                        "dev_index": i, "dev_text": dev_e["text"][:120], "jaccard": round(score, 3),
                    })

            crop_path = mf.parent / pair["crop"]
            if crop_path.exists():
                try:
                    h = phash(Image.open(crop_path))
                except Exception:
                    h = None
                if h is not None:
                    for i, dp in enumerate(dev_phashes):
                        if dp is None:
                            continue
                        d = hamming(h, dp)
                        if d <= PHASH_HAMMING_THRESHOLD:
                            image_flags.append({
                                "manifest": str(mf), "crop": pair["crop"],
                                "dev_index": i, "hamming_distance": d,
                            })

    return {
        "total_pairs_scanned": total_pairs,
        "text_flags": text_flags,
        "image_flags": image_flags,
        "n_text_flags": len(text_flags),
        "n_image_flags": len(image_flags),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True, help="dir to recursively search for paragraphs/manifest.json files")
    ap.add_argument("--dev-texts", default="competition_files/dev/texts.json")
    ap.add_argument("--dev-images", default="competition_files/dev")
    ap.add_argument("--out", default=None, help="output report JSON path (default: <corpus-root>/contamination_report.json)")
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    dev_entries, dev_phashes = load_dev(Path(args.dev_texts), Path(args.dev_images))
    report = scan_corpus(corpus_root, dev_entries, dev_phashes)

    out_path = Path(args.out) if args.out else corpus_root / "contamination_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[dedup] scanned {report['total_pairs_scanned']} corpus pairs against {len(dev_entries)} dev entries")
    print(f"[dedup] text-overlap flags (jaccard>={JACCARD_THRESHOLD}): {report['n_text_flags']}")
    print(f"[dedup] image-overlap flags (hamming<={PHASH_HAMMING_THRESHOLD}): {report['n_image_flags']}")
    print(f"[dedup] report: {out_path}")


if __name__ == "__main__":
    main()
