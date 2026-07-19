"""Shard generator for the v0 paragraph synth pipeline.

Writes (NNNNNN.{png|jpg}, NNNNNN.json) pairs under --out. The JSON carries
{label, label_parts, font, augmentations_applied, seed}. Reproducible from
--seed.

Drift control: this entrypoint is intentionally CPU-only and single-process for
v0 (writes ~10k samples in well under an hour on the dev box). Worker pools land
in a later PR once disk throughput and font cache behaviour are characterised.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datagen.augmentations import AugConfig, augment_cpu
from src.datagen.corpus_loader import iter_paragraphs
from src.datagen.font_loader import load_fonts
from src.datagen.maltese_paragraph import LayoutConfig, MalteseParagraph, SOFT_HYPHEN


# Canary set: Maltese diacritics plus the grave-accented vowels that actually
# occur in the gold inventory (lowercase à ì ò ù only; see competition_files/
# char_set.json). No uppercase graves, no è.
CANARY_CHARS = list("ĊċĠġĦħŻżàìòù")
# Label-bearing dashes: ASCII hyphen and em-dash only. En-dash U+2013 is an
# image-only glyph and must never appear in a label.
LABEL_DASHES = list("-—")
EN_DASH = "–"


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _save_image(img, out_dir: Path, idx: int, cfg: dict) -> Tuple[str, int]:
    if cfg["shard"].get("use_jpeg", True):
        name = f"{idx:06d}.jpg"
        path = out_dir / name
        img.convert("RGB").save(path, format="JPEG", quality=int(cfg["shard"]["jpeg_quality"]), optimize=True)
    else:
        name = f"{idx:06d}.png"
        path = out_dir / name
        img.save(path, format="PNG", optimize=True)
    return name, path.stat().st_size


def _summarise_aug(cfg: dict) -> List[str]:
    # Static manifest of augmentations enabled. The augment_cpu pipeline applies
    # them probabilistically; we record the menu plus the per-sample rng seed so
    # exact replay is possible from (seed, idx).
    menu = ["rotation", "blur", "brightness_contrast", "salt_pepper", "elastic", "jpeg_recompress"]
    if cfg["augmentations"]["ink_bleed_p"] > 0:
        menu.append("ink_bleed?")
    if cfg["augmentations"]["column_edge_crop_p"] > 0:
        menu.append("column_edge_crop?")
    return menu


def generate(cfg_path: Path, out_dir: Path, count: int) -> dict:
    cfg = _load_cfg(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["shard"].get("seed", 1337))
    rng = random.Random(seed)

    fonts = load_fonts(
        Path(cfg["fonts"]["dir"]),
        handwriting_rate=cfg["fonts"]["handwriting_rate"],
        rng=rng,
        fallback_system=cfg["fonts"]["fallback_system"],
    )
    if len(fonts) == 0:
        raise RuntimeError("no validated fonts under data/fonts/")
    print(f"[gen] fonts: printed={len(fonts.printed)} handwriting={len(fonts.handwriting)}")

    layout = LayoutConfig(**cfg["layout"])
    pipe = MalteseParagraph(font_sampler=fonts, layout=layout, rng=rng)
    aug = AugConfig(**cfg["augmentations"])
    aug_menu = _summarise_aug(cfg)

    corpus = iter_paragraphs(
        english_frac=cfg["corpus"]["english_frac"],
        use_streaming=cfg["corpus"]["use_streaming"],
        rng=rng,
    )

    font_counter: Counter = Counter()
    canary_counts: Counter = Counter()
    dash_counts: Counter = Counter()
    n_soft_hyphens = 0
    n_with_article = 0
    total_label_chars = 0
    total_bytes = 0
    n_lines_total = 0
    t0 = time.perf_counter()
    last_log = t0

    for idx in range(count):
        try:
            row = next(corpus)
        except StopIteration:
            print(f"[gen] corpus exhausted at idx={idx}")
            break

        sample_rng = random.Random(seed * 1_000_003 + idx)

        try:
            img, label_str, label_parts, meta = pipe.render_one(row)
        except Exception as exc:
            print(f"[gen] render failure idx={idx}: {exc}")
            continue

        try:
            img = augment_cpu(img, aug, sample_rng)
        except Exception as exc:
            print(f"[gen] augment failure idx={idx}: {exc}")
            continue

        name, nbytes = _save_image(img, out_dir, idx, cfg)
        total_bytes += nbytes

        # Stats
        font_counter[meta.font_family] += 1
        for c in CANARY_CHARS:
            if c in label_str:
                canary_counts[c] += label_str.count(c)
        if EN_DASH in label_str:
            raise RuntimeError(f"en-dash U+2013 leaked into label at idx={idx}; labels must use U+2014")
        for d in LABEL_DASHES:
            if d in label_str:
                dash_counts[d] += label_str.count(d)
        n_soft_hyphens += meta.n_soft_hyphens
        if meta.has_article_prefix:
            n_with_article += 1
        total_label_chars += len(label_str)
        n_lines_total += len(label_parts)

        rec = {
            "label": label_str,
            "label_parts": label_parts,
            "font": {
                "family": meta.font_family,
                "bucket": meta.font_bucket,
                "pt": meta.font_pt,
            },
            "layout": {
                "width_px": meta.width_px,
                "height_px": meta.height_px,
                "line_spacing": round(meta.line_spacing, 3),
                "justified": meta.justified,
                "leading_bullet": meta.leading_bullet,
                "n_lines": meta.n_lines,
                "n_soft_hyphens": meta.n_soft_hyphens,
            },
            "lang": meta.lang,
            "augmentations_applied": aug_menu,
            "seed": seed,
            "sample_seed": seed * 1_000_003 + idx,
            "image": name,
        }
        json_path = out_dir / f"{idx:06d}.json"
        json_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        if time.perf_counter() - last_log > 5.0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            mb = total_bytes / 1e6
            print(f"[gen] {idx+1}/{count} at {rate:.1f}/s, {mb:.1f} MB on disk")
            last_log = time.perf_counter()

    wall = time.perf_counter() - t0
    written = sum(1 for _ in out_dir.glob("*.json"))

    summary = {
        "out_dir": str(out_dir),
        "count": written,
        "wall_s": round(wall, 2),
        "rate_per_s": round(written / max(wall, 1e-6), 2),
        "disk_bytes": total_bytes,
        "disk_mb": round(total_bytes / 1e6, 2),
        "n_lines_total": n_lines_total,
        "n_soft_hyphens": n_soft_hyphens,
        "n_with_article_prefix": n_with_article,
        "total_label_chars": total_label_chars,
        "canary_char_counts": dict(canary_counts),
        "dash_counts": dict(dash_counts),
        "font_counts": dict(font_counter.most_common()),
        "font_count_min": min(font_counter.values()) if font_counter else 0,
        "font_count_median": sorted(font_counter.values())[len(font_counter) // 2] if font_counter else 0,
        "font_count_max": max(font_counter.values()) if font_counter else 0,
        "n_fonts_used": len(font_counter),
        "seed": seed,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] wrote {written} samples to {out_dir} in {wall:.1f}s, "
          f"{summary['disk_mb']} MB on disk")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True)
    args = ap.parse_args()
    if args.count > 10_000:
        print(f"[gen] refusing count={args.count}, v0 cap is 10000")
        return 2
    generate(Path(args.config), Path(args.out), args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
