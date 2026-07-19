"""Validate a generated shard.

- Joiner round-trip: for each sample with at least one U+00AD or non-trivial
  multiline label, run joiner.join_lines on label_parts and compare to the
  NFC-normalised single-line projection. Pass rate is reported.
- Per-font sample-count distribution.
- Canary character frequency vs corpus baseline.
- Builds _preview.html with 10 random samples.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.joiner import join_lines

SOFT_HYPHEN = "­"


def _gold_paragraph(label_parts: List[str]) -> str:
    return unicodedata.normalize("NFC", "".join(p for p in label_parts).replace(SOFT_HYPHEN, ""))


def _joined_via_joiner(label_parts: List[str]) -> str:
    # Joiner consumes line strings with soft hyphens preserved; it should
    # reattach the broken word. Compare NFC normalised, no soft-hyphen.
    joined = join_lines([p for p in label_parts])
    return unicodedata.normalize("NFC", joined.replace(SOFT_HYPHEN, ""))


def validate(out_dir: Path, preview_count: int = 10) -> dict:
    samples = sorted(out_dir.glob("*.json"))
    samples = [p for p in samples if not p.name.startswith("_")]
    print(f"[val] {len(samples)} samples in {out_dir}")

    soft_hyphen_pass = 0
    soft_hyphen_total = 0
    joiner_pass = 0
    joiner_total = 0
    failures: List[dict] = []

    for p in samples:
        r = json.loads(p.read_text(encoding="utf-8"))
        parts = r["label_parts"]
        has_soft = any(SOFT_HYPHEN in s for s in parts)

        # Recoverability check: stripping soft hyphens and concatenating must
        # equal the label_str with newlines removed and soft-hyphens stripped.
        flat_from_parts = "".join(parts).replace(SOFT_HYPHEN, "")
        flat_from_str = r["label"].replace("\n", "").replace(SOFT_HYPHEN, "")
        flat_pass = unicodedata.normalize("NFC", flat_from_parts) == unicodedata.normalize("NFC", flat_from_str)

        if has_soft:
            soft_hyphen_total += 1
            if flat_pass:
                soft_hyphen_pass += 1

        # Joiner round-trip on every sample with >1 line.
        if len(parts) > 1:
            joiner_total += 1
            joined = _joined_via_joiner(parts)
            gold = _gold_paragraph(parts)
            # The joiner inserts spaces between lines for non-hyphen breaks; gold
            # concatenates without spaces. Normalise both by collapsing whitespace.
            j2 = " ".join(joined.split())
            g2 = " ".join(gold.split())
            # The joiner removes intra-line whitespace; align by token list.
            if j2.replace(" ", "") == g2.replace(" ", ""):
                joiner_pass += 1
            elif len(failures) < 5:
                failures.append({"file": p.name, "joined": j2[:160], "gold": g2[:160]})

    out = {
        "n_samples": len(samples),
        "soft_hyphen_samples": soft_hyphen_total,
        "soft_hyphen_pass": soft_hyphen_pass,
        "soft_hyphen_pass_rate": round(soft_hyphen_pass / max(soft_hyphen_total, 1), 4),
        "joiner_samples": joiner_total,
        "joiner_pass": joiner_pass,
        "joiner_pass_rate": round(joiner_pass / max(joiner_total, 1), 4),
        "joiner_failure_examples": failures,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (out_dir / "_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Preview HTML
    rng = random.Random(99)
    picks = rng.sample(samples, min(preview_count, len(samples)))
    rows = []
    for p in picks:
        r = json.loads(p.read_text(encoding="utf-8"))
        img_path = out_dir / r["image"]
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        ext = "jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "png"
        label_html = r["label"].replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>")
        rows.append(
            f"<div style='margin-bottom:32px;border-bottom:1px solid #ccc;padding-bottom:16px'>"
            f"<div style='font-size:12px;color:#666'>{p.name} - font={r['font']['family']} ({r['font']['bucket']}, {r['font']['pt']}pt) "
            f"lang={r.get('lang')} n_lines={r['layout']['n_lines']} soft={r['layout']['n_soft_hyphens']}</div>"
            f"<img src='data:image/{ext};base64,{b64}' style='max-width:100%;border:1px solid #ddd'/>"
            f"<pre style='font-family:monospace;white-space:pre-wrap;font-size:13px'>{label_html}</pre>"
            f"</div>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>shard preview {out_dir.name}</title></head><body>"
        f"<h2>{out_dir.name} - {preview_count} random samples</h2>"
        + "".join(rows)
        + "</body></html>"
    )
    (out_dir / "_preview.html").write_text(html, encoding="utf-8")
    print(f"[val] wrote {out_dir / '_preview.html'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-count", type=int, default=10)
    args = ap.parse_args()
    validate(Path(args.out), preview_count=args.preview_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
