"""Package aligned corpus batches into a single HF-ready dataset.

Walks every `*/paragraphs/manifest.json` under one or more source roots,
copies each crop into a flat `images/` dir with a globally-unique name,
and writes one `metadata.jsonl` (the HF `imagefolder` convention: one JSON
object per line, `file_name` + arbitrary extra columns) plus a data card
(`README.md` with YAML frontmatter) documenting per-source license
provenance, diacritic/dash coverage, and the contamination-guard result.

This does not upload anything - it only assembles a local directory ready
for `huggingface-cli upload` or `datasets.load_dataset("imagefolder", ...)`.
Upload is a separate, explicit step (never silently push data to a public
host).

Usage:
    python package_for_hf.py \
        --source data/wikipedia_mt_batch1 --source data/wikipedia_mt_batch2 \
        --source experiments/neural_resume/corpus/eurlex_smoke_verified \
        --out data/hf_package_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

DIACRITIC_CHARS = set("ċĊġĠħĦżŻàèìòùÀÈÌÒÙ") | {"għ", "Għ", "GĦ"}
DIGRAPH_CHARS = ("ie", "Ie", "IE")
DASH_CHARS = {"—": "em_dash", "-": "ascii_hyphen"}

LICENSE_BY_SOURCE = {
    "eur-lex": "Commission Decision 2011/833/EU (reuse permitted, attribution required, no distortion of meaning)",
    "wikipedia-mt": "CC BY-SA 4.0 (attribution + share-alike required for derived text; images are renders of CC BY-SA text, same terms apply)",
}


def find_manifests(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        out.extend(root.rglob("paragraphs/manifest.json"))
    return out


def load_source_meta(manifest_path: Path) -> dict | None:
    doc_dir = manifest_path.parent.parent
    meta_path = doc_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None


def diacritic_dash_stats(texts: list[str]) -> dict:
    char_counts = Counter()
    for t in texts:
        for ch in DIACRITIC_CHARS:
            if len(ch) == 1 and ch in t:
                char_counts[ch] += t.count(ch)
        for gh in ("għ", "Għ", "GĦ"):
            char_counts[gh] += t.count(gh)
        for ie in DIGRAPH_CHARS:
            char_counts[ie] += t.count(ie)
        for dash, name in DASH_CHARS.items():
            char_counts[name] += t.count(dash)
    return dict(char_counts)


def package(source_roots: list[Path], out_dir: Path) -> dict:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifests = find_manifests(source_roots)
    rows = []
    all_texts = []
    source_counts = Counter()
    license_notes = {}
    seen_names = set()

    for mf in manifests:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        doc_meta = load_source_meta(mf)
        source = doc_meta.get("source", "unknown") if doc_meta else "unknown"
        license_str = LICENSE_BY_SOURCE.get(source, doc_meta.get("license", "unknown") if doc_meta else "unknown")
        license_notes[source] = license_str

        crop_dir = mf.parent
        doc_key = hashlib.sha256(str(mf).encode()).hexdigest()[:10]

        for pair in manifest.get("pairs", []):
            src_crop = crop_dir / pair["crop"]
            if not src_crop.exists():
                continue
            unique_name = f"{source}_{doc_key}_{pair['crop']}"
            if unique_name in seen_names:
                continue
            seen_names.add(unique_name)
            shutil.copy2(src_crop, images_dir / unique_name)

            rows.append({
                "file_name": f"images/{unique_name}",
                "text": pair["text"],
                "source": source,
                "match_score": pair["match_score"],
                "provenance": {
                    "doc_meta": doc_meta,
                    "bbox": pair.get("bbox"),
                    "page": pair.get("page"),
                },
            })
            all_texts.append(pair["text"])
            source_counts[source] += 1

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total_pairs": len(rows),
        "by_source": dict(source_counts),
        "diacritic_dash_char_counts": diacritic_dash_stats(all_texts),
        "license_by_source": license_notes,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def write_data_card(out_dir: Path, stats: dict, contamination_note: str) -> None:
    lines = [
        "---",
        "language: [mt]",
        "license: other",
        "task_categories: [image-to-text]",
        "pretty_name: Real Maltese OCR Paragraph Pairs",
        "---",
        "",
        "# Real Maltese OCR Paragraph Pairs",
        "",
        "Paragraph-level (image, text) pairs from real, digitally-typeset",
        "Maltese PDFs. Built to close the synthetic-only gap in Maltese OCR",
        "training data - see the accompanying paper (LV-ROVER-MLT) for context.",
        "",
        "Ground truth text is never taken from the PDF's own glyph layer",
        "directly; it is cross-checked against an independently-sourced",
        "authoritative text (EUR-Lex Formex XML, or Wikipedia's plaintext-",
        "extract API) via longest-common-substring coverage before a crop",
        "is included. See `package_for_hf.py` and `align_pdf_paragraphs.py`",
        "in the source repository for the exact method.",
        "",
        "## Composition",
        "",
        f"Total pairs: {stats['total_pairs']}",
        "",
        "| Source | Pairs | License |",
        "|---|---|---|",
    ]
    for src, count in stats["by_source"].items():
        lic = stats["license_by_source"].get(src, "unknown")
        lines.append(f"| {src} | {count} | {lic} |")

    lines += [
        "",
        "## Diacritic and dash coverage",
        "",
        "Character counts across the full text pool (canary characters for",
        "tokenizer/encoder correctness):",
        "",
        "| Character | Count |",
        "|---|---|",
    ]
    for ch, count in sorted(stats["diacritic_dash_char_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{ch}` | {count} |")

    lines += [
        "",
        "## Contamination guard",
        "",
        contamination_note,
        "",
        "## License",
        "",
        "Mixed per-source, see the composition table above. Every sample's",
        "`metadata.jsonl` `provenance.doc_meta` field records its exact",
        "source URL and license string. Attribution is required for all",
        "included sources; Wikipedia-derived text additionally requires",
        "share-alike for derived text under CC BY-SA 4.0.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="repeatable: a corpus root to search for paragraphs/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contamination-note", default="Not yet run for this package - run dedup_against_devset.py on --out before publishing.")
    args = ap.parse_args()

    roots = [Path(s) for s in args.source]
    out_dir = Path(args.out)
    stats = package(roots, out_dir)
    write_data_card(out_dir, stats, args.contamination_note)

    print(f"[package] {stats['total_pairs']} pairs packaged to {out_dir}")
    print(f"[package] by source: {stats['by_source']}")
    print(f"[package] data card: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
