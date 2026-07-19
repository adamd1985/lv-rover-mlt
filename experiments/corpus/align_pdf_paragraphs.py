"""Render EUR-Lex PDF paragraph blocks to crops, paired with authoritative text.

Two independent text sources per document: PyMuPDF's own glyph-layer block
text (position-aware, from the PDF itself) and the Formex-extracted
authoritative text (saved separately by fetch_eurlex_maltese.py, sequence-
only, no position). This script does NOT blindly trust the glyph layer
(broken ToUnicode CMaps are a real problem for scanned
Malta-government PDFs) - it cross-checks each block's glyph text against the
authoritative text via substring/fuzzy matching, and only keeps the block if
they agree. This is stricter than the plan's original design (label from
Formex only) but reuses the glyph layer's *position* information, which
Formex does not carry, while gating on Formex's *content* as ground truth.

Boilerplate filtering: the first block on page 0 is EUR-Lex's own disclaimer
("this document was produced for..."), and most pages open with an amendment
marker ("►B" / "▼B" for "current in-force version B") that is not
running prose. Both are dropped.

Usage:
    python align_pdf_paragraphs.py --doc corpus/eurlex_smoke_verified/<hash>
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF

RENDER_DPI = 200  # matches the real-crop DPI range after 0.5x post-scale; downscaling happens at synth-shard time, not corpus time
BOILERPLATE_MARKERS = ("►B", "▼B", "►M", "▼M")  # >B/<B, >M/<M consolidated-text revision markers
DISCLAIMER_PREFIX = "Dan id-dokument"  # EUR-Lex's own "this document was produced for..." notice


def normalize(s: str) -> str:
    s = re.sub(r"[■-◿]", "", s)  # strip geometric/revision-marker glyphs
    s = re.sub(r"\s+", " ", s).strip()
    return s


def block_matches_authoritative(block_text: str, authoritative_text: str, threshold: float = 0.7) -> tuple[bool, float]:
    """Fuzzy-match a glyph-layer block against the authoritative text pool.

    Since Formex text has no position info, we can't align block-to-line
    directly - instead we measure how much of the *block's own* content is
    covered by a matching run in the authoritative text, using
    SequenceMatcher.find_longest_match repeatedly (greedy longest-common-
    substring coverage). This is length-symmetric: a short block fully
    contained in a much longer document still scores near 1.0, unlike a raw
    ratio() over padded windows, which dilutes on length mismatch.
    """
    nb = normalize(block_text)
    if len(nb) < 15:
        return False, 0.0

    sm = SequenceMatcher(None, nb, authoritative_text, autojunk=False)
    matches = sm.get_matching_blocks()
    covered = sum(m.size for m in matches if m.size >= 8)  # ignore tiny incidental matches
    coverage = covered / len(nb)
    return coverage >= threshold, round(coverage, 3)


def process_doc(doc_dir: Path, out_dir: Path) -> dict:
    pdf_path = doc_dir / "source.pdf"
    text_path = doc_dir / "authoritative_text.txt"
    authoritative_text = normalize(text_path.read_text(encoding="utf-8"))

    doc = fitz.open(str(pdf_path))
    zoom = RENDER_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    dropped = {"boilerplate": 0, "too_short": 0, "no_match": 0}

    for page_idx, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        blocks = page.get_text("blocks")
        for block_idx, b in enumerate(blocks):
            x0, y0, x1, y1, text, bno, btype = b
            raw = text.strip()
            if not raw:
                continue
            if raw.startswith(DISCLAIMER_PREFIX) or any(m in raw for m in BOILERPLATE_MARKERS):
                dropped["boilerplate"] += 1
                # still may contain real prose after the marker on the same block
                raw2 = raw
                for m in BOILERPLATE_MARKERS:
                    raw2 = raw2.replace(m, "")
                raw2 = raw2.strip()
                if raw.startswith(DISCLAIMER_PREFIX) or len(normalize(raw2)) < 15:
                    continue
                raw = raw2

            ok, score = block_matches_authoritative(raw, authoritative_text)
            if not ok:
                dropped["no_match" if len(normalize(raw)) >= 15 else "too_short"] += 1
                continue

            crop_rect = fitz.Rect(x0, y0, x1, y1)
            crop_pix = page.get_pixmap(matrix=mat, clip=crop_rect)
            crop_name = f"p{page_idx:02d}_b{block_idx:03d}.png"
            crop_pix.save(str(out_dir / crop_name))

            pairs.append({
                "crop": crop_name,
                "page": page_idx,
                "block": block_idx,
                "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "text": normalize(raw),
                "match_score": round(score, 3),
            })

    manifest = {
        "source_doc": str(doc_dir),
        "n_pages": doc.page_count,
        "n_pairs": len(pairs),
        "dropped": dropped,
        "pairs": pairs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="path to a fetched EUR-Lex doc dir (contains source.pdf + authoritative_text.txt)")
    ap.add_argument("--out", default=None, help="output dir for crops+manifest (default: <doc>/paragraphs)")
    args = ap.parse_args()

    doc_dir = Path(args.doc)
    out_dir = Path(args.out) if args.out else doc_dir / "paragraphs"
    manifest = process_doc(doc_dir, out_dir)

    print(f"[align] {manifest['n_pages']} pages -> {manifest['n_pairs']} aligned paragraph pairs")
    print(f"[align] dropped: {manifest['dropped']}")
    print(f"[align] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
