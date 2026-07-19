"""I1 PDF-to-crops pipeline.

Opens scraped PDFs from data/unlabelled_real/ with pdf2image,
renders pages at 150 DPI, saves as page-level PNG crops to
data/unlabelled_crops/<source>/<hash_stem>_p<page>.png.

Does not run OCR or create labels - these images feed the SimMIM
pretrain stage (deferred) and/or the TTT pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IN_BASE = ROOT / "data" / "unlabelled_real"
OUT_BASE = ROOT / "data" / "unlabelled_crops"
DPI = 150
MAX_PAGES_PER_PDF = 20
MAX_CROPS_TOTAL = 200000


def convert_pdf(pdf_path: Path, out_dir: Path, stem: str) -> int:
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf_path), dpi=DPI, last_page=MAX_PAGES_PER_PDF)
    except Exception as e:
        print(f"[pdf2crops] skip {pdf_path.name}: {e}")
        return 0

    saved = 0
    for i, page in enumerate(pages):
        out_path = out_dir / f"{stem}_p{i:03d}.png"
        if out_path.exists():
            continue
        try:
            page.save(str(out_path), "PNG", optimize=False)
            saved += 1
        except Exception:
            pass
    return saved


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    total = 0

    for source_dir in sorted(IN_BASE.iterdir()):
        if not source_dir.is_dir():
            continue
        out_dir = OUT_BASE / source_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(source_dir.glob("*.pdf"))
        print(f"[pdf2crops] {source_dir.name}: {len(pdfs)} PDFs")

        for pdf_path in pdfs:
            if total >= MAX_CROPS_TOTAL:
                print(f"[pdf2crops] crop cap {MAX_CROPS_TOTAL} reached")
                return
            n = convert_pdf(pdf_path, out_dir, pdf_path.stem)
            total += n

        print(f"[pdf2crops] {source_dir.name} done. cumulative crops={total}")

    print(f"[pdf2crops] done. total crops={total}")


if __name__ == "__main__":
    main()
