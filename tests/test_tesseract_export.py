"""Tests for the tesstrain-format exporter.

The exporter emits matching single-line image / .gt.txt pairs and guarantees
the full 117-char inventory is covered across COVERAGE_LINES. lstmf generation
needs a Tesseract binary, so that part is gated behind a skip.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.datagen.tesstrain_export import (
    COVERAGE_LINES,
    _normalise_label,
    _row_text,
    export_shard,
    load_char_set,
)

ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "data" / "fonts"
DEV_TESSDATA = ROOT / "data" / "tesseract" / "tessdata"

# Fine-tuned traineddata is fetched, not committed (scripts/fetch_assets.sh).
pytestmark = pytest.mark.skipif(
    not (DEV_TESSDATA / "mlt.traineddata").exists(),
    reason="no tessdata under data/tesseract; run scripts/fetch_assets.sh",
)


def test_coverage_lines_cover_full_charset():
    charset = set(load_char_set())
    covered = set("".join(COVERAGE_LINES))
    missing = sorted(charset - covered - {" "})
    assert not missing, f"COVERAGE_LINES miss {missing}"


def test_normalise_label_drops_en_and_soft_hyphen():
    assert _normalise_label("fis­ seħħ") == "fis seħħ"
    assert _normalise_label("a – b") == "a — b"
    assert "–" not in _normalise_label("x – y")


def test_row_text_handles_list_text():
    assert _row_text({"text": ["a", "b"]}) == "a\nb"
    assert _row_text({"text": "plain"}) == "plain"


def test_export_pairs_match(tmp_path):
    summary = export_shard(
        shard_name="t",
        count=40,
        out_root=tmp_path,
        fonts_dir=FONTS_DIR,
        english_frac=0.0,
        tessdata_dir=DEV_TESSDATA,
        lang="mlt",
        make_lstmf=False,
    )
    shard_dir = tmp_path / "t"
    tifs = sorted(shard_dir.glob("*.tif"))
    assert tifs, "no images written"
    for tif in tifs:
        gt = tif.with_suffix(".gt.txt")
        box = tif.with_suffix(".box")
        assert gt.is_file(), f"missing gt for {tif.name}"
        assert box.is_file(), f"missing box for {tif.name}"
        assert gt.read_text(encoding="utf-8").strip()
    assert summary["written"] == len(tifs)


def test_export_charset_coverage_holds(tmp_path):
    # With make_lstmf off, every coverage line is emitted, so the .gt.txt set
    # must cover the whole inventory.
    summary = export_shard(
        shard_name="t",
        count=len(COVERAGE_LINES),
        out_root=tmp_path,
        fonts_dir=FONTS_DIR,
        english_frac=0.0,
        tessdata_dir=DEV_TESSDATA,
        lang="mlt",
        make_lstmf=False,
    )
    assert summary["charset_missing"] == []
    assert summary["charset_covered"] == summary["charset_total"]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not present")
def test_export_lstmf_generation(tmp_path):
    summary = export_shard(
        shard_name="t",
        count=12,
        out_root=tmp_path,
        fonts_dir=FONTS_DIR,
        english_frac=0.0,
        tessdata_dir=DEV_TESSDATA,
        lang="mlt",
        make_lstmf=True,
    )
    # Some all-symbol coverage packs fail box alignment; most lines must work.
    assert summary["lstmf_written"] >= summary["written"] // 2
    list_path = tmp_path / "t" / "t.training_files.txt"
    assert list_path.is_file()
