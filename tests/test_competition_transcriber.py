"""Smoke tests for the competition deliverable.

The transcriber needs a Tesseract binary plus an mlt traineddata. On a dev box
those live under data/tesseract/; the harness gets them from the HF bundle.
When neither is present the tests skip rather than fail - CI without Tesseract
is a valid state.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Organiser dev set when available, otherwise the bundled fixtures.
_ORG_DEV = ROOT / "competition_files" / "dev"
DEV_DIR = _ORG_DEV if _ORG_DEV.exists() else ROOT / "fixtures" / "dev"
DEV_TESSDATA = ROOT / "data" / "tesseract" / "tessdata"


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _mlt_available() -> bool:
    return (DEV_TESSDATA / "mlt.traineddata").is_file()


pytestmark = pytest.mark.skipif(
    not (_tesseract_available() and _mlt_available()),
    reason="tesseract binary or mlt.traineddata not present",
)


@pytest.fixture(scope="module")
def transcriber():
    import pytesseract
    from competition_transcriber import CompetitionTranscriber

    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    t = CompetitionTranscriber.__new__(CompetitionTranscriber)
    # Bypass HF download: point the transcriber at the dev tessdata directly.
    t.model_id = "dev-local"
    t.tessdata_dir = str(DEV_TESSDATA)
    t.config = f'--tessdata-dir "{t.tessdata_dir}" --psm 6'
    t.lang = "mlt"
    from malti.line_joiner import RBLineJoiner

    t.joiner = RBLineJoiner()
    # Mirror the optional-pipeline defaults __init__ sets before _warmup, so
    # the smoke test exercises the bare-Tesseract base path.
    t.corrector = None
    t.router = None
    t._lexicon = set()
    t._warmup()
    return t


def test_module_imports():
    import competition_transcriber  # noqa: F401


def test_transcribe_returns_str(transcriber):
    import PIL.Image

    images = sorted(DEV_DIR.glob("*.jpg"))
    assert images, "no dev images found"
    out = transcriber.transcribe(PIL.Image.open(images[0]))
    assert isinstance(out, str)
    assert out.strip()


def test_transcribe_deterministic(transcriber):
    import PIL.Image

    img = PIL.Image.open(sorted(DEV_DIR.glob("*.jpg"))[0])
    assert transcriber.transcribe(img) == transcriber.transcribe(img)


def test_transcribe_handles_non_rgb(transcriber):
    import PIL.Image

    img = PIL.Image.open(sorted(DEV_DIR.glob("*.jpg"))[0]).convert("L")
    out = transcriber.transcribe(img)
    assert isinstance(out, str)


@pytest.mark.skip(reason="requires HF bundle radmada/lv-rover-mlt - run after push")
def test_full_pipeline_integration():
    # Coverage gap: the smoke fixture runs corrector/router=None (bare
    # Tesseract base path). The LV-ROVER vote + confusion corrector that produce
    # the v20 win need the HF bundle (tess_confusion.json + lexicon). Un-skip and
    # construct CompetitionTranscriber() normally once the bundle is pushed to
    # assert the full 5-stream path returns sane output and the lead-marker fix
    # holds through combine_lv on the dev marker crops.
    raise NotImplementedError
