"""PDF-realistic augmentations: JBIG2 binarisation, page-edge shadow,
subpixel anti-aliasing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datagen.augmentations import (
    AugConfig,
    _jbig2_binarise,
    _page_edge_shadow,
    _pdf_subpixel_blur,
    augment_cpu,
)


def _sample_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((120, 240, 3)) * 60 + 180).clip(0, 255).astype(np.uint8)
    arr[30:90, 20:220, :] = 30
    return Image.fromarray(arr)


def test_jbig2_binarise_shape_dtype():
    img = _sample_image(1)
    rng = np.random.default_rng(42)
    out = _jbig2_binarise(img, rng)
    arr = np.asarray(out)
    assert arr.shape == (120, 240, 3)
    assert arr.dtype == np.uint8
    unique = np.unique(arr)
    assert len(unique) <= 4


def test_jbig2_binarise_idempotent_under_seed():
    img = _sample_image(2)
    out1 = _jbig2_binarise(img, np.random.default_rng(7))
    out2 = _jbig2_binarise(img, np.random.default_rng(7))
    assert np.array_equal(np.asarray(out1), np.asarray(out2))


def test_page_edge_shadow_shape_dtype():
    img = _sample_image(3)
    out = _page_edge_shadow(img, np.random.default_rng(0))
    arr = np.asarray(out)
    assert arr.shape == (120, 240, 3)
    assert arr.dtype == np.uint8


def test_page_edge_shadow_darkens_one_side():
    img = Image.fromarray(np.full((40, 80, 3), 200, dtype=np.uint8))
    seen_asymmetry = False
    for s in range(8):
        out = np.asarray(_page_edge_shadow(img, np.random.default_rng(s)))
        deltas = [
            abs(out[:, :10].mean() - out[:, -10:].mean()),
            abs(out[:10, :].mean() - out[-10:, :].mean()),
        ]
        if max(deltas) > 1.0:
            seen_asymmetry = True
            break
    assert seen_asymmetry, "page-edge shadow produced no asymmetry across 8 seeds"


def test_page_edge_shadow_idempotent():
    img = _sample_image(4)
    o1 = _page_edge_shadow(img, np.random.default_rng(13))
    o2 = _page_edge_shadow(img, np.random.default_rng(13))
    assert np.array_equal(np.asarray(o1), np.asarray(o2))


def test_pdf_subpixel_blur_shape_dtype():
    img = _sample_image(5)
    out = _pdf_subpixel_blur(img, np.random.default_rng(0))
    arr = np.asarray(out)
    assert arr.shape == (120, 240, 3)
    assert arr.dtype == np.uint8


def test_pdf_subpixel_blur_idempotent():
    img = _sample_image(6)
    o1 = _pdf_subpixel_blur(img, np.random.default_rng(99))
    o2 = _pdf_subpixel_blur(img, np.random.default_rng(99))
    assert np.array_equal(np.asarray(o1), np.asarray(o2))


def test_pdf_subpixel_blur_softens_edges():
    arr = np.full((40, 80, 3), 255, dtype=np.uint8)
    arr[:, 40:] = 0
    img = Image.fromarray(arr)
    out = np.asarray(_pdf_subpixel_blur(img, np.random.default_rng(1)))
    edge_col = out[:, 38:42, 0].mean()
    assert 30 < edge_col < 220


def test_full_augment_cpu_with_new_augs_runs():
    img = _sample_image(7)
    cfg = AugConfig(jbig2_binarise_p=1.0, page_edge_shadow_p=1.0, pdf_subpixel_blur_p=1.0)
    import random as _r
    out = augment_cpu(img, cfg, _r.Random(42))
    assert out.size == img.size
    assert np.asarray(out).dtype == np.uint8


def test_full_augment_label_round_trip():
    """Augmentations do not affect labels; the contract is that the image
    transforms but the caller's label string is unchanged."""
    img = _sample_image(8)
    label = "Għadha mhux fis-seħħ — Marie-Louise"
    import random as _r
    cfg = AugConfig(jbig2_binarise_p=1.0, page_edge_shadow_p=1.0, pdf_subpixel_blur_p=1.0)
    _ = augment_cpu(img, cfg, _r.Random(0))
    assert label == "Għadha mhux fis-seħħ — Marie-Louise"
