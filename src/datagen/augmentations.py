"""PDF-realistic augmentations.

the design decision set only: JPEG re-encode, rotation +-1.5deg, Gaussian blur sigma 0.3-0.8,
salt-pepper, mild elastic, brightness/contrast jitter, ink bleed, column edge
crop. No perspective warp, no curved baselines.

CPU path uses PIL+numpy. CUDA path uses torch tensors moved to device; we keep
it dependency-light so it runs without torchvision. The smoke test benchmarks
both and reports.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


@dataclass
class AugConfig:
    jpeg_q_lo: int = 40
    jpeg_q_hi: int = 95
    rotation_deg: float = 1.5
    blur_sigma_lo: float = 0.3
    blur_sigma_hi: float = 0.8
    sp_prob: float = 0.01
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    ink_bleed_p: float = 0.25
    column_edge_crop_p: float = 0.10
    elastic_alpha: float = 1.5
    # v2 PDF-realistic additions.
    jbig2_binarise_p: float = 0.05
    page_edge_shadow_p: float = 0.10
    pdf_subpixel_blur_p: float = 0.30
    # v1 domain-gap fix: grey background jitter.
    # 0 = disabled (pure white). Otherwise uniform(grey_bg_lo, grey_bg_hi) per image.
    grey_bg_lo: int = 0
    grey_bg_hi: int = 0


def _rotate(img: Image.Image, deg: float) -> Image.Image:
    return img.rotate(deg, resample=Image.BILINEAR, fillcolor=255, expand=False)


def _blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _jitter(img: Image.Image, b: float, c: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(1.0 + b)
    img = ImageEnhance.Contrast(img).enhance(1.0 + c)
    return img


def _salt_pepper(arr: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    if p <= 0:
        return arr
    mask = rng.random(arr.shape[:2])
    out = arr.copy()
    out[mask < (p / 2)] = 0
    out[mask > (1 - p / 2)] = 255
    return out


def _jpeg_recompress(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _ink_bleed(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)).convert("L"), dtype=np.float32)
    alpha = float(rng.uniform(0.15, 0.35))
    bled = arr * (1.0 - alpha) + np.minimum(arr, blurred) * alpha
    return Image.fromarray(np.clip(bled, 0, 255).astype(np.uint8)).convert("RGB")


def _column_edge_crop(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    side = rng.integers(0, 2)
    cut = int(rng.uniform(2, 8))
    box = (cut, 0, w, h) if side == 0 else (0, 0, w - cut, h)
    return img.crop(box).resize((w, h))


def _mild_elastic(img: Image.Image, alpha: float, rng: np.random.Generator) -> Image.Image:
    if alpha <= 0:
        return img
    w, h = img.size
    dx = rng.uniform(-alpha, alpha, size=(h, w)).astype(np.float32)
    dy = rng.uniform(-alpha, alpha, size=(h, w)).astype(np.float32)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    ys, xs = np.indices((h, w), dtype=np.float32)
    xs2 = np.clip(xs + dx, 0, w - 1).astype(np.int32)
    ys2 = np.clip(ys + dy, 0, h - 1).astype(np.int32)
    out = arr[ys2, xs2]
    return Image.fromarray(out.astype(np.uint8))


def _jbig2_binarise(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Otsu-thresholded 1-bit binarisation with optional 1px dilation. Emulates
    JBIG2 / fax-scan compression typical of older PDFs (Otsu 1979)."""
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    var_max = -1.0
    thr = 127
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t * hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > var_max:
            var_max = var
            thr = t
    bin_arr = (gray > thr).astype(np.uint8) * 255
    if rng.random() < 0.5:
        a = bin_arr
        eroded = np.minimum(np.minimum(
            np.pad(a, ((0, 0), (1, 0)), constant_values=255)[:, :-1],
            np.pad(a, ((0, 0), (0, 1)), constant_values=255)[:, 1:]),
            np.minimum(
                np.pad(a, ((1, 0), (0, 0)), constant_values=255)[:-1, :],
                np.pad(a, ((0, 1), (0, 0)), constant_values=255)[1:, :]),
        )
        bin_arr = eroded
    out = np.stack([bin_arr] * 3, axis=-1)
    return Image.fromarray(out)


def _page_edge_shadow(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Linear darkening gradient from one edge to the opposite. Emulates the
    common scanner-bed shadow at the bound spine or page edge."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    side = int(rng.integers(0, 4))
    strength = float(rng.uniform(0.15, 0.45))
    if side == 0:
        ramp = np.linspace(1.0 - strength, 1.0, w, dtype=np.float32)[None, :, None]
    elif side == 1:
        ramp = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)[None, :, None]
    elif side == 2:
        ramp = np.linspace(1.0 - strength, 1.0, h, dtype=np.float32)[:, None, None]
    else:
        ramp = np.linspace(1.0, 1.0 - strength, h, dtype=np.float32)[:, None, None]
    out = np.clip(arr * ramp, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _pdf_subpixel_blur(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Resample-down then resample-up with PIL.LANCZOS to emulate the
    subpixel anti-aliased look of rasterised PDF text (Smith 1995)."""
    w, h = img.size
    factor = float(rng.uniform(0.78, 0.94))
    nw = max(8, int(round(w * factor)))
    nh = max(8, int(round(h * factor)))
    small = img.resize((nw, nh), resample=Image.LANCZOS)
    return small.resize((w, h), resample=Image.LANCZOS)


def augment_cpu(
    img: Image.Image,
    cfg: AugConfig = AugConfig(),
    rng: Optional[random.Random] = None,
) -> Image.Image:
    rng = rng or random.Random()
    npr = np.random.default_rng(rng.randint(0, 2**31 - 1))

    img = _rotate(img, rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    img = _blur(img, rng.uniform(cfg.blur_sigma_lo, cfg.blur_sigma_hi))
    if rng.random() < cfg.pdf_subpixel_blur_p:
        img = _pdf_subpixel_blur(img, npr)
    img = _jitter(
        img,
        rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter),
        rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter),
    )
    if rng.random() < cfg.ink_bleed_p:
        img = _ink_bleed(img, npr)
    if rng.random() < cfg.column_edge_crop_p:
        img = _column_edge_crop(img, npr)
    img = _mild_elastic(img, cfg.elastic_alpha, npr)

    if rng.random() < cfg.page_edge_shadow_p:
        img = _page_edge_shadow(img, npr)
    if rng.random() < cfg.jbig2_binarise_p:
        img = _jbig2_binarise(img, npr)

    arr = np.asarray(img.convert("RGB")).astype(np.uint16)
    arr = _salt_pepper(arr.astype(np.uint8), cfg.sp_prob, npr).astype(np.uint16)
    if cfg.grey_bg_lo > 0 and cfg.grey_bg_hi >= cfg.grey_bg_lo:
        bg_val = rng.randint(cfg.grey_bg_lo, cfg.grey_bg_hi)
        # Blend white pixels (>240) toward bg_val. Non-text regions shift; ink stays.
        white_mask = (arr.min(axis=2) > 240)
        arr[white_mask] = bg_val
    img = Image.fromarray(arr.astype(np.uint8))

    return _jpeg_recompress(img, rng.randint(cfg.jpeg_q_lo, cfg.jpeg_q_hi))


def augment_cuda(
    img: Image.Image,
    cfg: AugConfig = AugConfig(),
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """GPU-accelerated subset.

    Moves the image tensor to CUDA for the per-pixel operations that benefit:
    brightness/contrast, salt-pepper, and elastic resampling via grid_sample.
    Rotation, blur, JPEG re-encode stay on CPU (PIL is already fast there).
    Falls back to CPU if CUDA is unavailable.
    """
    import torch

    rng = rng or random.Random()
    npr = np.random.default_rng(rng.randint(0, 2**31 - 1))
    if not torch.cuda.is_available():
        return augment_cpu(img, cfg, rng)
    device = torch.device("cuda")

    img = _rotate(img, rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    img = _blur(img, rng.uniform(cfg.blur_sigma_lo, cfg.blur_sigma_hi))

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0)

    b = 1.0 + rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)
    c = 1.0 + rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter)
    t = t * b
    mean = t.mean()
    t = (t - mean) * c + mean

    if cfg.sp_prob > 0:
        mask = torch.rand(t.shape[2:], device=device, dtype=torch.float16)
        t[:, :, mask < (cfg.sp_prob / 2)] = 0
        t[:, :, mask > (1 - cfg.sp_prob / 2)] = 1

    if cfg.elastic_alpha > 0:
        h, w = t.shape[2], t.shape[3]
        ys = torch.linspace(-1, 1, h, device=device, dtype=torch.float16)
        xs = torch.linspace(-1, 1, w, device=device, dtype=torch.float16)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dx = (torch.rand_like(gx) - 0.5) * (cfg.elastic_alpha / (w / 2))
        dy = (torch.rand_like(gy) - 0.5) * (cfg.elastic_alpha / (h / 2))
        grid = torch.stack([gx + dx, gy + dy], dim=-1).unsqueeze(0).float()
        t = torch.nn.functional.grid_sample(
            t.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        ).to(torch.float16)

    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    img = Image.fromarray((t * 255.0).astype(np.uint8))

    if rng.random() < cfg.pdf_subpixel_blur_p:
        img = _pdf_subpixel_blur(img, npr)
    if rng.random() < cfg.ink_bleed_p:
        img = _ink_bleed(img, npr)
    if rng.random() < cfg.column_edge_crop_p:
        img = _column_edge_crop(img, npr)
    if rng.random() < cfg.page_edge_shadow_p:
        img = _page_edge_shadow(img, npr)
    if rng.random() < cfg.jbig2_binarise_p:
        img = _jbig2_binarise(img, npr)

    return _jpeg_recompress(img, rng.randint(cfg.jpeg_q_lo, cfg.jpeg_q_hi))
