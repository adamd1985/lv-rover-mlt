"""Smoke test for the v0 synth pipeline.

Renders 8 paragraphs against the embedded Maltese fixture and asserts:
(a) canary characters round-trip in labels
(b) both label-bearing dashes can appear; en-dash never leaks into a label
(c) at least one il-/is-/id-/it-/l-/fis- structural article in 8 samples on average
(d) CUDA augmentation path runs if available, else skipped
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The validated font pool is fetched, not committed (scripts/fetch_assets.sh
# --fonts). Without it there is nothing to render, so skip rather than fail.
_FONT_DIR = ROOT / "data" / "fonts"
pytestmark = pytest.mark.skipif(
    not any(_FONT_DIR.rglob("*.ttf")) and not any(_FONT_DIR.rglob("*.otf")),
    reason="no font pool under data/fonts; run scripts/fetch_assets.sh --fonts",
)

from src.datagen.corpus_loader import iter_paragraphs
from src.datagen.font_loader import CANARY, load_fonts
from src.datagen.maltese_paragraph import (
    ARTICLE_PREFIXES,
    EM_DASH,
    EN_DASH,
    LayoutConfig,
    MalteseParagraph,
)
from src.datagen.augmentations import AugConfig, augment_cpu, augment_cuda


def _load_cfg():
    return yaml.safe_load((ROOT / "configs" / "synth_v0.yaml").read_text(encoding="utf-8"))


def _build_pipeline(cfg, seed: int = 42):
    rng = random.Random(seed)
    fonts = load_fonts(
        ROOT / cfg["fonts"]["dir"],
        handwriting_rate=cfg["fonts"]["handwriting_rate"],
        rng=rng,
        fallback_system=cfg["fonts"]["fallback_system"],
    )
    assert len(fonts) > 0, "no fonts available, even system fallback failed"
    layout = LayoutConfig(**cfg["layout"])
    pipe = MalteseParagraph(font_sampler=fonts, layout=layout, rng=rng)
    corpus = iter_paragraphs(
        english_frac=cfg["corpus"]["english_frac"],
        use_streaming=cfg["corpus"]["use_streaming"],
        rng=rng,
    )
    return pipe, corpus


def test_render_eight_samples_canary_and_articles():
    cfg = _load_cfg()
    pipe, corpus = _build_pipeline(cfg, seed=42)

    samples = []
    t0 = time.perf_counter()
    for _ in range(cfg["smoke"]["n_samples"]):
        samples.append(pipe.render_one(next(corpus)))
    wall = time.perf_counter() - t0
    print(f"\n[smoke] rendered 8 paragraphs in {wall:.3f}s")

    for img, lbl, parts, meta in samples:
        assert isinstance(parts, list) and all(isinstance(p, str) for p in parts)
        assert lbl == "\n".join(parts), "label_str must equal newline-joined parts"
        assert len(parts) == meta.n_lines
    all_labels = " ".join(s[1] for s in samples)
    print(f"[smoke] canary char hits in 8-sample labels: "
          f"{[c for c in CANARY if c in all_labels]}")

    article_hits = sum(
        1 for _, label, _, _ in samples if any(p in label.lower() for p in ARTICLE_PREFIXES)
    )
    print(f"[smoke] samples with article prefix: {article_hits}/8")
    assert article_hits >= 1


def test_canary_round_trips():
    cfg = _load_cfg()
    cfg["layout"]["article_inject_rate"] = 0.0
    cfg["layout"]["dash_substitute_p"] = 0.0
    cfg["layout"]["hyphenation_rate"] = 0.0
    rng = random.Random(99)
    fonts = load_fonts(
        ROOT / cfg["fonts"]["dir"],
        handwriting_rate=0.0,
        rng=rng,
        fallback_system=True,
    )
    pipe = MalteseParagraph(font_sampler=fonts, layout=LayoutConfig(**cfg["layout"]), rng=rng)
    src = "ĊENSURA ċ Ġid ġenwin Ħajja ħafna Żgur żball à dritt ì ò ù - —"
    _, label, _, _ = pipe.render_one({"text": src, "lang": "mt"})
    missing = [c for c in CANARY if c not in label]
    print(f"[smoke] canary round-trip missing: {missing}")
    assert not missing, f"canary chars dropped: {missing}"


def test_dash_codepoints_reachable():
    cfg = _load_cfg()
    cfg["layout"]["dash_substitute_p"] = 0.9
    pipe, corpus = _build_pipeline(cfg, seed=7)
    blob = ""
    for _ in range(8):
        _, label, _, _ = pipe.render_one(next(corpus))
        blob += " " + label
    has_hyphen = "-" in blob
    has_en = EN_DASH in blob
    has_em = EM_DASH in blob
    print(f"[smoke] label dashes: hyphen={has_hyphen} en={has_en} em={has_em}")
    assert has_hyphen
    assert has_em
    # En-dash U+2013 is image-only; it must never leak into a label.
    assert not has_en


def test_label_parts_contract_and_soft_hyphen_recoverable():
    cfg = _load_cfg()
    cfg["layout"]["hyphenation_rate"] = 1.0
    cfg["layout"]["paragraph_width_lo"] = 600
    cfg["layout"]["paragraph_width_hi"] = 650
    cfg["layout"]["font_pt_lo"] = 10
    cfg["layout"]["font_pt_hi"] = 12
    cfg["layout"]["article_inject_rate"] = 0.0
    cfg["layout"]["dash_substitute_p"] = 0.0
    cfg["layout"]["leading_bullet_p"] = 0.0
    rng = random.Random(123)
    fonts = load_fonts(ROOT / cfg["fonts"]["dir"], handwriting_rate=0.0, rng=rng, fallback_system=True)
    pipe = MalteseParagraph(font_sampler=fonts, layout=LayoutConfig(**cfg["layout"]), rng=rng)
    SOFT = "­"
    found_break = False
    for _ in range(30):
        text = ("ilkoll qed naħdmu fuq strateġija ġdida għall-iżvilupp ekonomiku ġenerali "
                "biex insostnu il-pajjiż b'inċentivi varji u investimenti fis-saħħa")
        img, label_str, parts, meta = pipe.render_one({"text": text, "lang": "mt"})
        assert label_str == "\n".join(parts)
        if any(SOFT in p for p in parts):
            found_break = True
            joined_no_soft = "".join(parts).replace(SOFT, "")
            paragraph_view = label_str.replace("\n", "").replace(SOFT, "")
            assert joined_no_soft == paragraph_view
            assert all("\n" not in p for p in parts)
            break
    assert found_break, "expected at least one soft-hyphen split with rate=1.0"


def test_aug_cpu_path():
    cfg = _load_cfg()
    pipe, corpus = _build_pipeline(cfg, seed=11)
    img, _, _, _ = pipe.render_one(next(corpus))
    acfg = AugConfig(**cfg["augmentations"])
    t0 = time.perf_counter()
    out = augment_cpu(img, acfg, random.Random(1))
    dt = time.perf_counter() - t0
    print(f"[smoke] augment_cpu: {dt*1000:.1f} ms")
    assert out.size[0] > 0 and out.size[1] > 0


def test_public_fallback_mock(monkeypatch):
    """Exercise the public_fallback path with both source kinds mocked.
    No network: HF dataset and urllib.urlopen are both stubbed."""
    from src.datagen import corpus_loader as cl

    long_mt_a = (
        "Il-Gvern ħabbar miżuri ġodda dwar is-saħħa pubblika fil-pajjiż. "
        "Ir-riċerkaturi tal-Università studjaw il-bidla fil-klima fil-Mediterran "
        "u t-temperaturi tal-baħar qed jogħlew b'rata mgħaġġla. "
        "L-għada filgħodu nġabret folla fil-pjazza prinċipali biex tesprimi "
        "s-solidarjetà tagħha mal-ħaddiema. Ir-rappreżentanti tat-trade unions "
        "tkellmu favur paga minima diċenti u kundizzjonijiet xierqa tax-xogħol. "
        "Il-Ministru tal-Finanzi wieġeb li l-pjan jinkludi investiment fl-edukazzjoni."
    )
    long_mt_b = (
        "Il-każ tressaq quddiem il-qorti tal-maġistrati ilbieraħ filgħaxija. "
        "Ix-xufiera huma mħeġġa jirrispettaw is-sinjali tat-triq fil-bliet ewlenin. "
        "L-orizzont kien imdallam meta s-sajjieda telqu mill-port qabel is-sebħ. "
        "Id-dgħajjes tagħhom kienu mgħobbija bix-xbieki u l-armar kollu meħtieġ. "
        "Il-konferenza tinżamm bejn it-12 u l-14 ta' Ġunju fil-Belt Valletta. "
        "Fost il-kelliema hemm akkademiċi minn diversi pajjiżi Ewropej."
    )
    wiki_rows = [{"text": long_mt_a}, {"text": long_mt_b}] * 50

    def fake_load_dataset(path, config, split="train", streaming=True):
        assert path == "wikimedia/wikipedia"
        assert config == "20231101.mt"
        return iter(wiki_rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset, raising=False)
    import sys, types
    if "datasets" not in sys.modules:
        mod = types.ModuleType("datasets")
        mod.load_dataset = fake_load_dataset
        sys.modules["datasets"] = mod
    else:
        sys.modules["datasets"].load_dataset = fake_load_dataset

    ud_blob = (
        "# sent_id = 1\n"
        "# text = Ħafna nies fis-suq it-Tlieta li ġej.\n"
        "1\tĦafna\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "\n"
        "# text = Il-pulizija tinforza l-limiti tal-veloċità għall-pedestrijani.\n"
        "1\tIl\t_\t_\t_\t_\t_\t_\t_\t_\n"
    ) * 30

    class _FakeResp:
        def __init__(self, data): self._d = data
        def read(self): return self._d.encode("utf-8")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout=30: _FakeResp(ud_blob)
    )

    cfg = {
        "english_frac": 0.0,
        "sources": [
            {"name": "wikipedia_mt", "kind": "hf", "path": "wikimedia/wikipedia",
             "config": "20231101.mt", "split": "train", "text_field": "text",
             "weight": 0.8},
            {"name": "ud_mudt", "kind": "ud_conllu", "weight": 0.2,
             "urls": ["https://example.invalid/ud.conllu"]},
        ],
    }
    rng = random.Random(42)
    it = cl.public_fallback_stream(cfg, rng=rng, do_canary_scan=False)
    seen = {"wikipedia_mt": 0, "ud_mudt": 0}
    blob = ""
    for _ in range(80):
        rec = next(it)
        assert rec["lang"] == "mt"
        assert rec["domain"] in seen
        seen[rec["domain"]] += 1
        blob += " " + rec["text"]
    print(f"[smoke] public_fallback source mix: {seen}")
    assert seen["wikipedia_mt"] > 0
    assert seen["ud_mudt"] > 0
    missing = [c for c in ("ħ", "ġ", "ċ", "ż") if c not in blob]
    assert not missing, f"canary chars missing from public_fallback stream: {missing}"


def test_aug_cuda_path_if_available():
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = _load_cfg()
    pipe, corpus = _build_pipeline(cfg, seed=13)
    img, _, _, _ = pipe.render_one(next(corpus))
    acfg = AugConfig(**cfg["augmentations"])

    out = augment_cuda(img, acfg, random.Random(1))  # warmup
    torch.cuda.synchronize()

    n = 8
    t0 = time.perf_counter()
    for i in range(n):
        out = augment_cuda(img, acfg, random.Random(1000 + i))
    torch.cuda.synchronize()
    dt_cu = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for i in range(n):
        out_c = augment_cpu(img, acfg, random.Random(1000 + i))
    dt_cpu = (time.perf_counter() - t0) / n

    print(f"[smoke] augment_cuda: {dt_cu*1000:.1f} ms/img  cpu: {dt_cpu*1000:.1f} ms/img")
    assert out.size[0] > 0
