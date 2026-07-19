"""4-stage CER ablation, Luxembourgish: stock_raw -> stock_plus_postproc,
five_stream_pre_postproc -> five_stream_full_pipeline. Additive diagnostic
script, no retraining. Reuses bootstrap_stats and ltz_transcriber as-is.
One Tesseract call per stream per image: stock is called once (reused by
both stock stages), transcribe() is called once with apply_postproc=False
and the post-processing chain is applied manually to derive the
full-pipeline hypothesis, so the five streams are never OCR'd twice."""
import json
import sys
import unicodedata
from pathlib import Path

import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from bootstrap_stats import aggregate_cer, bootstrap_ci, edit_distance, permutation_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltz_transcriber import _fix_apostrophe, _fix_doublequote, _fix_lead_marker, _normalise, transcribe

_CURLY_TO_STRAIGHT = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
})


def _convention_normalize(text: str) -> str:
    return text.translate(_CURLY_TO_STRAIGHT)


with open("data/eval_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

stages = {
    "stock_raw": [],
    "stock_plus_postproc": [],
    "five_stream_pre_postproc": [],
    "five_stream_full_pipeline": [],
}
conv_stages = {
    "stock_raw": [],
    "five_stream_pre_postproc": [],
}

for i, item in enumerate(manifest):
    img = Image.open(f"data/antiqua/antiqua/{item['stem']}.png")
    ref = unicodedata.normalize("NFC", item["text"])

    stock_raw_hyp = unicodedata.normalize(
        "NFC", pytesseract.image_to_string(img, lang="ltz", config="--psm 7")
    )
    stock_postproc_hyp = _fix_doublequote(_fix_apostrophe(_fix_lead_marker(_normalise(stock_raw_hyp))))

    joined = transcribe(img, apply_postproc=False)
    full_hyp = _fix_doublequote(_fix_apostrophe(_fix_lead_marker(joined)))

    stages["stock_raw"].append((edit_distance(ref, stock_raw_hyp), len(ref)))
    stages["stock_plus_postproc"].append((edit_distance(ref, stock_postproc_hyp), len(ref)))
    stages["five_stream_pre_postproc"].append((edit_distance(ref, joined), len(ref)))
    stages["five_stream_full_pipeline"].append((edit_distance(ref, full_hyp), len(ref)))

    ref_conv = _convention_normalize(ref)
    conv_stages["stock_raw"].append(
        (edit_distance(ref_conv, _convention_normalize(stock_raw_hyp)), len(ref_conv))
    )
    conv_stages["five_stream_pre_postproc"].append(
        (edit_distance(ref_conv, _convention_normalize(joined)), len(ref_conv))
    )

    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(manifest)} scored")

summary = {name: aggregate_cer(pairs) for name, pairs in stages.items()}
conv_summary = {name: aggregate_cer(pairs) for name, pairs in conv_stages.items()}

print("stage CER:")
for name, cer in summary.items():
    print(f"  {name}: {cer:.5f}")
print("convention-normalized CER:")
for name, cer in conv_summary.items():
    print(f"  {name}: {cer:.5f}")

comparisons = {}
for name, (stage_a, stage_b) in {
    "stock_raw_vs_five_stream_pre_postproc": ("stock_raw", "five_stream_pre_postproc"),
    "five_stream_pre_postproc_vs_full_pipeline": ("five_stream_pre_postproc", "five_stream_full_pipeline"),
    "stock_raw_vs_five_stream_full_pipeline": ("stock_raw", "five_stream_full_pipeline"),
}.items():
    per_item = [
        (a[0], a[1], b[0], b[1])
        for a, b in zip(stages[stage_a], stages[stage_b])
    ]
    cer_a, cer_b, delta, lo, hi = bootstrap_ci(per_item, seed=42)
    p_value, critical_value = permutation_test(per_item, delta, seed=43)
    comparisons[name] = {
        "cer_a": cer_a,
        "cer_b": cer_b,
        "delta": delta,
        "ci_95_lo": lo,
        "ci_95_hi": hi,
        "excludes_zero": lo > 0 or hi < 0,
        "permutation_p_value": p_value,
        "permutation_critical_value_alpha05": critical_value,
        "significant_alpha05": p_value < 0.05,
    }
    print(f"{name}: CER_a={cer_a:.5f} CER_b={cer_b:.5f} delta={delta:.5f} "
          f"CI=[{lo:.5f}, {hi:.5f}] p={p_value:.5f}")

with open("results/ablation_stages.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_items": len(manifest),
        "per_item": stages,
        "per_item_convention_normalized": conv_stages,
        "summary_cer": summary,
        "convention_normalized_cer": conv_summary,
        "comparisons": comparisons,
        "seed": 42,
    }, f, indent=2)
print("wrote results/ablation_stages.json")
