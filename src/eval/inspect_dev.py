"""Dev-set inspection. Populates data/dev/_stats.json from the dropped set.

Organiser format (`competition_files/dev/texts.json`):
  A single JSON list + cropped paragraph JPGs. Each entry:
    {"image": "001.jpg", "text": "<paragraph>", "as_lines": ["line", ...]}
  `text` is the paragraph-form gold (scored for CER). `as_lines` is the line
  strings as in the PDF, including hyphens. The key is `as_lines`, not `lines`.

Auto-detects two layouts:
  (a) one manifest JSON at the root of dev_root (any *.json that isn't _stats
      or _summary). Holds a list of paragraph dicts.
  (b) per-paragraph sidecars: each image has a sibling .json with one entry.

Invocation:
    python -m src.eval.inspect_dev --dev_root competition_files/dev \\
        --out data/dev/_stats.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import random
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.eval.buckets import (
    STRUCTURAL_PREFIX,
    bucket_em_dash,
    bucket_il_prefix,
    bucket_language,
    bucket_length,
    bucket_line_count,
    bucket_line_hyphen,
    compute_quartiles,
)
from src.eval.cer import normalise

ARTICLE_PREFIXES: Tuple[str, ...] = (
    "il-", "is-", "id-", "it-", "in-", "ir-", "ix-", "iz-",
    "l-", "fil-", "fis-", "fit-", "fid-", "fl-",
    "bil-", "bis-", "bit-", "bid-",
    "tal-", "tas-", "tat-", "tad-",
    "min-", "mil-", "mis-", "mit-",
)

MALTESE_CANARIES = ["Ċ", "ċ", "Ġ", "ġ",
                    "Ħ", "ħ", "Ż", "ż"]
GRAVE_VOWELS = ["À", "à", "È", "è",
                "Ì", "ì", "Ò", "ò",
                "Ù", "ù"]

RANDOM_SEED = 42
SAMPLE_N_FOR_PADDING = 50


def _is_manifest(p: Path) -> bool:
    name = p.name
    return p.suffix == ".json" and name not in ("_stats.json",)


def discover_manifest(dev_root: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (mode, entries). Raises FileNotFoundError if neither layout found."""
    if not dev_root.exists():
        raise FileNotFoundError(f"dev_root does not exist: {dev_root}")
    candidates = sorted(p for p in dev_root.glob("*.json") if _is_manifest(p))
    for cand in candidates:
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) \
                and ("text" in data[0] or "label" in data[0]):
            return f"single_manifest:{cand.name}", data
    sidecar_entries: List[Dict[str, Any]] = []
    images = sorted(
        p for p in dev_root.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for img in images:
        sidecar = img.with_suffix(".json")
        if sidecar.exists():
            try:
                entry = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entry.setdefault("image", str(img.relative_to(dev_root)))
                entry.setdefault("id", img.stem)
                sidecar_entries.append(entry)
    if sidecar_entries:
        return "per_paragraph_sidecars", sidecar_entries
    raise FileNotFoundError(
        f"No manifest found in {dev_root}. Expected either a single *.json "
        f"with a list of paragraph dicts, or per-image .json sidecars."
    )


def _entry_text(e: Dict[str, Any]) -> str:
    return e.get("text") or e.get("label") or e.get("transcription") or ""


def _entry_image(e: Dict[str, Any]) -> Optional[str]:
    return e.get("image") or e.get("image_path") or e.get("file") or None


def _entry_id(e: Dict[str, Any]) -> str:
    return str(e.get("id") or _entry_image(e) or "")


def _entry_lines(e: Dict[str, Any]) -> Optional[List[str]]:
    for key in ("as_lines", "lines"):
        if key in e and isinstance(e[key], list):
            return [str(x) for x in e[key]]
    t = _entry_text(e)
    if "\n" in t:
        return t.split("\n")
    return None


def _has_line_field(e: Dict[str, Any]) -> bool:
    return ("as_lines" in e and isinstance(e["as_lines"], list)) or (
        "lines" in e and isinstance(e["lines"], list)
    )


def _build_char_inventory(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_first: Dict[str, str] = {}
    counter: Dict[str, int] = {}
    for e in entries:
        eid = _entry_id(e)
        text = normalise(_entry_text(e))
        for ch in text:
            counter[ch] = counter.get(ch, 0) + 1
            if ch not in seen_first:
                seen_first[ch] = eid
    inventory: List[Dict[str, Any]] = []
    for ch, count in counter.items():
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNNAMED"
        inventory.append({
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "name": name,
            "count": count,
            "first_image_id": seen_first[ch],
        })
    inventory.sort(key=lambda d: (-d["count"], d["codepoint"]))
    return inventory


def _measure_white_margin(img: "Image.Image") -> Dict[str, int]:
    """Top/right/bottom/left non-white margin in pixels via Otsu binarise."""
    import numpy as np  # local: keep import optional
    g = img.convert("L")
    arr = np.asarray(g)
    h, w = arr.shape
    if h == 0 or w == 0:
        return {"top": 0, "right": 0, "bottom": 0, "left": 0}
    threshold = _otsu(arr)
    ink = arr < threshold
    rows = ink.any(axis=1)
    cols = ink.any(axis=0)
    if not rows.any() or not cols.any():
        return {"top": h, "right": w, "bottom": h, "left": w}
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    top = int(row_idx[0])
    bottom = int(h - 1 - row_idx[-1])
    left = int(col_idx[0])
    right = int(w - 1 - col_idx[-1])
    return {"top": top, "right": right, "bottom": bottom, "left": left}


def _otsu(arr) -> int:
    import numpy as np
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    threshold = 127
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
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def _crop_padding_stats(
    entries: List[Dict[str, Any]],
    dev_root: Path,
    rng: random.Random,
) -> Dict[str, Any]:
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return {
            "n_sampled": 0,
            "method": "schema_field_absent_in_dev",
            "margin_pixels": {
                "top":    {"median": "schema_field_absent_in_dev",
                           "p25": "schema_field_absent_in_dev",
                           "p75": "schema_field_absent_in_dev"},
                "right":  {"median": "schema_field_absent_in_dev",
                           "p25": "schema_field_absent_in_dev",
                           "p75": "schema_field_absent_in_dev"},
                "bottom": {"median": "schema_field_absent_in_dev",
                           "p25": "schema_field_absent_in_dev",
                           "p75": "schema_field_absent_in_dev"},
                "left":   {"median": "schema_field_absent_in_dev",
                           "p25": "schema_field_absent_in_dev",
                           "p75": "schema_field_absent_in_dev"},
            },
            "tight_crop_share": "schema_field_absent_in_dev",
            "verdict": "schema_field_absent_in_dev",
        }
    from PIL import Image  # noqa: F811
    with_img = [e for e in entries if _entry_image(e)]
    if not with_img:
        return {
            "n_sampled": 0,
            "method": "Otsu binarise, non-background bbox to crop edge",
            "margin_pixels": {k: {"median": None, "p25": None, "p75": None}
                              for k in ("top", "right", "bottom", "left")},
            "tight_crop_share": None,
            "verdict": "no_images_found",
        }
    pool = sorted(with_img, key=_entry_id)
    sample = pool if len(pool) <= SAMPLE_N_FOR_PADDING \
        else rng.sample(pool, SAMPLE_N_FOR_PADDING)
    margins: Dict[str, List[int]] = {"top": [], "right": [], "bottom": [], "left": []}
    for e in sample:
        img_name = str(_entry_image(e))
        ip = dev_root / img_name
        if not ip.exists():
            ip = Path(img_name)
        if not ip.exists():
            continue
        try:
            with Image.open(ip) as im:
                m = _measure_white_margin(im)
        except (OSError, ValueError):
            continue
        for k, v in m.items():
            margins[k].append(v)
    n_sampled = len(margins["top"])

    def _q(vals: List[int], q: float) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return float(s[i])

    mp = {
        k: {
            "median": _q(margins[k], 0.5),
            "p25": _q(margins[k], 0.25),
            "p75": _q(margins[k], 0.75),
        }
        for k in margins
    }
    medians = [mp[k]["median"] for k in mp if mp[k]["median"] is not None]
    overall_median = statistics.median(medians) if medians else None
    tight_share: Optional[float] = None
    if n_sampled:
        tight = 0
        for i in range(n_sampled):
            edges = [margins[k][i] for k in margins]
            if max(edges) <= 4:
                tight += 1
        tight_share = tight / n_sampled
    verdict = "unknown"
    if overall_median is not None:
        verdict = "tight" if overall_median <= 4 else "padded"
    return {
        "n_sampled": n_sampled,
        "method": "Otsu binarise, non-background bbox to crop edge",
        "margin_pixels": mp,
        "tight_crop_share": tight_share,
        "verdict": verdict,
    }


def _hyphen_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(entries)
    line_hyp_count = 0
    line_hyp_split: Dict[str, int] = {"U+002D": 0, "U+2013": 0, "U+2014": 0}
    dashes = {"U+002D": 0, "U+2013": 0, "U+2014": 0,
              "U+2212": 0, "U+00AD": 0}
    prefix_any = 0
    prefix_counts: Dict[str, int] = {p: 0 for p in
        ("il-", "is-", "id-", "it-", "l-", "fis-", "bil-")}
    lines_seen = False
    for e in entries:
        text = normalise(_entry_text(e))
        for ch, key in (("-", "U+002D"), ("–", "U+2013"),
                        ("—", "U+2014"), ("−", "U+2212"),
                        ("­", "U+00AD")):
            dashes[key] += text.count(ch)
        lines = _entry_lines(e)
        if lines is not None and _has_line_field(e):
            lines_seen = True
        has_lh = False
        if lines:
            for ln in lines[:-1]:
                s = (ln or "").rstrip()
                if not s:
                    continue
                last = s[-1]
                if last == "-":
                    has_lh = True
                    line_hyp_split["U+002D"] += 1
                elif last == "–":
                    has_lh = True
                    line_hyp_split["U+2013"] += 1
                elif last == "—":
                    has_lh = True
                    line_hyp_split["U+2014"] += 1
        if has_lh:
            line_hyp_count += 1
        if STRUCTURAL_PREFIX.search(text):
            prefix_any += 1
        low = text.lower()
        for p in prefix_counts:
            if re.search(rf"(?:^|[\s\(\[\"]){re.escape(p)}[a-zàèìòùċġħż]", low):
                prefix_counts[p] += 1
    frac = lambda x: (x / n) if n else 0.0
    return {
        "n_paragraphs_scanned": n,
        "line_break_hyphen": {
            "fraction": frac(line_hyp_count) if (lines_seen or any(
                _entry_lines(e) for e in entries)) else "schema_field_absent_in_dev",
            "count": line_hyp_count,
            "split_by_terminator": line_hyp_split,
        },
        "structural_prefix": {
            "fraction_any": frac(prefix_any),
            "count_any": prefix_any,
            "fraction_by_prefix": {p: frac(c) for p, c in prefix_counts.items()},
            "count_by_prefix": prefix_counts,
        },
        "unicode_dashes": dashes,
        "il_prefix_density": frac(prefix_any),
    }


def _language_mix(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"mt": 0, "en": 0, "other": 0, "mixed": 0}
    for e in entries:
        text = normalise(_entry_text(e))
        tag = bucket_language(text)
        if tag == "lang-mt":
            counts["mt"] += 1
        elif tag == "lang-en":
            counts["en"] += 1
        else:
            counts["other"] += 1
        if re.search(r"[A-Za-z]", text) and re.search(r"[ĊċĠġĦħŻż]", text):
            from src.eval.buckets import _english_score as _es
            if _es(text) >= 0.10:
                counts["mixed"] += 1
    return counts


def _length_distribution(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    lens = [len(normalise(_entry_text(e))) for e in entries]
    line_counts: List[int] = []
    for e in entries:
        ls = _entry_lines(e)
        if ls is not None:
            line_counts.append(len(ls))

    def _q(vals: List[int], q: float) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return float(s[i])

    quartiles = compute_quartiles([normalise(_entry_text(e)) for e in entries])
    return {
        "chars_per_paragraph": {
            "min": min(lens) if lens else None,
            "p25": _q(lens, 0.25),
            "median": _q(lens, 0.5),
            "p75": _q(lens, 0.75),
            "p95": _q(lens, 0.95),
            "max": max(lens) if lens else None,
            "quartiles": list(quartiles),
        },
        "lines_per_paragraph": {
            "min": min(line_counts) if line_counts else "schema_field_absent_in_dev",
            "median": _q(line_counts, 0.5) if line_counts else "schema_field_absent_in_dev",
            "max": max(line_counts) if line_counts else "schema_field_absent_in_dev",
        },
    }


def _bucket_distribution(entries: List[Dict[str, Any]],
                         quartiles: Tuple[int, int, int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in entries:
        text = normalise(_entry_text(e))
        lines = _entry_lines(e)
        tags = {
            bucket_length(text, quartiles),
            bucket_language(text),
            bucket_il_prefix(text),
            bucket_line_hyphen(lines),
            bucket_em_dash(text),
            bucket_line_count(lines),
        }
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def _populate_q1(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    inventory = _build_char_inventory(entries)
    counts_by_char: Dict[str, int] = {row["char"]: row["count"] for row in inventory}
    grave = {ch: counts_by_char.get(ch, 0) for ch in GRAVE_VOWELS}
    canaries = {ch: counts_by_char.get(ch, 0) for ch in MALTESE_CANARIES}

    def _ratio(plain: str, diac: str) -> Optional[float]:
        d = counts_by_char.get(diac, 0)
        p = counts_by_char.get(plain, 0)
        if d == 0:
            return None if p == 0 else float("inf")
        return p / d

    expected = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "0123456789 \n\t.,;:!?'\"()[]{}/-")
    expected |= set(MALTESE_CANARIES) | set(GRAVE_VOWELS)
    expected |= {"–", "—", "­", "−", "’", "‘",
                 "“", "”", "«", "»"}
    unexpected = [
        {"codepoint": row["codepoint"], "name": row["name"], "count": row["count"]}
        for row in inventory if row["char"] not in expected
    ]
    return {
        "inventory": inventory,
        "italian_grave_accents_present": grave,
        "maltese_canaries_present": canaries,
        "ascii_fallback_risk": {
            "c_over_c_dot": _ratio("c", "ċ"),
            "g_over_g_dot": _ratio("g", "ġ"),
            "h_over_h_bar": _ratio("h", "ħ"),
            "z_over_z_dot": _ratio("z", "ż"),
        },
        "dash_inventory": {
            "U+002D_hyphen_minus": counts_by_char.get("-", 0),
            "U+2013_en_dash": counts_by_char.get("–", 0),
            "U+2014_em_dash": counts_by_char.get("—", 0),
            "U+2212_minus": counts_by_char.get("−", 0),
            "U+00AD_soft_hyphen": counts_by_char.get("­", 0),
        },
        "unexpected_codepoints": unexpected,
    }


def inspect(dev_root: Path, out_path: Path,
            entry_md_path: Optional[Path] = None,
            summary_md_path: Optional[Path] = None) -> Dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    mode, entries = discover_manifest(dev_root)
    entries = sorted(entries, key=_entry_id)
    n_images = sum(1 for e in entries if _entry_image(e))

    q1 = _populate_q1(entries)
    q2 = _crop_padding_stats(entries, dev_root, rng)
    q3 = _hyphen_stats(entries)
    length = _length_distribution(entries)
    quartiles = tuple(length["chars_per_paragraph"]["quartiles"]) \
        if length["chars_per_paragraph"]["quartiles"] else (80, 200, 400)
    if len(quartiles) != 3:
        quartiles = (80, 200, 400)
    buckets = _bucket_distribution(entries, quartiles)
    lang = _language_mix(entries)

    stats = {
        "_schema_only": False,
        "_produced_by": "src/eval/inspect_dev.py",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_root": str(dev_root).rstrip("/") + "/",
        "manifest_mode": mode,
        "n_paragraphs": len(entries),
        "n_images": n_images,
        "q1_character_set": q1,
        "q2_crop_padding": q2,
        "q3_hyphen_buckets": q3,
        "length_distribution": length,
        "language_mix": lang,
        "bucket_distribution": buckets,
        "notes": [
            "Heuristic: language tagging via diacritic presence + English wordlist hit rate; see src/eval/buckets.py.",
            "Heuristic: crop padding via Otsu binarise + non-background bbox; verdict 'tight' if median margin <= 4 px.",
            "Re-run is deterministic: random sampling uses seed=42; entries sorted by id.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _emit_summary_md(stats, summary_md_path or (out_path.parent / "_summary.md"))
    _emit_entry_md(stats, entry_md_path)
    _assert_populated(stats)
    return stats


def _emit_summary_md(stats: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Dev set summary",
        "",
        f"- n_paragraphs: {stats['n_paragraphs']}",
        f"- n_images: {stats['n_images']}",
        f"- manifest_mode: {stats['manifest_mode']}",
        f"- crop verdict: {stats['q2_crop_padding'].get('verdict')}",
        "",
        "## Bucket counts",
        "",
    ]
    for k, v in stats["bucket_distribution"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Language mix", ""]
    for k, v in stats["language_mix"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Top 30 codepoints", "", "| codepoint | name | count |", "| --- | --- | --- |"]
    for row in stats["q1_character_set"]["inventory"][:30]:
        lines.append(f"| {row['codepoint']} | {row['name']} | {row['count']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_entry_md(stats: Dict[str, Any], out: Optional[Path] = None) -> None:
    if out is None:
        out = Path(__file__).resolve().parents[2] / "outputs" / "dev_inspection_results.md"
    n = stats["n_paragraphs"]
    canaries = stats["q1_character_set"]["maltese_canaries_present"]
    grave = stats["q1_character_set"]["italian_grave_accents_present"]
    body = [
        "---",
        "id: 151",
        "title: Dev inspection results",
        "category: G",
        "status: live",
        "depth: deep",
        "tokens_estimated: 1600",
        f"date: {stats['generated_at'][:10]}",
        "---",
        "",
        "# Headline",
        "",
        f"Dev set contains {n} paragraphs across {stats['n_images']} images (manifest: {stats['manifest_mode']}). Crop verdict: {stats['q2_crop_padding'].get('verdict')}. il-prefix density: {stats['q3_hyphen_buckets']['il_prefix_density']:.3f}.",
        "",
        "# Q1 - Character set",
        "",
        "Maltese canary counts (must all be > 0 for a Maltese dev set):",
        "",
    ]
    for ch, c in canaries.items():
        body.append(f"- `{ch}` (U+{ord(ch):04X}): {c}")
    body += ["", "Italian grave-accent counts:", ""]
    for ch, c in grave.items():
        body.append(f"- `{ch}` (U+{ord(ch):04X}): {c}")
    body += [
        "",
        "Unexpected codepoints (review for noise vs. real signal):",
        "",
    ]
    for row in stats["q1_character_set"]["unexpected_codepoints"][:20]:
        body.append(f"- {row['codepoint']} {row['name']}: {row['count']}")
    body += [
        "",
        "# Q2 - Crop padding",
        "",
        f"Sample n={stats['q2_crop_padding']['n_sampled']}. Tight share: {stats['q2_crop_padding'].get('tight_crop_share')}.",
        "Margins (px): " + json.dumps(stats["q2_crop_padding"]["margin_pixels"], ensure_ascii=False),
        "",
        "Implication: if verdict is 'tight', drop column-edge augmentation; if 'padded', keep it.",
        "",
        "# Q3 - Hyphens and il-prefix",
        "",
        f"Line-break hyphen fraction: {stats['q3_hyphen_buckets']['line_break_hyphen']['fraction']}",
        f"Structural prefix (any) fraction: {stats['q3_hyphen_buckets']['structural_prefix']['fraction_any']:.3f}",
        f"Unicode dashes: {stats['q3_hyphen_buckets']['unicode_dashes']}",
        "",
        "# Bucket distribution",
        "",
    ]
    for k, v in stats["bucket_distribution"].items():
        body.append(f"- {k}: {v}")
    body += [
        "",
        "# Synth distribution gap",
        "",
        "Compare these dev-side counts against the current synth shard mix. Variants likely to win: those that target buckets where dev share exceeds synth share (e.g. high il-prefix density argues for joiner-aware decoder; high em-dash-yes argues for the dash-preserving normalisation).",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n", encoding="utf-8")


def _assert_populated(stats: Dict[str, Any]) -> None:
    def walk(obj: Any, path: str) -> None:
        if obj is None:
            raise AssertionError(f"unpopulated field at {path}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("_"):
                    continue
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(stats, "$")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    inspect(args.dev_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
