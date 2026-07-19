"""Stratified eval over the dev set.

Dev format (organiser, `competition_files/dev/`): a single `texts.json` list
plus cropped paragraph JPGs. Each entry has `image` (filename), `text`
(paragraph-form gold, used for CER) and `as_lines` (the line strings as in the
PDF, including hyphens). Note the key is `as_lines`, not `lines`.

JSON loader is tolerant of a few shapes:
  - top-level list of {image, text, as_lines}
  - top-level dict keyed by image filename -> {text, as_lines}
  - any of `image|file|filename`, `text|paragraph|gt`, `as_lines|lines`

Buckets:
  - length: short / mid / long by char count
  - language: malti / english / other-lang (heuristic)
  - prefix: has `il-` style article hyphen vs none
  - line-hyphen: any raw line ends in `-` vs none
  - em-dash: contains em-dash U+2014 vs not (gold has no en-dash)
  - line count: single-line vs multi-line (from `as_lines`)

Outputs aggregate CER, per-bucket CER, char substitution matrix focused on the
diacritic confusions and dash confusions.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEV_DIR = Path("competition_files/dev")
DIACRITICS = "ċĊġĠħĦżŻ"
EM_DASH = "—"
ARTICLE_PREFIXES = re.compile(r"\b([ilms]|il|id|is|in|ir|it|ix|iz)-", re.IGNORECASE)


def _pick(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_dev(dev_dir: Path, json_path: Path = None) -> List[Tuple[Path, str, List[str]]]:
    """Returns [(image_path, paragraph_gt, raw_lines), ...]."""
    if json_path is None:
        candidates = list(dev_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no JSON file under {dev_dir}")
        json_path = candidates[0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        entries = [(k, v) for k, v in payload.items()]
    elif isinstance(payload, list):
        entries = []
        for row in payload:
            name = _pick(row, ("image", "file", "filename", "img", "path"))
            entries.append((name, row))
    else:
        raise ValueError(f"unsupported JSON shape in {json_path}")

    out: List[Tuple[Path, str, List[str]]] = []
    for name, row in entries:
        if name is None:
            continue
        gt = _pick(row, ("text", "paragraph", "gt", "transcription"))
        lines = _pick(row, ("as_lines", "lines", "raw_lines", "line_list")) or []
        img_path = dev_dir / name
        if not img_path.exists() and not str(name).endswith((".png", ".jpg", ".jpeg")):
            img_path = dev_dir / f"{name}.jpg"
        out.append((img_path, gt or "", list(lines)))
    return out


def bucket_length(s: str) -> str:
    n = len(s)
    if n < 80:
        return "short"
    if n < 200:
        return "mid"
    return "long"


def bucket_language(s: str) -> str:
    if any(c in DIACRITICS for c in s):
        return "malti"
    if re.search(r"[A-Za-z]", s):
        return "english"
    return "other-lang"


def bucket_prefix(s: str) -> str:
    return "il-prefix" if ARTICLE_PREFIXES.search(s) else "no-prefix"


def bucket_line_hyphen(lines: List[str]) -> str:
    for ln in lines[:-1]:
        if ln.rstrip().endswith("-"):
            return "line-hyphen"
    return "no-line-hyphen"


def bucket_em_dash(s: str) -> str:
    return "em-dash" if EM_DASH in s else "no-em-dash"


def bucket_line_count(lines: List[str]) -> str:
    return "multi-line" if lines and len(lines) > 1 else "single-line"


def cer(ref: str, hyp: str) -> float:
    import jiwer
    if not ref:
        return 0.0
    return float(jiwer.cer(ref, hyp))


def confusion(ref: str, hyp: str, mat: Counter) -> None:
    for r, h in zip(ref, hyp):
        if r != h:
            mat[(r, h)] += 1


def run(config: Path, dev_dir: Path, report_path: Path) -> int:
    from PIL import Image as _Image

    from competition_transcriber import CompetitionTranscriber

    ocr = CompetitionTranscriber()
    samples = load_dev(dev_dir)
    if not samples:
        print(f"no dev samples found under {dev_dir}")
        return 1

    per_bucket: Dict[str, List[float]] = defaultdict(list)
    confusion_mat: Counter = Counter()
    total: List[float] = []

    for img_path, ref, lines in samples:
        if not img_path.exists():
            print(f"missing image: {img_path}")
            continue
        hyp = ocr.transcribe(_Image.open(img_path).convert("RGB"))
        c = cer(ref, hyp)
        total.append(c)
        for tag in (
            bucket_length(ref),
            bucket_language(ref),
            bucket_prefix(ref),
            bucket_line_hyphen(lines),
            bucket_em_dash(ref),
            bucket_line_count(lines),
        ):
            per_bucket[tag].append(c)
        confusion(ref, hyp, confusion_mat)

    report = {
        "n": len(total),
        "cer_mean": (sum(total) / len(total)) if total else None,
        "by_bucket": {
            k: {"n": len(v), "cer_mean": sum(v) / len(v), "small": len(v) < 20}
            for k, v in per_bucket.items() if v
        },
        "top_confusions": confusion_mat.most_common(50),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dev", default=str(DEV_DIR))
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    return run(Path(args.config), Path(args.dev), Path(args.report))


if __name__ == "__main__":
    raise SystemExit(main())
