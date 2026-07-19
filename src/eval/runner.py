"""Top-level eval harness.

`run_eval(predictions_jsonl, gold_jsonl, organiser_script_path=None)` is the
entrypoint. JSONL schemas:

  predictions: one object per line with keys `{id, hypothesis}`
  gold:        one object per line with keys `{id, paragraph|text, as_lines?|lines?}`

Both files are matched by `id`. Missing ids are skipped (and counted).

Organiser-script cross-validation: when `organiser_script_path` is given we
load it as a Python module via importlib and call its `run(predictions, gold)`
entrypoint. The kill criterion is `abs(internal_cer - organiser_cer) > 1e-3`
per the design decision. We raise RuntimeError on disagreement; the design decision status stays in the
'pending audit' state until this passes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import audit, buckets, cer, report


D14_TOL = 1e-3


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pair_by_id(preds: List[dict], gold: List[dict]) -> Tuple[List[str], List[str], List[List[str]], List[str], List[str]]:
    pred_by_id = {p["id"]: p for p in preds}
    gold_by_id = {g["id"]: g for g in gold}
    common = [i for i in gold_by_id if i in pred_by_id]
    missing = [i for i in gold_by_id if i not in pred_by_id]
    refs = [gold_by_id[i].get("paragraph", gold_by_id[i].get("text", "")) for i in common]
    hyps = [pred_by_id[i].get("hypothesis", "") for i in common]
    line_lists = [
        list(gold_by_id[i].get("as_lines") or gold_by_id[i].get("lines") or [])
        for i in common
    ]
    return refs, hyps, line_lists, common, missing


def _load_organiser_module(path: Path):
    spec = importlib.util.spec_from_file_location("organiser_eval", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load organiser eval module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["organiser_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_eval(
    predictions_jsonl: Path,
    gold_jsonl: Path,
    organiser_script_path: Optional[Path] = None,
    *,
    out_dir: Optional[Path] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    preds = _read_jsonl(predictions_jsonl)
    gold = _read_jsonl(gold_jsonl)
    refs, hyps, line_lists, common, missing = _pair_by_id(preds, gold)

    quartiles = buckets.compute_quartiles(refs)
    bucket_tags = buckets.tag_corpus(refs, line_lists, quartiles)

    aggregate = cer.cer_organiser(refs, hyps)
    dual = cer.compute_cer_dual(refs, hyps)
    by_bucket = cer.cer_per_bucket(refs, hyps, bucket_tags)
    boot = audit.bootstrap_cer(refs, hyps, n_boot=n_boot, seed=seed)
    boot_buckets = audit.bootstrap_per_bucket(refs, hyps, bucket_tags, n_boot=n_boot, seed=seed)
    canaries = cer.cer_on_char_subset(refs, hyps, audit.CANARY_CHARS)
    confusion = cer.confusion_matrix(refs, hyps)

    result: dict = {
        "n_pairs": len(common),
        "n_missing": len(missing),
        "missing_ids": missing[:20],
        "quartiles": list(quartiles),
        "aggregate_cer": aggregate,
        "aggregate_cer_dual": dual,
        "aggregate_bootstrap": boot,
        "per_bucket": by_bucket,
        "per_bucket_bootstrap": boot_buckets,
        "canary_cer": canaries,
        "top_confusions": [
            {"ref": r, "hyp": h, "count": c}
            for (r, h), c in confusion.most_common(50)
        ],
        "organiser": None,
    }

    if organiser_script_path is not None:
        org = _load_organiser_module(Path(organiser_script_path))
        if not hasattr(org, "run"):
            raise RuntimeError("organiser eval module has no 'run' entrypoint")
        org_cer = float(org.run(predictions=predictions_jsonl, gold=gold_jsonl))
        delta = abs(org_cer - aggregate)
        result["organiser"] = {
            "cer": org_cer,
            "delta_vs_internal": delta,
            "passes_d14": delta <= D14_TOL,
        }
        if delta > D14_TOL:
            raise RuntimeError(
                f"the design decision kill criterion: organiser CER {org_cer:.6f} vs "
                f"internal {aggregate:.6f}, delta {delta:.6f} > {D14_TOL}"
            )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.render(result, out_dir / "report.md", out_dir / "report.json")
    return result


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--organiser", type=Path, default=None,
                     help="path to organiser eval Python module (with a `run` function)")
    ap.add_argument("--out", type=Path, default=Path("submissions/last_eval"))
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    args = ap.parse_args()
    result = run_eval(
        args.predictions, args.gold,
        organiser_script_path=args.organiser,
        out_dir=args.out,
        n_boot=args.bootstrap_samples,
    )
    print(json.dumps({"aggregate_cer": result["aggregate_cer"],
                       "n_pairs": result["n_pairs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
