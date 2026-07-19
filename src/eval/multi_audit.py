"""Multi-variant audit driver.

Reads a manifest of (variant_name, predictions_path) entries, runs
:func:`audit_variant` per row against a shared baseline, and emits a
leaderboard sorted by normalised-CER delta. Filtering rule: only entries
whose 95 percent CI excludes zero are flagged on the leaderboard; the rest
are listed with verdict INSIDE NOISE for full traceability.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from .audit_runner import AuditReport, audit_variant


def load_manifest(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    n_bootstrap: int = 1000,
    n_shuffle: int = 1000,
    cv_folds: int = 5,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> List[AuditReport]:
    m = load_manifest(Path(manifest_path))
    baseline = Path(m["baseline"]["predictions"])
    gold = Path(m["gold"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[AuditReport] = []
    for v in m.get("variants", []):
        name = v["name"]
        pred = Path(v["predictions"])
        if not pred.exists():
            continue
        rep = audit_variant(
            baseline, pred, gold, out_dir,
            variant_name=name,
            n_bootstrap=n_bootstrap,
            n_shuffle=n_shuffle,
            cv_folds=cv_folds,
            fdr_alpha=fdr_alpha,
            seed=seed,
        )
        reports.append(rep)
    write_leaderboard(reports, out_dir / "_leaderboard.md")
    return reports


def _sort_key(rep: AuditReport) -> float:
    return rep.delta_normalised


def write_leaderboard(reports: Sequence[AuditReport], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(reports, key=_sort_key)
    lines: List[str] = []
    lines.append("# Variant audit leaderboard")
    lines.append("")
    lines.append("Sorted by normalised-CER delta (most improvement first). "
                 "Sign convention: delta = variant - baseline.")
    lines.append("")
    lines.append("| rank | variant | delta | 95% CI | shuffle p | verdict |")
    lines.append("|---:|---|---:|---|---:|---|")
    for i, r in enumerate(ranked, start=1):
        lo, hi = r.delta_ci
        lines.append(
            f"| {i} | {r.variant_name} | {r.delta_normalised:.4f} | "
            f"{lo:.4f} - {hi:.4f} | {r.shuffle_p:.4f} | {r.verdict} |"
        )
    lines.append("")
    keepers = [r for r in ranked if r.verdict == "KEEP"]
    lines.append(f"KEEP count: {len(keepers)}")
    if keepers:
        lines.append("")
        lines.append("Promoted variants:")
        for r in keepers:
            lines.append(f"- {r.variant_name} (delta {r.delta_normalised:.4f})")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="multi_audit")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-shuffle", type=int, default=1000)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    reports = run_manifest(
        a.manifest, a.out,
        n_bootstrap=a.n_bootstrap,
        n_shuffle=a.n_shuffle,
        cv_folds=a.cv_folds,
        fdr_alpha=a.fdr_alpha,
        seed=a.seed,
    )
    for r in sorted(reports, key=_sort_key):
        print(f"{r.variant_name}: delta={r.delta_normalised:.4f} "
              f"verdict={r.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
