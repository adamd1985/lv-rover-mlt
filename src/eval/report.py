"""Render eval results to markdown + JSON.

JSON is what the eval-runner subagent reads. Markdown is paper-facing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def render(result: dict, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(result), encoding="utf-8")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        if x != x:  # NaN
            return "nan"
    except TypeError:
        return str(x)
    return f"{x:.4f}"


def _to_markdown(r: dict) -> str:
    lines = []
    lines.append("# Eval report")
    lines.append("")
    lines.append(f"n_pairs: {r.get('n_pairs')}  n_missing: {r.get('n_missing')}")
    lines.append("")
    agg = r.get("aggregate_cer")
    boot = r.get("aggregate_bootstrap", {}) or {}
    lines.append(f"aggregate CER: {_fmt(agg)}  "
                  f"(95% CI {_fmt(boot.get('ci_lo'))} - {_fmt(boot.get('ci_hi'))})")
    dual = r.get("aggregate_cer_dual") or {}
    if dual:
        lines.append(
            f"raw CER (leaderboard): {_fmt(dual.get('raw_cer'))}  "
            f"normalised CER (internal signal): {_fmt(dual.get('normalised_cer'))}  "
            f"delta: {_fmt(dual.get('delta'))}"
        )
    q = r.get("quartiles")
    if q:
        lines.append(f"length quartiles: {q}")
    lines.append("")

    org = r.get("organiser")
    if org:
        lines.append("## Organiser cross-check")
        lines.append(f"organiser CER: {_fmt(org.get('cer'))}  "
                      f"delta: {_fmt(org.get('delta_vs_internal'))}  "
                      f"organiser pass: {org.get('passes_d14')}")
        lines.append("")

    lines.append("## Per-bucket CER")
    lines.append("")
    lines.append("| bucket | n | CER | 95% CI | small |")
    lines.append("|---|---:|---:|---|---|")
    pb = r.get("per_bucket", {}) or {}
    pbb = r.get("per_bucket_bootstrap", {}) or {}
    for k in sorted(pb.keys()):
        row = pb[k]
        b = pbb.get(k, {})
        ci = f"{_fmt(b.get('ci_lo'))} - {_fmt(b.get('ci_hi'))}"
        small = "yes" if row.get("small") else ""
        lines.append(f"| {k} | {row.get('n')} | {_fmt(row.get('cer'))} | {ci} | {small} |")
    lines.append("")

    lines.append("## Canary CER (subset CER on paragraphs containing the char)")
    lines.append("")
    lines.append("| char | CER |")
    lines.append("|---|---:|")
    for ch, c in (r.get("canary_cer") or {}).items():
        lines.append(f"| `{ch}` (U+{ord(ch):04X}) | {_fmt(c)} |")
    lines.append("")

    lines.append("## Top confusions")
    lines.append("")
    lines.append("| ref | hyp | count |")
    lines.append("|---|---|---:|")
    for c in (r.get("top_confusions") or [])[:20]:
        lines.append(f"| `{c['ref']}` | `{c['hyp']}` | {c['count']} |")
    lines.append("")

    return "\n".join(lines) + "\n"
