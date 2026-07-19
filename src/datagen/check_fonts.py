"""Validate every font in data/fonts renders the Maltese diacritics.

Fonts that silently substitute `c` for `ċ` (etc.) are the single biggest synth
bug. This script must pass before any rendering job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED = "ĊċĠġĦħŻżÀàÈè"


def check_font(path: Path) -> Tuple[bool, List[str]]:
    from fontTools.ttLib import TTFont
    try:
        font = TTFont(str(path), lazy=True)
    except Exception as e:
        return False, [f"open failed: {e}"]
    cmap = font.getBestCmap()
    missing = [c for c in REQUIRED if ord(c) not in cmap]
    return (len(missing) == 0), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", required=True)
    args = ap.parse_args()
    root = Path(args.fonts_dir)
    if not root.exists():
        print(f"fonts dir not found: {root}", file=sys.stderr)
        return 2
    bad: List[Path] = []
    total = 0
    for p in sorted(list(root.glob("*.ttf")) + list(root.glob("*.otf"))):
        total += 1
        ok, missing = check_font(p)
        flag = "OK" if ok else f"MISSING {''.join(missing)}"
        print(f"{flag:30s} {p.name}")
        if not ok:
            bad.append(p)
    print(f"\n{total - len(bad)} / {total} fonts pass.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
