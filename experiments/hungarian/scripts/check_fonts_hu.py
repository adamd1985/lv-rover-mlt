"""Validate every font in the existing (Maltese-validated) font pool against
the Hungarian diacritic canary set. Same check as check_fonts.py, retargeted
glyph set - no logic changes, only which characters are required."""
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"


def check_font(path: Path):
    from fontTools.ttLib import TTFont
    try:
        font = TTFont(str(path), lazy=True)
    except Exception as e:
        return False, [f"open failed: {e}"]
    cmap = font.getBestCmap()
    missing = [c for c in REQUIRED if ord(c) not in cmap]
    return (len(missing) == 0), missing


def main() -> int:
    root = Path("/home/adamd1985/doceng2026/data/fonts")
    bad = []
    good = []
    total = 0
    for p in sorted(list(root.rglob("*.ttf")) + list(root.rglob("*.otf"))):
        if "_rejected" in str(p):
            continue
        total += 1
        ok, missing = check_font(p)
        if ok:
            good.append(p)
        else:
            bad.append((p, missing))
    print(f"{len(good)} / {total} fonts pass Hungarian canary check.")
    if bad:
        print(f"\n{len(bad)} fonts fail (missing glyphs):")
        for p, missing in bad:
            print(f"  {p.name}: missing {''.join(missing)}")
    with open("data/hu_validated_fonts.txt", "w") as f:
        for p in good:
            f.write(str(p) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
