"""Validate the font pool against Luxembourgish's canary set."""
from pathlib import Path

REQUIRED = "àâäæçèéêëîïòóôöûüœÈ"


def check_font(path: Path):
    from fontTools.ttLib import TTFont
    try:
        font = TTFont(str(path), lazy=True)
    except Exception as e:
        return False, [f"open failed: {e}"]
    cmap = font.getBestCmap()
    missing = [c for c in REQUIRED if ord(c) not in cmap]
    return (len(missing) == 0), missing


def main():
    root = Path("/home/adamd1985/doceng2026/data/fonts")
    good, bad = [], []
    total = 0
    for p in sorted(list(root.rglob("*.ttf")) + list(root.rglob("*.otf"))):
        if "_rejected" in str(p):
            continue
        total += 1
        ok, missing = check_font(p)
        (good if ok else bad).append((p, missing))
    print(f"{len(good)} / {total} fonts pass Luxembourgish canary check.")
    for p, missing in bad:
        print(f"  {p.name}: missing {''.join(missing)}")
    with open("data/ltz_validated_fonts.txt", "w") as f:
        for p, _ in good:
            f.write(str(p) + "\n")


if __name__ == "__main__":
    main()
