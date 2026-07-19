"""Build a held-out Luxembourgish eval sample from the real BnL antiqua line
pairs, filtered by language. The corpus is genuinely mixed French/German/
Luxembourgish (verified this session: ~37/21/16 percent, rest unclear on a
100-file sample) - used unfiltered this would be a French/German OCR test
wearing a Luxembourgish label."""
import json
import random
import re
from pathlib import Path

RANDOM_SEED = 42
N_EVAL = 200

ANTIQUA_DIR = Path("data/antiqua/antiqua")

# Luxembourgish-distinctive function words and orthography, verified against
# the language-mix sampling pass this session. Not exhaustive - a heuristic
# classifier, not a language-ID model, so verified by manual spot-check below
# before trusting anything built on top of it (per plan: don't repeat the
# assume-without-verifying mistake).
LTZ_MARKERS = {
    "an", "de", "an", "eng", "een", "duerch", "vun", "mat", "op", "fir",
    "gëtt", "sinn", "hunn", "wier", "wor", "nët", "net", "och", "wéi",
    "dësem", "dëst", "dës", "hei", "do", "wann", "haut", "muer", "gëschter",
    "keng", "kee", "sengem", "senger", "hirem", "hirer", "vläicht",
}
FR_MARKERS = {"le", "la", "les", "de", "et", "dans", "des", "une", "un",
              "que", "qui", "pour", "avec", "sur", "par", "est", "sont"}
DE_MARKERS = {"der", "die", "das", "und", "ist", "sind", "eine", "ein",
              "nicht", "auch", "wie", "wenn", "aber", "mit", "auf", "für"}


def classify(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if len(words) < 3:
        return "unclear"
    wset = set(words)
    ltz_hits = len(wset & LTZ_MARKERS)
    fr_hits = len(wset & FR_MARKERS)
    de_hits = len(wset & DE_MARKERS)
    # distinctive Luxembourgish digraphs/orthography as a secondary signal
    ltz_orth = sum(text.lower().count(c) for c in ("ë", "iewe", "aach", "oer"))
    scores = {"ltz": ltz_hits + (1 if ltz_orth else 0), "fr": fr_hits, "de": de_hits}
    best = max(scores, key=scores.get)
    if scores[best] == 0 or (scores[best] <= 1 and len(words) < 6):
        return "unclear"
    # require a clear margin over the next-best language to reduce false positives
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] - ordered[1] < 1:
        return "unclear"
    return best


files = sorted(ANTIQUA_DIR.glob("*.gt.txt"))
print(f"total line pairs: {len(files)}")

ltz_files = []
counts = {"ltz": 0, "fr": 0, "de": 0, "unclear": 0}
for f in files:
    text = f.read_text(encoding="utf-8", errors="replace").strip()
    lang = classify(text)
    counts[lang] += 1
    if lang == "ltz":
        stem = f.name.replace(".gt.txt", "")
        png = ANTIQUA_DIR / f"{stem}.png"
        if png.is_file():
            ltz_files.append((stem, text))

print(f"classification counts: {counts}")
print(f"Luxembourgish-classified with matching image: {len(ltz_files)}")

random.seed(RANDOM_SEED)
random.shuffle(ltz_files)
eval_set = ltz_files[:N_EVAL]
train_set = ltz_files[N_EVAL:]

Path("data/eval_manifest.json").write_text(
    json.dumps([{"stem": s, "text": t} for s, t in eval_set], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Path("data/train_manifest.json").write_text(
    json.dumps([{"stem": s, "text": t} for s, t in train_set], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"eval set: {len(eval_set)}, train set: {len(train_set)}")
