"""Push the v20 Tesseract LV-ROVER submission bundle to HuggingFace.

Requires HF_TOKEN with `write` permission in `.env` or the environment.
Uploads everything `competition_transcriber.py` (v20) resolves at init:

    tessdata/mlt.traineddata       fine-tuned LSTM (anchor + mlt candidate)
    tessdata/mltstock.traineddata  stock LSTM (independent error stream, v13)
    tessdata/ita.traineddata       for mlt+ita anchor and mlt+ita+fra candidate
    tessdata/fra.traineddata       for mlt+ita+fra candidate (v12)
    tess_confusion.json            confusion corrector lookup table
    maltese_en_it_lexicon.json     router + corrector lexicon

The default repo must match HF_REPO_DEFAULT in competition_transcriber.py
(radmada/lv-rover-mlt) or be overridden there.

Run after regenerating the HF token with write scope:
    python scripts/push_submission_bundle.py --repo-id radmada/lv-rover-mlt
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (local path, path in repo) - matches the wrapper's resolve/load logic.
BUNDLE = [
    ("data/tesseract/tessdata/mlt.traineddata", "tessdata/mlt.traineddata"),
    ("data/tesseract/tessdata/mltstock.traineddata", "tessdata/mltstock.traineddata"),
    ("data/tesseract/tessdata/ita.traineddata", "tessdata/ita.traineddata"),
    ("data/tesseract/tessdata/fra.traineddata", "tessdata/fra.traineddata"),
    ("data/tess_confusion.json", "tess_confusion.json"),
    ("data/maltese_en_it_lexicon.json", "maltese_en_it_lexicon.json"),
    ("competition_transcriber.py", "competition_transcriber.py"),
    ("rover_mt_banner.png", "rover_mt_banner.png"),
]


def _load_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in env or .env file")


def main() -> None:
    from huggingface_hub import HfApi, create_repo

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="radmada/lv-rover-mlt")
    ap.add_argument("--dry-run", action="store_true", help="verify files, do not upload")
    args = ap.parse_args()

    missing = [p for p, _ in BUNDLE if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"missing bundle files: {missing}")
    total_mb = sum((ROOT / p).stat().st_size for p, _ in BUNDLE) / 1e6
    print(f"[push] bundle: {len(BUNDLE)} files, {total_mb:.1f} MB")
    for p, dst in BUNDLE:
        print(f"        {p}  ->  {dst}")
    if args.dry_run:
        print("[push] dry-run: not uploading")
        return

    token = _load_token()
    api = HfApi(token=token)
    me = api.whoami()
    print(f"[push] HF user: {me.get('name')}")

    create_repo(repo_id=args.repo_id, token=token, repo_type="model", exist_ok=True)
    print(f"[push] repo ready: {args.repo_id}")

    for local, dst in BUNDLE:
        api.upload_file(
            path_or_fileobj=str(ROOT / local),
            path_in_repo=dst,
            repo_id=args.repo_id, token=token,
            commit_message=f"Add {dst}",
        )
        print(f"[push] uploaded {dst}")

    readme = textwrap.dedent("""
    ---
    license: apache-2.0
    language: mt
    tags: ['ocr', 'tesseract', 'maltese', 'doceng2026']
    ---

    ![LV-ROVER banner](rover_mt_banner.png)

    # lv-rover-mlt

    OCR for cropped paragraph images from Maltese PDF documents. Takes a PIL
    image, returns a clean joined paragraph string. Handles the full Maltese
    alphabet including ċ ġ ħ ż għ, structural hyphens (il-kelb), and
    line-break rejoining.

    Submitted to the [DocEng 2026 Maltese OCR competition](https://doceng.org/doceng2026/maltese-ocr).
    Dev set CER: **0.00700** on 422 paragraphs - 70% below the
    competition's published Tesseract baseline (0.0234).

    No neural models. No GPU needed.

    ## How it works

    Five independent Tesseract passes run over each image using different
    language chain combinations and image scales. Each pass makes different
    mistakes. A lexicon-anchored plurality vote (LV-ROVER) then picks the best
    reading word by word. A confusion corrector handles systematic Tesseract
    errors on Maltese characters. Finally the output is normalised to match
    Maltese typographic conventions (em-dashes, curly quotes, soft-hyphen
    removal at line breaks).

    ## Setup

    **Step 1 - Install Tesseract 5.x**

    Windows: get the installer from the UB Mannheim builds at
    https://github.com/UB-Mannheim/tesseract/wiki and make sure `tesseract`
    is on your PATH.

    Linux:
    ```bash
    sudo apt install tesseract-ocr
    ```

    macOS:
    ```bash
    brew install tesseract
    ```

    **Step 2 - Install Python packages**

    ```bash
    pip install pytesseract malti pillow huggingface_hub
    ```

    **Step 3 - Get the script**

    ```bash
    huggingface-cli download radmada/lv-rover-mlt competition_transcriber.py --local-dir .
    ```

    This saves `competition_transcriber.py` into your current directory.
    That is the only file you need to copy - all model files (tessdata,
    lexicon, confusion table) download automatically on first use.

    ## Usage

    ```python
    from competition_transcriber import CompetitionTranscriber
    import PIL.Image

    # First run downloads ~260 MB of model files from HuggingFace.
    # Subsequent runs use the local cache.
    transcriber = CompetitionTranscriber()

    image = PIL.Image.open("your_paragraph.jpg")
    text = transcriber.transcribe(image)
    print(text)
    ```

    ## For DocEng 2026 competition judges

    The evaluator imports this class directly:

    ```python
    from competition_transcriber import CompetitionTranscriber
    transcriber = CompetitionTranscriber()
    text = transcriber.transcribe(pil_image)
    ```

    Place `competition_transcriber.py` (from this repo) alongside the
    organiser's `competition_evaluator.py` and run as normal. Tesseract 5.x
    must be on PATH. All model files download automatically from this repo on
    first call to `__init__`. No network access happens during `transcribe`.

    ## Files in this repo

    | File | What it is |
    |---|---|
    | `competition_transcriber.py` | The OCR class - this is all you need to copy |
    | `tessdata/mlt.traineddata` | Fine-tuned Tesseract LSTM for Maltese (50k synthetic lines) |
    | `tessdata/mltstock.traineddata` | Stock Tesseract Maltese model (independent error stream) |
    | `tessdata/ita.traineddata` | Italian language chain (anchor stream) |
    | `tessdata/fra.traineddata` | French language chain (third candidate stream) |
    | `tess_confusion.json` | Character-level confusion correction table |
    | `maltese_en_it_lexicon.json` | Maltese/English/Italian lexicon for vote arbitration |

    ## Training data

    Synthetic only. 50,000 line images rendered from the MLRS Korpus Malti
    v4.2 corpus using 30+ fonts that correctly render Maltese diacritics,
    with PDF-realistic augmentations (JPEG re-encoding, slight rotation, mild
    blur, brightness variation). No real document images were used in training.

    """).strip()
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id, token=token,
        commit_message="Add README",
    )
    print("[push] uploaded README")

    info = api.repo_info(args.repo_id, repo_type="model")
    print(f"[push] siblings: {[s.rfilename for s in info.siblings]}")


if __name__ == "__main__":
    main()
