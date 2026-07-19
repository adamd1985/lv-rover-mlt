![LV-ROVER banner](rover_mt_banner.png)

# lv-rover-mlt

OCR for Maltese paragraph images from PDF documents. Submitted to the [DocEng 2026 Maltese OCR competition](https://doceng.org/doceng2026/maltese-ocr).

Model weights and data files: https://huggingface.co/radmada/lv-rover-mlt

Paper: *LV-ROVER-MLT: Low-Resource Maltese OCR by Fine-Tuning and Multi-Stream Arbitration* - [preprint on arXiv](https://arxiv.org/abs/2607.00250).

Dev set CER: **0.00700** on 422 paragraphs - roughly 70% below the competition's published Tesseract baseline (0.0234). That gain breaks down as: 44% from recognition improvements (fine-tuned Tesseract ensemble), plus 26 percentage points from aligning to the gold label convention (curly quotes, em-dash). See the paper for the decomposition.

---

## How it works

The core idea is that multiple Tesseract passes, each using a slightly different language chain or image scale, make different errors on the same image. Running five of them and arbitrating at the word level under a lexicon-gated plurality rule (no quorum required) recovers most of those errors. Tesseract 5 itself is an LSTM recognizer; the arbitration step does not add a second neural network on top of it.

The five streams are: `mlt`, `mlt+ita` (anchor), `mlt+ita+fra`, stock Maltese, and `mlt+ita` on a 2x upscaled image. A confusion table built from synthetic Maltese text handles systematic Tesseract character substitutions (mainly the four diacritics - ċ ġ ħ ż).

Post-processing normalises line-break hyphens (removed and word rejoined), structural hyphens (kept), en-dash to em-dash, and curly quotes, to match the gold convention.

No GPU needed.

---

## Setup

You need Tesseract 5.x installed on the system. Get it from:

- Windows: https://github.com/UB-Mannheim/tesseract/wiki (add to PATH)
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

Then install the Python packages. Install this file **first**, before the organiser's `requirements.txt`:

```
pip install -r requirements.txt
```

---

## Usage

```python
from competition_transcriber import CompetitionTranscriber
import PIL.Image

# First run downloads model files (~260 MB) from HuggingFace.
t = CompetitionTranscriber()

text = t.transcribe(PIL.Image.open("paragraph.jpg"))
print(text)
```

`transcribe` returns a single joined paragraph string. It does not access the network.

---

## For competition judges

Use the `competition_transcriber.py` from **this GitHub repo** — not the copy on the
HuggingFace model page. The GitHub copy carries a Windows path fix (it sets
`TESSDATA_PREFIX` instead of passing `--tessdata-dir "…"` in the Tesseract config
string, which `pytesseract` mangles via `shlex.split` on Windows). The model weights
still download from HuggingFace automatically; only the script differs.

```python
from competition_transcriber import CompetitionTranscriber
t = CompetitionTranscriber()
text = t.transcribe(pil_image)
```

Environment notes for the evaluation machine (Windows 11, Python 3.9, Anaconda):

- **Tesseract 5.x** must be installed. It does not need to be on PATH — the loader
  finds it on PATH or at the standard `C:\Program Files\Tesseract-OCR\tesseract.exe`
  location. (Override with the `TESSERACT_CMD` env var if it lives elsewhere.)
- **Install order**: install this repo's `requirements.txt` *before* the organiser's.
  The pins do not conflict (`malti==0.3.1` pulls only `sentence-splitter==1.4`).
- On first `__init__`, all model files (tessdata, lexicon, confusion table, ~260 MB)
  download from HuggingFace. **Nothing downloads during `transcribe`.**

---

## Using the organiser's dev-set assets

`OCR competition assets for participants/` (organiser-provided, not part of this
repo's own code) holds the dev-set images + gold labels, the official evaluator, and
a placeholder `competition_transcriber.py` template that's meant to be replaced with
a real submission — this repo's `competition_transcriber.py` is that submission.

To evaluate it against the dev set with the organiser's own evaluator:

```
cp competition_transcriber.py "OCR competition assets for participants/competition_transcriber.py"
cd "OCR competition assets for participants"
pip install -r requirements.txt   # organiser's requirements (evaluate, transformers, torch)
python competition_evaluator.py dev
```

This prints CER and runtime, and appends a row to `results.txt`.

---

## What's in the HuggingFace repo

| File | Description |
|---|---|
| `competition_transcriber.py` | The OCR class |
| `tessdata/mlt.traineddata` | Fine-tuned Tesseract LSTM (50k synthetic Maltese lines) |
| `tessdata/mltstock.traineddata` | Stock Tesseract Maltese (independent error stream) |
| `tessdata/ita.traineddata` | Italian chain (used in mlt+ita and mlt+ita+fra) |
| `tessdata/fra.traineddata` | French chain (mlt+ita+fra stream) |
| `tess_confusion.json` | Character confusion correction table |
| `maltese_en_it_lexicon.json` | Lexicon for vote arbitration |

The `competition_transcriber.py` on HuggingFace is the frozen deadline submission and
fails on Windows (see [For competition judges](#for-competition-judges)). Run the copy
from this GitHub repo instead; the weights above are unchanged and still load from HuggingFace.

---

## Training data

Synthetic only. 50,000 line images rendered from the MLRS Korpus Malti v4.2 corpus using 68 fonts covering the full Maltese diacritic set. Augmentations matched to real PDF crops: JPEG re-encoding at quality ~72, slight rotation, mild blur, brightness variation, paragraph width and font size variation. No real document images were used.
