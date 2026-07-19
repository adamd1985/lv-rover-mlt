![LV-ROVER banner](rover_mt_banner.png)

# lv-rover-mlt

OCR for Maltese paragraph images from PDF documents. Submitted to the [DocEng 2026 Maltese OCR competition](https://www.um.edu.mt/projects/nomocrat/doceng26competition/).

Paper: *LV-ROVER-MLT: Low-Resource Maltese OCR by Synthetic Fine-Tuning and Multi-Stream Arbitration* ([arXiv preprint](https://arxiv.org/abs/2607.00250)).

| Artifact | Location |
|---|---|
| Weights, lexicon, confusion table | [huggingface.co/radmada/lv-rover-mlt](https://huggingface.co/radmada/lv-rover-mlt) |
| Released Maltese OCR corpus, 36,803 pairs | [huggingface.co/datasets/radmada/maltese-ocr-corpus](https://huggingface.co/datasets/radmada/maltese-ocr-corpus) |

## Results

Development set, 422 paragraphs, scored with the organisers' own scorer at seed 42. CPU only.

| Stage | CER |
|---|---|
| Organisers' fine-tuned Tesseract baseline | 0.0234 |
| Our fine-tuned recogniser, anchor stream | 0.01605 |
| Five-stream arbitration, pre-convention | 0.01317 |
| Full pipeline, after label-convention rules | 0.00700 |

Synthetic fine-tuning gives the largest single gain. Arbitration adds a further reduction, reaching 0.01220 on a fresh replay with a paired-bootstrap interval of [0.00266, 0.00517]. The 0.00700 figure is listed separately because label-convention normalisation improves agreement with the benchmark's quote and dash conventions rather than visual recognition.

The held-out competition result is under organiser embargo.

## How it works

Maltese has no public, reusable paragraph-scale OCR training corpus. Everything the recogniser learns comes from synthetic renders.

**Training.** Text from eleven `korpus_malti` domain splits, screened so diacritic-stripped sources never enter the pool. 68 fonts, each checked glyph by glyph so no face silently substitutes `c` for `ċ`. PDF-realistic augmentation at the measured resolution of the real crops. The Tesseract 5 LSTM is fine-tuned on the resulting line images.

**Inference.** Five Tesseract configurations read the same image and fail on different words:

| Stream | Configuration |
|---|---|
| 1 | `mlt` fine-tuned |
| 2 | `mlt+ita` fine-tuned, default anchor |
| 3 | `mlt+ita+fra` fine-tuned |
| 4 | stock Maltese |
| 5 | `mlt+ita` on a 2x upscaled crop |

One stream anchors the output structure. The others propose word-level substitutions at anchor-aligned positions and cannot insert or delete. A proposal is eligible only if it is in the lexicon, sits within edit distance 2, and neither shortens the anchor's alphabetic length nor drops a diacritic. A diacritic-restoration gate runs first and promotes a lexicon-valid candidate carrying more canary characters, which is how `zwieg` becomes `żwieġ`. Plurality decides the winner, with no quorum, so a single eligible proposal can carry a position.

Tesseract 5 is itself an LSTM recogniser. Arbitration adds no second neural network, and no GPU is needed.

## Quickstart

Requires Python 3.9+ and Tesseract 5.

- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` checks the environment, pulls the weights, transcribes the bundled fixtures, and runs the tests.

```python
from PIL import Image
from competition_transcriber import CompetitionTranscriber

t = CompetitionTranscriber()          # downloads ~260 MB on first construction
print(t.transcribe(Image.open("crop.png")))
```

`transcribe` returns one joined paragraph string and touches the network never; all downloads happen in `__init__`.

## For competition judges

Use the `competition_transcriber.py` in this repository rather than the copy on the Hugging Face model page. This one carries a Windows path fix: it sets `TESSDATA_PREFIX` instead of passing `--tessdata-dir "…"` inside the Tesseract config string, which `pytesseract` mangles through `shlex.split` on Windows. The weights are identical either way.

On the evaluation machine:

- Tesseract 5 need not be on `PATH`. The loader also checks `C:\Program Files\Tesseract-OCR\tesseract.exe`, and honours `TESSERACT_CMD`.
- Install this repository's `requirements.txt` **before** the organisers'. The pins do not conflict; `malti==0.3.1` pulls only `sentence-splitter==1.4`.
- Everything downloads during `__init__`. Nothing downloads during `transcribe`.

To score against the organisers' development set, copy `competition_transcriber.py` next to their `competition_evaluator.py` and run it there.

## Replication

```bash
pip install -r requirements-replication.txt
make help
```

| Step | Command |
|---|---|
| Validate fonts cover the diacritics | `make fonts-check` |
| Pull corpus text | `make corpus` |
| Render a synthetic shard | `make synth` |
| Cut line crops and `.lstmf` | `make export` |
| Fine-tune the LSTM | `make finetune` |
| Per-bucket stratified CER | `make eval` |
| Bootstrap and permutation audit | `make audit` |

`scripts/bootstrap_stats.py` holds the shared paired-bootstrap and permutation helpers. Every script that recomputes a CER reported in the paper calls `jiwer`, and nothing else, so the numbers cannot drift between backends.

Cross-language runs sit under `experiments/hungarian/` and `experiments/luxembourgish/`, each the same shape: build an evaluation set, run the stock baseline, run ours, then test significance. Recorded outputs are in their `results/` directories.

The corpus builder is `experiments/corpus/`: SPARQL against EUR-Lex CELLAR, the Wikipedia REST export, paragraph alignment against an authoritative text rather than the PDF glyph layer, and a contamination check against the development set.

## Layout

```
competition_transcriber.py   the deliverable, self-contained
src/datagen/                 synthetic renderer, augmentation, font validation
src/eval/                    CER, per-bucket stratification, bootstrap audit
src/joiner/                  soft vs structural hyphen resolution
scripts/                     audit harness, lexicon and confusion builders,
                             Tesseract fine-tune, Hugging Face packaging
experiments/                 cross-language runs, corpus builder
configs/                     synthetic generation configs
fixtures/                    small self-rendered sample crops
tests/
```

## Data and assets

The competition development set belongs to the organisers and is not mirrored here. Request it from the [competition page](https://www.um.edu.mt/projects/nomocrat/doceng26competition/), then point the eval scripts at it with `--dev-dir`.

`fixtures/dev/` holds five self-rendered paragraphs covering the canary diacritics, the clitic article hyphen, a soft line-break hyphen, and the em-dash clause marker. They let the smoke test run offline. They are not a benchmark.

Weights and fonts are fetched, not committed:

```bash
scripts/fetch_assets.sh           # weights, lexicon, confusion table
scripts/fetch_assets.sh --fonts   # plus a font pool for synthesis
```

What the Hugging Face model repository contains:

| File | Description |
|---|---|
| `tessdata/mlt.traineddata` | Fine-tuned Tesseract LSTM, 50k synthetic Maltese lines |
| `tessdata/mltstock.traineddata` | Stock Maltese, kept as an independent error stream |
| `tessdata/ita.traineddata` | Italian chain, used by streams 2, 3 and 5 |
| `tessdata/fra.traineddata` | French chain, used by stream 3 |
| `tess_confusion.json` | Character confusion table |
| `maltese_en_it_lexicon.json` | Lexicon for arbitration |

## Maltese conventions the pipeline preserves

The clitic article attaches with a structural hyphen (`il-kelb`, `fis-seħħ`) that shares a glyph with the soft line-break hyphen. The joiner removes the second and keeps the first. Gold labels use curly quotes and an em-dash, and an en-dash drawn in an image maps to an em-dash in the label. The four pairs `ċ/c`, `ġ/g`, `ħ/h`, `ż/z` are tracked at every stage, because a font or tokeniser that collapses them corrupts the label before training starts.

## Licence

Code is Apache 2.0. See [LICENSE](LICENSE).

The weights are trained on `korpus_malti` text, which is CC BY-NC-SA 4.0 and access-gated. Whether those terms carry across to weights trained on rendered text is unsettled, so treat the weights as carrying that restriction until a rights holder says otherwise.

The released corpus is licensed per sample: EUR-Lex material under Commission Decision 2011/833/EU, Wikipedia-derived text under CC BY-SA 4.0.

## Citation

```bibtex
@misc{darmanin2026lvrovermlt,
  title         = {LV-ROVER-MLT: Low-Resource Maltese OCR by Synthetic
                   Fine-Tuning and Multi-Stream Arbitration},
  author        = {Darmanin, Adam},
  year          = {2026},
  eprint        = {2607.00250},
  archivePrefix = {arXiv}
}
```
