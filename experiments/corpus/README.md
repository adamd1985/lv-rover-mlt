# Real Maltese OCR corpus pipeline

Builds real (page-crop image, ground-truth text) pairs for Maltese OCR
training from open-licensed sources, with an independently-sourced text
ground truth for every pair (never the PDF's own glyph layer directly -
see `align_pdf_paragraphs.py` for why). Each script is documented in its
own module docstring; this file is the map showing how they chain together.

## Pipeline order

```
1. fetch_eurlex_maltese.py      \
                                  } -> source.pdf + authoritative_text.txt
   fetch_wikipedia_maltese.py   /     per document, in a hashed subdir

2. align_pdf_paragraphs.py         -> per-document paragraphs/manifest.json
   (run once per fetched doc dir)     + cropped PNGs, cross-checked against
                                       the authoritative text before keeping

3. dedup_against_devset.py         -> contamination_report.json
   (run once per corpus batch)        (text n-gram + image dHash vs the
                                        competition dev set - review before
                                        publishing, does not auto-delete)

4. package_for_hf.py               -> images/ + metadata.jsonl + README.md
   (run once across all sources)      (HF imagefolder format, ready for
                                        huggingface-cli upload - upload
                                        itself is a separate, explicit step)
```

## Sources

| Source | Script | Scale ceiling | License |
|---|---|---|---|
| EUR-Lex / CELLAR | `fetch_eurlex_maltese.py` | 1 document (verified: only 1 Maltese expression in CELLAR has both a PDF and Formex4 manifestation) | Commission Decision 2011/833/EU |
| Malta gov (gov.mt / parlament.mt / justice.gov.mt) | `../../../scripts/scrape_maltese_pdfs.py` | 0 - blocked | n/a, never reached |
| Maltese Wikipedia | `fetch_wikipedia_maltese.py` | ~7,885 articles, ~74% yield real PDF+text pairs | CC BY-SA 4.0 |

Malta's `*.gov.mt` domain family (including `data.gov.mt`) sits behind
Cloudflare bot-mitigation that returns 403 to every request tried from this
environment (default UA, browser UA, curl, Python - all identical result).
Not routed around; that would cross into detection-evasion territory this
project doesn't do. `legislation.mt` is reachable but its listing page is a
server-side DataTables AJAX endpoint needing a session-bound ASP.NET
antiforgery token, not reverse-engineered further (diminishing returns).
Wikipedia carries essentially all of the corpus's scale as a result.

## Known operational quirks (read before re-running at scale)

- **Wikimedia rate limits from this environment are real and persistent**,
  not just a burst you can retry past once. `list=allpages` pagination
  gets hard-throttled; `list=random` in batches of 20 is the working title-
  discovery method. Both `fetch_plaintext` and `fetch_pdf` retry on 429
  with exponential backoff (`_get_with_retry`) - do not remove this, an
  earlier version without it silently recorded empty results under load
  instead of erroring, which looked like real stub articles until manually
  re-queried.
- **Large batches (500+) run for hours**, not minutes, once rate limiting
  kicks in properly. Expect roughly 25-45 successful fetches per 10
  minutes once warmed up; the first few minutes are often the slowest
  (heaviest backoff on the initial listing calls).
- **Transient network errors happen** (DNS resolution failures observed
  mid-run once) - the fetch script does not auto-resume from where it left
  off if the whole process dies; whatever's on disk when it stops is kept,
  nothing is lost, but you may want to just start a fresh batch into a new
  `--out` dir rather than chase resume logic that doesn't exist yet.
- **No self-dedup across batches**: running `fetch_wikipedia_maltese.py`
  twice into different `--out` dirs may re-fetch the same article if
  `list=random` happens to resample it. Harmless (wastes a little time and
  disk, does not corrupt anything) but worth knowing before assuming batch
  N and batch N+1 are disjoint.
- **Image contamination-hash choice matters**: average-hash (aHash)
  produces false-positive "duplicates" on these images specifically (thin,
  mostly-white text-line crops dilute the mean-brightness signal aHash
  relies on) - `dedup_against_devset.py` uses gradient-based dHash instead.
  If you change the crop shape or background style, re-validate the
  dHash threshold rather than assuming it still holds.

## Current state (see `../STATUS.md` for the full log)

14,165 real pairs packaged in `../../../data/hf_package_v1/` as of the last
run (94 EUR-Lex + 14,071 Wikipedia across batches 1-2). Not yet uploaded to
HuggingFace - that requires an explicit go-ahead, this pipeline only
produces the local package.
