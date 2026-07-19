"""Fetch real (page image, authoritative text) pairs from Maltese Wikipedia.

Same shape as fetch_eurlex_maltese.py, different source: mt.wikipedia.org
has 7,885 articles (verified via action=query&meta=siteinfo, 2026-07-02),
each independently reachable as (a) a real rendered PDF via the REST PDF
export endpoint and (b) plaintext ground truth via the plaintext-extract
API - two independent code paths, same non-glyph-layer-as-ground-truth
principle as EUR-Lex/Formex. Output layout matches fetch_eurlex_maltese.py
exactly (source.pdf + authoritative_text.txt + meta.json per doc dir), so
align_pdf_paragraphs.py works unchanged on either source.

Third version. The first version reacted to 429s with an ad-hoc exponential
backoff loop; the second added a hand-rolled global pacer. Both worked but
reinvented what urllib3 and pyrate-limiter already do correctly. This
version uses the real libraries instead:

- `urllib3.util.Retry` mounted on the `requests.Session` (via `HTTPAdapter`)
  handles retry-on-429/503 with exponential backoff AND
  `respect_retry_after_header=True`, so a `Retry-After` header from the API
  is honored automatically at the transport layer - no manual header
  parsing.
- `pyrate_limiter.Limiter` enforces the proactive request pacing (the
  actual fix for getting throttled in the first place - pace requests
  below the limit instead of bursting and backing off after the fact),
  blocking internally up to `max_delay` rather than a hand-written sleep
  loop.

Persists a title-attempt cache (`_attempted_titles.json` in --out) so a
second run against the same --out dir resumes rather than re-fetching.
`_fetch_summary.json` is written incrementally, not just at the end.

License: Wikipedia text is CC BY-SA 4.0 (also GFDL-dual-licensed);
redistribution requires attribution and share-alike for derived text. Every
sample's article title, revision, and URL is recorded for the HF data card.

Usage:
    python fetch_wikipedia_maltese.py --n 50 --out data/wikipedia_mt_batch1
    # re-running with the same --out resumes rather than re-fetching
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from pyrate_limiter import Duration, Limiter, Rate
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

API = "https://mt.wikipedia.org/w/api.php"
PDF_ENDPOINT = "https://mt.wikipedia.org/api/rest_v1/page/pdf/{title}"

HEADERS = {
    "User-Agent": "MaltaOCR-research/1.0 (academic, non-competition post-hoc corpus build; contact via repo)",
}

MIN_TEXT_CHARS = 400  # skip stub articles, too short to yield useful paragraph crops

# Proactive pacing: at most 1 request per 1.2s, sustained - the actual fix
# for triggering the host's rate limiter in the first place. Blocks
# internally (raise_when_fail=False) up to max_delay rather than a manual
# sleep loop.
_limiter = Limiter(
    Rate(1, Duration.SECOND * 1.2),
    raise_when_fail=False,
    max_delay=Duration.SECOND * 120,
)

# Reactive safety net: if a 429/503 gets through despite pacing, retry with
# real exponential backoff and Retry-After compliance via urllib3, not a
# hand-rolled loop.
class _LoggingRetry(Retry):
    """Same as urllib3.Retry, but prints on every retry.

    Plain Retry is silent - it just sleeps inside session.get() with no
    visibility, which made an earlier run look "stuck" when it was actually
    correctly backing off through real, sustained 429s. Logging here is
    the fix: same retry behavior, but observable.
    """

    def increment(self, *args, **kwargs):
        new_retry = super().increment(*args, **kwargs)
        print(f"[fetch_wiki] retry: {self.total} attempts left after this one, "
              f"backoff_factor={self.backoff_factor}")
        return new_retry


_retry = _LoggingRetry(
    total=6,
    backoff_factor=2.0,
    status_forcelist=[429, 503],
    respect_retry_after_header=True,
    allowed_methods=frozenset(["GET"]),
)
_session = requests.Session()
_session.headers.update(HEADERS)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def _paced_get(url: str, params: dict | None, timeout: int) -> requests.Response | None:
    _limiter.try_acquire(url)
    try:
        return _session.get(url, params=params, timeout=timeout)
    except requests.exceptions.RetryError:
        return None


def list_articles_batch(rnlimit: int = 20) -> list[str]:
    """One page of real (non-redirect, ns=0) article titles via list=random."""
    params = {
        "action": "query", "list": "random", "rnnamespace": 0,
        "rnlimit": str(rnlimit), "format": "json",
    }
    resp = _paced_get(API, params, timeout=30)
    if resp is None:
        raise RuntimeError("random listing exhausted retries")
    resp.raise_for_status()
    data = resp.json()
    return [p["title"] for p in data.get("query", {}).get("random", [])]


def fetch_plaintext(title: str) -> str | None:
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json",
    }
    resp = _paced_get(API, params, timeout=30)
    if resp is None or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return page.get("extract")
    return None


def fetch_pdf(title: str) -> bytes | None:
    url = PDF_ENDPOINT.format(title=title.replace(" ", "_"))
    resp = _paced_get(url, None, timeout=60)
    if resp is None or resp.status_code != 200 or resp.headers.get("Content-Type", "") != "application/pdf":
        return None
    return resp.content


def process_one(title: str, out_dir: Path) -> dict:
    stem = hashlib.sha256(title.encode()).hexdigest()[:16]
    sample_dir = out_dir / stem

    text = fetch_plaintext(title)
    if not text or len(text) < MIN_TEXT_CHARS:
        return {"title": title, "status": "text_too_short", "text_chars": len(text or "")}

    pdf_bytes = fetch_pdf(title)
    if not pdf_bytes:
        return {"title": title, "status": "pdf_fetch_failed"}

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "source.pdf").write_bytes(pdf_bytes)
    (sample_dir / "authoritative_text.txt").write_text(text, encoding="utf-8")

    meta = {
        "title": title,
        "url": f"https://mt.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "source": "wikipedia-mt",
        "license": "CC BY-SA 4.0 (Wikipedia text; attribution + share-alike required for derived text)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "pdf_bytes": len(pdf_bytes),
        "text_chars": len(text),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


class AttemptCache:
    """Persists which titles have already been attempted, across runs into the same --out."""

    def __init__(self, path: Path):
        self.path = path
        self.attempted: dict[str, str] = {}
        if path.exists():
            self.attempted = json.loads(path.read_text(encoding="utf-8"))

    def has(self, title: str) -> bool:
        return title in self.attempted

    def record(self, title: str, status: str) -> None:
        self.attempted[title] = status
        self.path.write_text(json.dumps(self.attempted, ensure_ascii=False), encoding="utf-8")

    def ok_count(self) -> int:
        return sum(1 for s in self.attempted.values() if s == "ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of NEW successful fetches to reach this run (on top of any already-cached ok count)")
    ap.add_argument("--out", default="data/wikipedia_mt_batch1")
    ap.add_argument("--max-candidates", type=int, default=20000, help="hard stop on total candidates considered, regardless of --n, to avoid an unbounded run")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = AttemptCache(out_dir / "_attempted_titles.json")

    start_ok = cache.ok_count()
    target_ok = start_ok + args.n
    print(f"[fetch_wiki] resuming into {out_dir}: {start_ok} already ok, {len(cache.attempted)} already attempted, target {target_ok}")

    results = []
    considered = 0
    while cache.ok_count() < target_ok and considered < args.max_candidates:
        try:
            batch = list_articles_batch()
        except RuntimeError as e:
            print(f"[fetch_wiki] {e} - stopping this run, cache is saved, re-run the same command to resume")
            break

        for title in batch:
            if cache.has(title):
                continue
            if cache.ok_count() >= target_ok or considered >= args.max_candidates:
                break
            considered += 1
            r = process_one(title, out_dir)
            status = r.get("status")
            cache.record(title, status)
            results.append(r)
            print(f"[fetch_wiki] ok={cache.ok_count()}/{target_ok} considered={considered} [{title}] -> {status} "
                  f"({r.get('pdf_bytes', '?')} pdf bytes, {r.get('text_chars', '?')} text chars)")

    (out_dir / "_fetch_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fetch_wiki] done: {cache.ok_count()}/{target_ok} ok (this run added {cache.ok_count() - start_ok}), "
          f"{considered} new candidates considered")


if __name__ == "__main__":
    main()
