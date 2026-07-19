"""Fetch real Maltese (page image, authoritative text) pairs from EUR-Lex/CELLAR.

Pipeline (all steps verified against the live endpoint, 2026-07-02):
1. SPARQL-query the EU Publications Office CELLAR endpoint (cdm ontology,
   http://publications.europa.eu/webapi/rdf/sparql) for Maltese-language
   expressions (cdm:expression_uses_language = authority/language/MLT) that
   have BOTH a "pdf" and an "fmx4" (Formex 4 XML) manifestation
   (cdm:manifestation_type). Formex, not HTML, is the reliable inline text
   format for legal acts specifically - HTML manifestations often 404 on
   direct fetch, and many Maltese CELLAR expressions overall (mostly
   case-law judgments) have no PDF at all (0/20 in an unfiltered sample).
2. For each pair, GET the PDF manifestation URI with Accept: application/pdf
   (returns the raw PDF), and the Formex manifestation URI with
   Accept: application/zip;mtype=fmx4 (returns a zip containing the
   consolidated-act XML - the base filename without ".doc.xml" is the one
   with the actual ENACTING.TERMS body text; the ".doc.xml" sibling is a
   thin bibliographic pointer file, not the article text).
3. Extract running text from <ARTICLE>/<PARAG>/<ALINEA>/<P> elements - never
   from the PDF's own glyph layer (broken ToUnicode CMaps silently drop
   diacritics and confuse dashes, which is why Maltese PDF text layers are
   unreliable). EUR-Lex publishes the
   authoritative text as a *separate* Formex document from the same CELLAR
   expression, so this sidesteps that failure mode entirely.

License: EUR-Lex content is reusable, including for redistribution, under
Commission Decision 2011/833/EU (attribution, no distortion of meaning,
non-liability). Every fetched document's CELEX/CELLAR id and source URL is
recorded per-sample for the eventual HF data card.

Usage:
    python fetch_eurlex_maltese.py --n 10 --out data/eurlex_mt_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

SPARQL_ENDPOINT = "http://publications.europa.eu/webapi/rdf/sparql"

# Verified against the live endpoint 2026-07-02: cdm:manifestation_type
# gives "pdf"/"fmx4" as a plain string literal per manifestation; joining
# two manifestations of the same expression on type finds real PDF+Formex
# pairs for legal acts. (An earlier attempt joined on "html" instead of
# "fmx4" - technically found pairs, but they turned out to be Publications
# Office factsheets/brochures whose HTML manifestation redirects through an
# external DOI resolver that 403s on a plain fetch, not directly-fetchable
# legal-act text. fmx4 is the right join for Official Journal legal acts.)
SPARQL_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?expr ?pdfManif ?fmxManif WHERE {{
  ?expr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/MLT> .
  ?pdfManif cdm:manifestation_manifests_expression ?expr .
  ?pdfManif cdm:manifestation_type "pdf" .
  ?fmxManif cdm:manifestation_manifests_expression ?expr .
  ?fmxManif cdm:manifestation_type "fmx4" .
}}
LIMIT {limit}
"""

HEADERS = {
    "Accept-Language": "mlt",
    "User-Agent": "MaltaOCR-research/1.0 (academic, non-competition post-hoc corpus build)",
}


def sparql_query(limit: int) -> list[dict]:
    """Return [{expr, pdf_uri, fmx_uri}, ...] for Maltese expressions with both formats."""
    resp = requests.post(
        SPARQL_ENDPOINT,
        data={"query": SPARQL_QUERY.format(limit=limit)},
        headers={"Accept": "application/sparql-results+json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for b in data["results"]["bindings"]:
        rows.append({
            "expr": b["expr"]["value"],
            "pdf_uri": b["pdfManif"]["value"],
            "fmx_uri": b["fmxManif"]["value"],
        })
    return rows


def fetch_bytes(url: str, accept: str) -> requests.Response:
    headers = dict(HEADERS)
    headers["Accept"] = accept
    return requests.get(url, headers=headers, timeout=60)


# Formex text-bearing elements, roughly in document order. TITLE holds the
# act's own title (also useful ground truth); ENACTING.TERMS holds the
# articles. <NOTE>...</NOTE> (footnotes) are dropped since they don't
# correspond to the main running text a paragraph crop would show.
_FOOTNOTE_RE = re.compile(r"<NOTE\b.*?</NOTE>", re.S)
_TEXT_TAG_RE = re.compile(r"<(P|TI|TI\.ART|STI|ALINEA|NP)\b[^>]*>(.*?)</\1>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&apos;": "'", "&quot;": '"'}


def extract_formex_text(xml_bytes: bytes) -> str:
    """Pull running paragraph text out of a Formex4 consolidated-act XML.

    Extracts text from the tags that carry actual prose (title lines,
    article headers, alinea/paragraph bodies), strips footnote markers, and
    joins with newlines so each element is a separate ground-truth line -
    this is the unit later aligned to rendered PDF paragraph regions.
    """
    xml = xml_bytes.decode("utf-8", errors="replace")
    xml = _FOOTNOTE_RE.sub("", xml)
    lines = []
    for m in _TEXT_TAG_RE.finditer(xml):
        inner = m.group(2)
        inner = _TAG_RE.sub(" ", inner)
        for ent, ch in _ENTITY_MAP.items():
            inner = inner.replace(ent, ch)
        inner = re.sub(r"&#(\d+);", lambda mm: chr(int(mm.group(1))), inner)
        inner = re.sub(r"\s+", " ", inner).strip()
        if inner:
            lines.append(inner)
    return "\n".join(lines)


def unzip_by_suffix(content: bytes, content_type: str, exclude_suffix: str, include_suffix: str) -> bytes | None:
    """Return bytes of the first zip member matching include_suffix and not exclude_suffix."""
    if "zip" not in content_type:
        return content if include_suffix.lstrip(".") in content_type else None
    try:
        zf = zipfile.ZipFile(__import__("io").BytesIO(content))
        for name in sorted(zf.namelist()):
            lname = name.lower()
            if lname.endswith(include_suffix) and not lname.endswith(exclude_suffix):
                return zf.read(name)
    except zipfile.BadZipFile:
        return None
    return None


def process_one(row: dict, out_dir: Path) -> dict:
    expr = row["expr"]
    cellar_id = expr.rstrip("/").split("/")[-1]
    stem = hashlib.sha256(cellar_id.encode()).hexdigest()[:16]
    sample_dir = out_dir / stem

    pdf_resp = fetch_bytes(row["pdf_uri"], "application/pdf")
    fmx_resp = fetch_bytes(row["fmx_uri"], "application/zip;mtype=fmx4")

    if pdf_resp.status_code != 200 or fmx_resp.status_code != 200:
        return {
            "cellar_id": cellar_id,
            "status": "http_error",
            "pdf_status": pdf_resp.status_code,
            "fmx_status": fmx_resp.status_code,
        }

    pdf_bytes = unzip_by_suffix(
        pdf_resp.content, pdf_resp.headers.get("Content-Type", ""), "__none__", ".pdf"
    )
    # the consolidated-act XML (has the article body); its ".doc.xml"
    # sibling is a thin bibliographic pointer file, excluded here.
    fmx_bytes = unzip_by_suffix(
        fmx_resp.content, fmx_resp.headers.get("Content-Type", ""), ".doc.xml", ".xml"
    )

    if not pdf_bytes or not fmx_bytes:
        return {
            "cellar_id": cellar_id,
            "status": "unzip_failed",
            "has_pdf": bool(pdf_bytes),
            "has_fmx": bool(fmx_bytes),
        }

    text = extract_formex_text(fmx_bytes)
    if len(text) < 50:
        return {"cellar_id": cellar_id, "status": "text_too_short", "text_chars": len(text)}

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "source.pdf").write_bytes(pdf_bytes)
    (sample_dir / "authoritative_text.txt").write_text(text, encoding="utf-8")

    meta = {
        "cellar_id": cellar_id,
        "expression_uri": expr,
        "pdf_uri": row["pdf_uri"],
        "fmx_uri": row["fmx_uri"],
        "source": "eur-lex",
        "license": "Commission Decision 2011/833/EU (reuse with attribution)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "pdf_bytes": len(pdf_bytes),
        "text_chars": len(text),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of documents to fetch")
    ap.add_argument("--out", default="data/eurlex_mt_smoke")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_eurlex] querying CELLAR for {args.n} Maltese PDF+HTML pairs...")
    try:
        rows = sparql_query(args.n)
    except Exception as e:
        print(f"[fetch_eurlex] SPARQL query failed: {e}")
        sys.exit(1)
    print(f"[fetch_eurlex] got {len(rows)} candidate pairs")

    results = []
    for i, row in enumerate(rows):
        print(f"[fetch_eurlex] {i+1}/{len(rows)}: {row['expr']}")
        r = process_one(row, out_dir)
        results.append(r)
        print(f"  -> {r.get('status')} ({r.get('pdf_bytes', '?')} pdf bytes, {r.get('text_chars', '?')} text chars)")
        time.sleep(1.0)  # be polite

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n[fetch_eurlex] done: {ok}/{len(results)} succeeded")
    (out_dir / "_fetch_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
