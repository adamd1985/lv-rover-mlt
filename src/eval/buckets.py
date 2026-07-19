"""Bucket classifier.

Six axes:
  1. length quartile (q1..q4) using corpus-derived quartile thresholds
  2. language (mt | en | other)
  3. il-prefix density (prefix-yes | prefix-no)
  4. line-break-hyphen presence (linehyp-yes | linehyp-no)
  5. em-dash presence (em-dash-yes | em-dash-no)
  6. line count (single-line | multi-line) from the gold `as_lines` list

The gold uses only ASCII hyphen and em-dash U+2014. There is no en-dash
U+2013 in the gold - en-dash is image-only and normalised to em-dash in the
labels - so there is no en-dash stratum.

`tag_paragraph` returns a flat set of bucket labels for one paragraph.
`compute_quartiles` derives thresholds from a corpus of ref strings.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

DIACRITICS = set("ċĊġĠħĦżŻ")
EM_DASH = "—"
HYPHEN_MINUS = "-"
SOFT_HYPHEN = "­"

# `il-` family. Word boundary on the left; one of the article forms; ASCII
# hyphen; an alphabetic char on the right.
STRUCTURAL_PREFIX = re.compile(
    r"(?:^|[\s\(\[\"‘“])(?:il|id|is|it|in|ir|ix|iz|l|fil|fis|fit|fid|fl|"
    r"bil|bis|bit|bid|tal|tas|tat|tad|min|mil|mis|mit)-[a-zA-Zàèìòùċġħż]",
    re.IGNORECASE,
)

# Small clean English-only wordlist for the code-switch heuristic. Kept tiny
# and high-precision; a larger list would false-positive on Maltese loanwords.
_EN_TOKENS = frozenset(
    """
    the and that with from this they have been which would could should
    about there their what when where while because between through above
    after before under over into onto off out up down only also even more
    less most least very much many some such only than then them these
    those does did doing done being having will shall might must can may
    of in on at to is it as be by or if so we us our you your he she his
    her its but not no yes one two three four five six seven eight nine ten
    """.split()
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def compute_quartiles(refs: Sequence[str]) -> Tuple[int, int, int]:
    """Returns (q1, q2, q3) char-length cutoffs from the corpus.
    Empty corpus -> placeholder thresholds (80, 200, 400)."""
    lengths = sorted(len(_nfc(r)) for r in refs if r)
    n = len(lengths)
    if n < 4:
        return (80, 200, 400)
    q1 = lengths[n // 4]
    q2 = lengths[n // 2]
    q3 = lengths[(3 * n) // 4]
    return (q1, q2, q3)


def bucket_length(s: str, quartiles: Tuple[int, int, int]) -> str:
    n = len(_nfc(s))
    q1, q2, q3 = quartiles
    if n <= q1:
        return "len-q1"
    if n <= q2:
        return "len-q2"
    if n <= q3:
        return "len-q3"
    return "len-q4"


def _english_score(s: str) -> float:
    toks = re.findall(r"[A-Za-z]+", s)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t.lower() in _EN_TOKENS)
    return hits / max(len(toks), 1)


def bucket_language(s: str) -> str:
    s_n = _nfc(s)
    has_diac = any(c in DIACRITICS for c in s_n)
    has_latin = bool(re.search(r"[A-Za-z]", s_n))
    en_ratio = _english_score(s_n)
    if has_diac:
        return "lang-mt"
    if has_latin and en_ratio >= 0.20:
        return "lang-en"
    if has_latin:
        # Latin script, no diacritics, low English hits -> default to mt
        return "lang-mt"
    return "lang-other"


def bucket_il_prefix(s: str) -> str:
    return "prefix-yes" if STRUCTURAL_PREFIX.search(_nfc(s)) else "prefix-no"


def bucket_line_hyphen(lines: Optional[Sequence[str]]) -> str:
    """A line-break hyphen is an ASCII `-` (or soft hyphen) at the end of a
    non-final raw line. Standalone dashes (`-`, `–`, `—`) preceded by a space
    are not line-break hyphens."""
    if not lines:
        return "linehyp-no"
    for ln in list(lines)[:-1]:
        stripped = (ln or "").rstrip()
        if not stripped:
            continue
        last = stripped[-1]
        if last in (HYPHEN_MINUS, SOFT_HYPHEN) and len(stripped) >= 2 and stripped[-2] != " ":
            return "linehyp-yes"
    return "linehyp-no"


def bucket_em_dash(s: str) -> str:
    return "em-dash-yes" if EM_DASH in _nfc(s) else "em-dash-no"


def bucket_line_count(lines: Optional[Sequence[str]]) -> str:
    """single-line vs multi-line, from the gold `as_lines` list. A missing or
    empty list (no line info) is treated as single-line."""
    return "multi-line" if lines and len(lines) > 1 else "single-line"


def tag_paragraph(
    ref: str,
    lines: Optional[Sequence[str]] = None,
    quartiles: Tuple[int, int, int] = (80, 200, 400),
) -> Set[str]:
    return {
        bucket_length(ref, quartiles),
        bucket_language(ref),
        bucket_il_prefix(ref),
        bucket_line_hyphen(lines),
        bucket_em_dash(ref),
        bucket_line_count(lines),
    }


def tag_corpus(
    refs: Sequence[str],
    line_lists: Optional[Sequence[Optional[Sequence[str]]]] = None,
    quartiles: Optional[Tuple[int, int, int]] = None,
) -> List[Set[str]]:
    q = quartiles or compute_quartiles(refs)
    ll = list(line_lists) if line_lists is not None else [None] * len(refs)
    return [tag_paragraph(r, l, q) for r, l in zip(refs, ll)]
