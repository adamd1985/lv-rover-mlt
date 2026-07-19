"""CER recipe.

NFC-normalise both sides, strip, replace internal newlines in the hypothesis
with a single space, call `jiwer.cer` on lists (global aggregation, sum of
numerators). No case-folding, no glyph equivalence, dash triplet preserved.

Functions:
  normalise          - the one canonical text pre-pass
  cer_aggregate      - global CER over a list of (ref, hyp) pairs
  cer_per_bucket     - CER per bucket tag, using sum-of-numerators
  confusion_matrix   - per-character substitution counts
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jiwer


CLITIC_ARTICLES: Tuple[str, ...] = (
    "il", "is", "id", "it", "in", "ir", "ix", "iz", "iż",
    "fis", "fil", "fid", "fit", "fix", "fl",
    "bil", "bis", "bid", "bit", "bix", "biż",
    "mil", "mis", "mid", "mit", "mix", "miż",
    "tal", "tas", "tad", "tat", "tax", "taż",
    "sal", "sas", "sad", "sat", "sax",
    "ġos", "għax", "għal", "għall",
    "sa", "ma", "kull", "lil", "biex", "bħal",
    "l",
)


def _build_clitic_regex() -> re.Pattern[str]:
    longest_first = sorted(set(CLITIC_ARTICLES), key=len, reverse=True)
    alt = "|".join(re.escape(a) for a in longest_first)
    return re.compile(rf"(?<![\w'’])({alt})-[ \t]+", re.IGNORECASE)


_CLITIC_RE = _build_clitic_regex()


def clitic_space_normalise(s: str) -> str:
    """Collapse stray whitespace after a Maltese article clitic hyphen.

    Korpus_malti v4.2 contains noise like ``il- foo`` and ``tal- Kummissjoni``
    where a space follows the clitic hyphen. The rule-based joiner emits the
    canonical compact form (``il-foo``); under raw CER the gold-side noise is
    scored as a model error. This pass collapses ``<article>-<space>+`` to
    ``<article>-`` on a copy so the normalised CER reflects model error only.
    Case is preserved. Dashes U+2013 and U+2014 are never touched - only the
    ASCII hyphen U+002D in clitic position is in scope.
    """
    if not s:
        return s
    return _CLITIC_RE.sub(lambda m: m.group(1) + "-", s)


def normalise(s: str, *, is_hyp: bool = False) -> str:
    """NFC, strip outer whitespace, replace internal \\n with space if hyp."""
    if s is None:
        return ""
    out = unicodedata.normalize("NFC", s)
    if is_hyp:
        out = out.replace("\n", " ").replace("\r", " ")
    return out.strip()


def _prep_pair(ref: str, hyp: str) -> Tuple[str, str]:
    return normalise(ref, is_hyp=False), normalise(hyp, is_hyp=True)


def cer_organiser(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Leaderboard-faithful CER. Byte-for-byte parity with the organiser's
    ``evaluate.load('cer').compute(...)`` call (jiwer 4.0.0 backed).

    The organiser applies NO normalisation: no NFC, no strip, no newline
    handling. ``evaluate``'s CER metric forwards the raw lists straight to
    ``jiwer.process_characters``, which aggregates as total (S+D+I) over total
    reference characters across the whole set. This function reproduces that
    exactly and is the only number that may be quoted as the leaderboard CER.

    Everything else in this module (``cer_aggregate`` and below) applies our
    internal normalisation and is a diagnostic, not the leaderboard metric.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")
    if not refs:
        return 0.0
    return float(jiwer.cer(list(refs), list(hyps)))


def cer_aggregate(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Global CER over the full corpus, with internal normalisation applied.

    This is a diagnostic. For the leaderboard number use :func:`cer_organiser`.
    Empty refs are dropped (with a hyp they would force a divide-by-zero on the
    bucket; jiwer's behaviour is documented but we keep the contract explicit).
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")
    r_list: List[str] = []
    h_list: List[str] = []
    for r, h in zip(refs, hyps):
        r_n, h_n = _prep_pair(r, h)
        if not r_n:
            continue
        r_list.append(r_n)
        h_list.append(h_n)
    if not r_list:
        return 0.0
    return float(jiwer.cer(r_list, h_list))


def compute_cer_dual(
    refs: Sequence[str],
    hyps: Sequence[str],
) -> Dict[str, float]:
    """Return raw and clitic-normalised CER plus the gap.

    ``raw_cer`` is the leaderboard-faithful number; it matches the organiser
    script and must never be silently replaced. ``normalised_cer`` applies
    :func:`clitic_space_normalise` to a copy of both ref and hyp before
    scoring, so gold-side stray-space artefacts cannot be charged to the
    model. ``delta = raw_cer - normalised_cer`` is positive when gold-side
    noise inflates raw CER.
    """
    raw = cer_aggregate(refs, hyps)
    refs_n = [clitic_space_normalise(r) for r in refs]
    hyps_n = [clitic_space_normalise(h) for h in hyps]
    norm = cer_aggregate(refs_n, hyps_n)
    return {
        "raw_cer": float(raw),
        "normalised_cer": float(norm),
        "delta": float(raw - norm),
    }


def cer_per_bucket(
    refs: Sequence[str],
    hyps: Sequence[str],
    bucket_tags: Sequence[Iterable[str]],
    *,
    min_n: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Sum-of-numerators per bucket. `bucket_tags[i]` is the set of bucket
    labels the i-th paragraph belongs to (one paragraph can be in many).
    Returns {bucket: {"n": int, "cer": float, "small": bool}}.
    """
    if not (len(refs) == len(hyps) == len(bucket_tags)):
        raise ValueError("refs/hyps/bucket_tags length mismatch")
    by_bucket_refs: Dict[str, List[str]] = defaultdict(list)
    by_bucket_hyps: Dict[str, List[str]] = defaultdict(list)
    for r, h, tags in zip(refs, hyps, bucket_tags):
        r_n, h_n = _prep_pair(r, h)
        if not r_n:
            continue
        for t in tags:
            by_bucket_refs[t].append(r_n)
            by_bucket_hyps[t].append(h_n)
    out: Dict[str, Dict[str, float]] = {}
    for t in by_bucket_refs:
        n = len(by_bucket_refs[t])
        c = float(jiwer.cer(by_bucket_refs[t], by_bucket_hyps[t])) if n else 0.0
        out[t] = {"n": n, "cer": c, "small": n < min_n}
    return out


def confusion_matrix(refs: Sequence[str], hyps: Sequence[str]) -> Counter:
    """Per-character substitution count via the jiwer character alignment.

    Implementation note: jiwer's `process_characters` produces alignments
    that include substitutions, insertions and deletions. We bucket them into
    a single counter keyed by (ref_char, hyp_char). Insertions use ref_char
    = "<INS>" and deletions hyp_char = "<DEL>".
    """
    mat: Counter = Counter()
    for r, h in zip(refs, hyps):
        r_n, h_n = _prep_pair(r, h)
        if not r_n:
            continue
        out = jiwer.process_characters(r_n, h_n)
        # alignments is a list (per pair) of lists of AlignmentChunk objects
        for chunks in out.alignments:
            for c in chunks:
                if c.type == "equal":
                    continue
                ref_slice = r_n[c.ref_start_idx:c.ref_end_idx]
                hyp_slice = h_n[c.hyp_start_idx:c.hyp_end_idx]
                if c.type == "substitute":
                    for a, b in zip(ref_slice, hyp_slice):
                        mat[(a, b)] += 1
                elif c.type == "delete":
                    for a in ref_slice:
                        mat[(a, "<DEL>")] += 1
                elif c.type == "insert":
                    for b in hyp_slice:
                        mat[("<INS>", b)] += 1
    return mat


def cer_on_char_subset(
    refs: Sequence[str],
    hyps: Sequence[str],
    chars: Sequence[str],
) -> Dict[str, float]:
    """Per-character canary CER: for each char in `chars`, restrict ref
    to paragraphs that contain it and report CER on that subset.
    """
    refs_n = [normalise(r) for r in refs]
    hyps_n = [normalise(h, is_hyp=True) for h in hyps]
    out: Dict[str, float] = {}
    for ch in chars:
        sub_r = [r for r in refs_n if ch in r]
        sub_h = [h for r, h in zip(refs_n, hyps_n) if ch in r]
        sub_r = [r for r in sub_r if r]
        if not sub_r:
            out[ch] = float("nan")
            continue
        out[ch] = float(jiwer.cer(sub_r, sub_h))
    return out


def _edit_count(r: str, h: str) -> Tuple[int, int]:
    """Return (edit_distance, ref_length) for a single pair.
    Used by audit.py for bootstrap.
    """
    r_n, h_n = _prep_pair(r, h)
    if not r_n:
        return 0, 0
    res = jiwer.process_characters(r_n, h_n)
    # jiwer reports cer = (S+D+I)/N; we reconstruct numerator from cer * N.
    n = len(r_n)
    # process_characters returns counts in res.substitutions etc.
    ed = int(res.substitutions + res.deletions + res.insertions)
    return ed, n
