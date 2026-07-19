"""Hungarian LV-ROVER transcriber - same method as competition_transcriber.py,
only the language changes. Classes/functions below are ported verbatim from
that file (edit distance, soft-lexicon ROVER vote, post-processing chain);
only language codes, canary set, and the joiner call site are Hungarian."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import List, Optional

import pyphen
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/home/adamd1985/miniforge3/envs/sys/bin/tesseract"
TESSDATA_DIR = "tessdata_hu"

_TESS_LANG_PRIMARY = "hun"
_TESS_LANG_AUGMENTED = "hun+deu"
_TESS_LANG_ROMANCE = "hun+deu+slk"
_TESS_LANG_STOCK = "hunstock"
_ITA_FALLBACK_RATIO = 0.6
_CROSS_ENGINE_MAX_SWAP_DIST = 2

_MALTESE_DIAC = "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"  # Hungarian canary set, same role as ċġħż
_DIAC_FOLD = str.maketrans("áÁéÉíÍóÓöÖőŐúÚüÜűŰ", "aAeEiIoOoOoOuUuUuU")
_PUNCT_STRIP = ".,;:!?\"'()[]{}«»—–-‘’“”"
_WS_SPLIT = re.compile(r"[ \t]+")
_DIC = pyphen.Pyphen(lang="hu")

with open("data/hu_lexicon.json", encoding="utf-8") as f:
    _LEXICON = set(json.load(f).keys())


def _norm_lookup(w: str) -> str:
    return unicodedata.normalize("NFC", w).strip(_PUNCT_STRIP)


def _non_ascii_alpha(s: str) -> int:
    return sum(ord(c) > 127 and c.isalpha() for c in s)


def _strip_diac(s: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return base.translate(_DIAC_FOLD)


def _diac_count(s: str) -> int:
    return sum(c in _MALTESE_DIAC for c in s)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _align_word_seqs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub)
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                out.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((a[i - 1], None))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            out.append((None, b[j - 1]))
            j -= 1
            continue
        break
    out.reverse()
    return out


class _CrossEngineRouter:
    def __init__(self, lexicon: set, max_swap_dist: int = 2) -> None:
        self.lex = lexicon
        self.max_swap_dist = max_swap_dist

    def _in_lex(self, w: str) -> bool:
        n = _norm_lookup(w)
        return bool(n) and (n in self.lex or n.lower() in self.lex)

    def _decide(self, anchor: str, candidate: Optional[str]) -> str:
        if not candidate or candidate == anchor:
            return anchor
        na, nc = _norm_lookup(anchor), _norm_lookup(candidate)
        if (
            na and nc
            and _strip_diac(na.lower()) == _strip_diac(nc.lower())
            and _diac_count(candidate) > _diac_count(anchor)
            and self._in_lex(candidate)
            and _edit_distance(na, nc) <= self.max_swap_dist
        ):
            return candidate
        if self._in_lex(anchor):
            return anchor
        if not self._in_lex(candidate):
            return anchor
        a_alpha = sum(c.isalpha() for c in anchor)
        c_alpha = sum(c.isalpha() for c in candidate)
        if c_alpha < a_alpha or a_alpha < 3 or c_alpha < 3:
            return anchor
        if len(candidate) < len(anchor) - 1 or abs(len(anchor) - len(candidate)) > 2:
            return anchor
        if _non_ascii_alpha(candidate) < _non_ascii_alpha(anchor):
            return anchor
        d = _edit_distance(anchor, candidate)
        if d == 0 or d > self.max_swap_dist:
            return anchor
        return candidate

    def combine_lv(self, anchor: str, candidate_streams: List[str]) -> str:
        cand_token_lists = [
            [w for w in _WS_SPLIT.split(c.replace("\n", " ").strip()) if w]
            for c in candidate_streams
        ]
        cursors = [0] * len(candidate_streams)
        out_lines = []
        for anchor_line in anchor.split("\n"):
            a_words = [w for w in _WS_SPLIT.split(anchor_line.strip()) if w]
            if not a_words:
                out_lines.append("")
                continue
            aligned_per_stream = []
            for k, tokens in enumerate(cand_token_lists):
                window = tokens[cursors[k]: cursors[k] + 2 * len(a_words)]
                alignment = _align_word_seqs(a_words, window)
                per_pos: List[Optional[str]] = []
                for a, b in alignment:
                    if a is None:
                        continue
                    per_pos.append(b)
                while len(per_pos) < len(a_words):
                    per_pos.append(None)
                aligned_per_stream.append(per_pos[: len(a_words)])
                cursors[k] += len(a_words)
            line_out = []
            for i, anc in enumerate(a_words):
                votes = Counter()
                stream_order = []
                for k in range(len(aligned_per_stream)):
                    c = aligned_per_stream[k][i]
                    swap = self._decide(anc, c)
                    if swap != anc:
                        if swap not in votes:
                            stream_order.append(swap)
                        votes[swap] += 1
                if votes:
                    best = max(stream_order, key=lambda w: (votes[w], -stream_order.index(w)))
                    line_out.append(best)
                else:
                    line_out.append(anc)
            out_lines.append(" ".join(line_out))
        return "\n".join(out_lines)


_LEAD_MARKER = re.compile(
    r"^(\s*\d{1,2})(\s+[-–—]\s*|\s*[-–—]\s+|\s*[–—]\s*)(?=[^\W\d_])", re.UNICODE,
)


def _fix_lead_marker(text: str) -> str:
    return _LEAD_MARKER.sub(lambda m: m.group(1).strip() + " — ", text, count=1)


def _fix_apostrophe(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch == "'":
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            out.append("‘" if (prev.isspace() or i == 0) and nxt.isalnum() else "’")
        else:
            out.append(ch)
    return "".join(out)


def _fix_doublequote(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch == '"':
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            out.append("“" if (prev.isspace() or i == 0) and nxt.isalnum() else "”")
        else:
            out.append(ch)
    return "".join(out)


def _pyphen_join(raw_text: str) -> str:
    """Line-join repair using Pyphen instead of the malti-package RBLineJoiner -
    Hungarian needs no clitic/structural-hyphen disambiguation."""
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    out_words: List[str] = []
    for i, ln in enumerate(lines):
        words = ln.split()
        if not words:
            continue
        if i > 0 and out_words and out_words[-1].endswith("-"):
            stem = out_words[-1][:-1]
            candidate = stem + words[0]
            if _DIC.inserted(candidate) or candidate.lower() in _LEXICON:
                out_words[-1] = candidate
                out_words.extend(words[1:])
                continue
        out_words.extend(words)
    return " ".join(out_words)


def _normalise(text: str) -> str:
    text = text.replace("­", "").replace("–", "—")
    return unicodedata.normalize("NFC", text)


def _tess_paragraph(image: Image.Image, lang: str) -> str:
    raw = pytesseract.image_to_string(
        image, lang=lang, config=f'--tessdata-dir "{TESSDATA_DIR}" --psm 3'
    )
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    return _normalise(_pyphen_join("\n".join(lines)))


def transcribe(image: Image.Image, apply_postproc: bool = True) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    hun_out = _tess_paragraph(image, _TESS_LANG_PRIMARY)
    aug_out = _tess_paragraph(image, _TESS_LANG_AUGMENTED)
    romance_out = _tess_paragraph(image, _TESS_LANG_ROMANCE)
    stock_out = _tess_paragraph(image, _TESS_LANG_STOCK)
    try:
        w, h = image.size
        up = image.resize((w * 2, h * 2), Image.LANCZOS)
        up_out = _tess_paragraph(up, _TESS_LANG_PRIMARY)
    except Exception:
        up_out = ""

    tess_streams = [s for s in (hun_out, aug_out, romance_out, stock_out, up_out) if s]
    if not tess_streams:
        return ""

    base = hun_out if hun_out else aug_out
    longest = max(tess_streams, key=len)
    if len(longest) > 10 and len(base) < _ITA_FALLBACK_RATIO * len(longest):
        base = longest

    router = _CrossEngineRouter(_LEXICON, max_swap_dist=_CROSS_ENGINE_MAX_SWAP_DIST)
    candidates = [c for c in (aug_out, romance_out, stock_out, up_out) if c and c != base]
    joined = router.combine_lv(base, candidates) if candidates else base

    if not apply_postproc:
        return joined
    return _fix_doublequote(_fix_apostrophe(_fix_lead_marker(joined)))
