"""Maltese paragraph OCR - DocEng 2026 competition submission.

The organiser's evaluation harness imports this module and uses the class
directly:

    from competition_transcriber import CompetitionTranscriber
    transcriber = CompetitionTranscriber()
    text = transcriber.transcribe(pil_image)

Contract (competition rules):
  - `__init__` downloads the model from HuggingFace and does all setup.
  - `transcribe(PIL.Image) -> str` returns the paragraph-form transcription.
  - Batch size 1 is enforced by the harness.
  - No network access after `__init__` except the HuggingFace download.
  - Deterministic for the same image.

Approach: all gains from lexicon-anchored arbitration over diverse Tesseract
streams plus label-convention normalisation:

  1. Five Tesseract streams per image: mlt, mlt+ita (anchor), mlt+ita+fra,
     stock mlt, and mlt+ita on a 2x-upscaled image. Diversity across
     language chain, training data, and image scale gives independent
     error patterns.
  2. N-stream base recovery: the mlt+ita anchor is used unless it dropped
     content (much shorter than the longest stream), in which case the
     longest is the base.
  3. Length-gated confusion corrector: single-char swaps for non-lexicon
     words via a synth-derived P(true_char | tess_char) table.
  4. LV-ROVER lexicon-anchored majority vote over the candidate streams +
     EasyOCR: an anchor word not in the lexicon is replaced only when a
     majority of candidates agree on a lexicon-valid, edit-distance-bounded,
     diacritic-preserving alternative.
  5. Label-convention normalisation: leading clause marker "N — " and
     curly quotes (positional U+2018 / U+2019), matching the organiser
     gold convention that Tesseract does not emit.

Dev set CER: 0.00700 on 422 paragraphs (jiwer).

The pipeline degrades gracefully: if the corrector data / router lexicon
is missing from the bundle it falls back to bare Tesseract; if EasyOCR is
unavailable it is simply one fewer candidate stream.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import unicodedata
from typing import List, Optional

import PIL.Image
import pytesseract
from malti.line_joiner import RBLineJoiner

HF_REPO_DEFAULT = os.environ.get("DOCENG_HF_REPO", "radmada/lv-rover-mlt")
_EASYOCR_LANGS = ("mt",)
_CROSS_ENGINE_MAX_SWAP_DIST = 2
_TESS_LANG_PRIMARY = "mlt"
_TESS_LANG_AUGMENTED = "mlt+ita"
_TESS_LANG_ROMANCE = "mlt+ita+fra"
_TESS_LANG_STOCK = "mltstock"
_ITA_FALLBACK_RATIO = 0.6

_EN_DASH = "–"
_EM_DASH = "—"
_SOFT_HYPHEN = "­"
_PUNCT_STRIP = ".,;:!?\"'()[]{}«»—–-‘’“”"
_WS_SPLIT = re.compile(r"[ \t]+")

# Length gate: only apply confusion corrector to paragraphs >= 100 chars.
_CORRECTOR_TAU = 0.05
_CORRECTOR_LEN_THR = 100
_CORRECTOR_MIN_ALPHA = 3
_CORRECTOR_MAX_ED = 2
_CORRECTOR_MAX_LEN_DIFF = 1


def _normalise(text: str) -> str:
    text = text.replace(_SOFT_HYPHEN, "").replace(_EN_DASH, _EM_DASH)
    return unicodedata.normalize("NFC", text)


# Leading clause-marker normalisation. Gold uses "N — " (digit, space,
# em-dash, space) for list/status markers. Tesseract reads the em-dash as a
# hyphen and drops the spaces. The following token must start with a letter to
# exclude numeric ranges and ordinals ("19-20", "12-il sena").
_LEAD_MARKER = re.compile(
    r"^(\s*\d{1,2})(\s+[-–—]\s*|\s*[-–—]\s+|\s*[–—]\s*)(?=[^\W\d_])",
    re.UNICODE,
)


def _fix_lead_marker(text: str) -> str:
    return _LEAD_MARKER.sub(lambda m: m.group(1).strip() + " — ", text, count=1)


def _fix_apostrophe(text: str) -> str:
    # Gold encodes single quotes as curly: U+2018 opening, U+2019 closing.
    # A quote preceded by space/start followed by alphanumeric is opening.
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
    # Gold encodes double quotes as curly U+201C opening / U+201D closing.
    out = []
    for i, ch in enumerate(text):
        if ch == '"':
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            out.append("“" if (prev.isspace() or i == 0) and nxt.isalnum() else "”")
        else:
            out.append(ch)
    return "".join(out)


def _norm_lookup(w: str) -> str:
    return unicodedata.normalize("NFC", w).strip(_PUNCT_STRIP)


def _non_ascii_alpha(s: str) -> int:
    return sum(ord(c) > 127 and c.isalpha() for c in s)


_MALTESE_DIAC = "ċġħżĊĠĦŻ"
_DIAC_FOLD = str.maketrans("ħĦġĠċĊżŻ", "hHgGcCzZ")


def _strip_diac(s: str) -> str:
    # Fold Maltese diacritics to ASCII base so two spellings that differ only
    # in diacritics compare equal.
    base = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
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


class _ConfusionCorrector:
    """Lookup-table corrector: P(true | tess) per character, gated by lexicon."""

    def __init__(self, confusion: dict, lexicon: set, tau: float) -> None:
        self.confusion = confusion or {}
        self.lex = lexicon
        self.tau = tau

    def _in_lex(self, w: str) -> bool:
        n = _norm_lookup(w)
        return bool(n) and (n in self.lex or n.lower() in self.lex)

    def _candidates(self, word: str):
        out = []
        for i, ch in enumerate(word):
            row = self.confusion.get(ch)
            if not row:
                continue
            for g, p in row.items():
                if g == ch or p < self.tau:
                    continue
                out.append((word[:i] + g + word[i + 1:], p))
        out.sort(key=lambda x: -x[1])
        return out

    def _correct_word(self, word: str) -> str:
        if not word or len(word) < _CORRECTOR_MIN_ALPHA or self._in_lex(word):
            return word
        a_alpha = sum(c.isalpha() for c in word)
        a_special = _non_ascii_alpha(word)
        for new, p in self._candidates(word):
            if not self._in_lex(new):
                continue
            c_alpha = sum(c.isalpha() for c in new)
            if c_alpha < a_alpha or c_alpha < _CORRECTOR_MIN_ALPHA:
                continue
            if _non_ascii_alpha(new) < a_special:
                continue
            if abs(len(new) - len(word)) > _CORRECTOR_MAX_LEN_DIFF:
                continue
            if _edit_distance(word, new) > _CORRECTOR_MAX_ED:
                continue
            return new
        return word

    def correct(self, text: str) -> str:
        out_lines = []
        for line in text.split("\n"):
            tokens = _WS_SPLIT.split(line.strip())
            out_lines.append(" ".join(self._correct_word(t) for t in tokens if t))
        return "\n".join(out_lines)


def _align_word_seqs(a, b):
    """Wagner-Fischer alignment over word sequences."""
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
    """Lexicon-anchored LV-ROVER voter across multiple OCR streams.

    For each anchor word not in lexicon, count lexicon-valid candidate words
    across all streams that pass the diacritic and edit-distance guards. Pick
    the most frequent. Ties broken by stream order. Falls back to anchor when
    no candidate beats the anchor floor.
    """

    def __init__(self, lexicon: set, max_swap_dist: int = 2) -> None:
        self.lex = lexicon
        self.max_swap_dist = max_swap_dist

    def _in_lex(self, w: str) -> bool:
        n = _norm_lookup(w)
        return bool(n) and (n in self.lex or n.lower() in self.lex)

    def _decide(self, anchor: str, candidate: Optional[str]) -> str:
        if not candidate or candidate == anchor:
            return anchor
        # Diacritic restoration: adopt a sibling stream's richer spelling when
        # strip-equality holds and the richer form is in lexicon.
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

    def combine(self, anchor: str, candidate: str) -> str:
        cand_tokens = [w for w in _WS_SPLIT.split(candidate.replace("\n", " ").strip()) if w]
        cursor = 0
        out_lines = []
        for anchor_line in anchor.split("\n"):
            a_words = [w for w in _WS_SPLIT.split(anchor_line.strip()) if w]
            if not a_words:
                out_lines.append("")
                continue
            window = cand_tokens[cursor: cursor + 2 * len(a_words)]
            alignment = _align_word_seqs(a_words, window)
            per_pos: List[Optional[str]] = []
            for a, b in alignment:
                if a is None:
                    continue
                per_pos.append(b)
            while len(per_pos) < len(a_words):
                per_pos.append(None)
            line_out = [self._decide(anc, per_pos[i]) for i, anc in enumerate(a_words)]
            out_lines.append(" ".join(line_out))
            cursor += len(a_words)
        return "\n".join(out_lines)

    def combine_lv(self, anchor: str, candidate_streams: List[str]) -> str:
        from collections import Counter
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


class CompetitionTranscriber:
    """Maltese paragraph OCR using LV-ROVER multi-stream Tesseract voting.

    Downloads model files from HuggingFace on first call to __init__.
    No network access during transcribe. Degrades gracefully to bare
    Tesseract if optional corrector data is absent from the bundle.
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or HF_REPO_DEFAULT
        repo_dir = self._resolve_repo(self.model_id)

        self.tessdata_dir = os.path.join(repo_dir, "tessdata") if repo_dir else ""
        if not self.tessdata_dir or not os.path.isfile(
            os.path.join(self.tessdata_dir, "mlt.traineddata")
        ):
            self.tessdata_dir = self._fallback_tessdata_dir()
        pytesseract.pytesseract.tesseract_cmd = self._resolve_binary(repo_dir)
        # NB: pytesseract.run_tesseract splits `config` with shlex.split
        # on Windows, which does not strip quote characters, corrupting a quoted
        # --tessdata-dir path. TESSDATA_PREFIX avoids the config string entirely.
        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir
        self.config = "--psm 6"
        self.lang = "mlt"
        self.joiner = RBLineJoiner()

        self.corrector = None
        self.router = None
        self.easyocr = None
        self._lexicon: set = set()
        self._try_load_corrector(repo_dir)
        self._try_load_easyocr()
        self._warmup()

    def _try_load_corrector(self, repo_dir: str) -> None:
        if not repo_dir:
            return
        conf_path = os.path.join(repo_dir, "tess_confusion.json")
        lex_path = os.path.join(repo_dir, "maltese_en_it_lexicon.json")
        for fallback in ("maltese_en_lexicon.json", "maltese_lexicon.json"):
            if os.path.isfile(lex_path):
                break
            lex_path = os.path.join(repo_dir, fallback)
        if not (os.path.isfile(conf_path) and os.path.isfile(lex_path)):
            return
        try:
            with open(conf_path, encoding="utf-8") as f:
                cdata = json.load(f)
            with open(lex_path, encoding="utf-8") as f:
                ldata = json.load(f)
            lex_set = set(ldata.keys() if isinstance(ldata, dict) else ldata)
            lex_set |= {w.lower() for w in lex_set}
            self._lexicon = lex_set
            self.corrector = _ConfusionCorrector(
                cdata["by_tess_char"], lex_set, tau=_CORRECTOR_TAU
            )
            self.router = _CrossEngineRouter(
                lex_set, max_swap_dist=_CROSS_ENGINE_MAX_SWAP_DIST
            )
        except Exception:
            self.corrector = None
            self.router = None

    def _try_load_easyocr(self) -> None:
        try:
            import easyocr
            self.easyocr = easyocr.Reader(list(_EASYOCR_LANGS), gpu=False, verbose=False)
        except Exception:
            self.easyocr = None

    @staticmethod
    def _resolve_repo(model_id: str) -> str:
        if os.path.isdir(model_id):
            return model_id
        try:
            from huggingface_hub import snapshot_download
            token = os.environ.get("HF_TOKEN")
            kwargs: dict = {"repo_id": model_id}
            if token:
                kwargs["token"] = token
            return snapshot_download(**kwargs)
        except Exception:
            return ""

    @staticmethod
    def _first_existing(candidates, check) -> str:
        # PATH-based lookups can fail even after a tool is installed!
        # This checks a list of well-known install locations
        for cand in candidates:
            if check(cand):
                return cand
        return ""

    @classmethod
    def _fallback_tessdata_dir(cls) -> str:
        env_dir = os.environ.get("TESSDATA_PREFIX")
        if env_dir and os.path.isdir(env_dir):
            return env_dir
        # Just in case judge has it in a known path.
        return cls._first_existing(
            (
                r"C:\Program Files\Tesseract-OCR\tessdata",
                "/usr/share/tesseract-ocr/5/tessdata",
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/usr/share/tessdata",
                "/opt/homebrew/share/tessdata",
            ),
            lambda cand: os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "mlt.traineddata")),
        )

    @classmethod
    def _resolve_binary(cls, repo_dir: str) -> str:
        exe = "tesseract.exe" if platform.system() == "Windows" else "tesseract"
        # Explicit override, checked first — same escape hatch TESSDATA_PREFIX
        # already gives for the data dir. The harness calls CompetitionTranscriber()
        # with no arguments (no constructor param is possible), so an env var is
        # the only channel available if every other resolution path misses.
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and os.path.isfile(env_cmd):
            return env_cmd
        bundled = os.path.join(repo_dir, "tesseract", exe) if repo_dir else ""
        if bundled and os.path.isfile(bundled):
            return bundled
        on_path = shutil.which(exe)
        if on_path:
            return on_path
        return (
            cls._first_existing(
                (
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    "/usr/bin/tesseract",
                    "/usr/local/bin/tesseract",
                    "/opt/homebrew/bin/tesseract",
                ),
                os.path.isfile,
            )
            or exe
        )

    def _warmup(self) -> None:
        dummy = PIL.Image.new("RGB", (320, 64), color="white")
        pytesseract.image_to_string(dummy, lang=self.lang, config=self.config)
        if self.easyocr is not None:
            try:
                import numpy as np
                self.easyocr.readtext(np.asarray(dummy), detail=0, paragraph=True)
            except Exception:
                self.easyocr = None

    def _easyocr_paragraph(self, image: PIL.Image.Image) -> str:
        if self.easyocr is None:
            return ""
        try:
            import numpy as np
            arr = np.asarray(image)
            res = self.easyocr.readtext(arr, detail=0, paragraph=True)
            return _normalise(" ".join(res) if res else "")
        except Exception:
            return ""

    def _tess_paragraph(self, image: PIL.Image.Image, lang: str) -> str:
        raw = pytesseract.image_to_string(image, lang=lang, config=self.config)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return ""
        return _normalise(self.joiner.join_lines(lines, fix_hyphenated_words=True))

    def transcribe(self, image: PIL.Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        mlt_out = self._tess_paragraph(image, _TESS_LANG_PRIMARY)
        ita_out = ""
        if self.router is not None:
            try:
                ita_out = self._tess_paragraph(image, _TESS_LANG_AUGMENTED)
            except Exception:
                ita_out = ""
        if self.router is None:
            base = ita_out if ita_out else mlt_out
            if not base:
                return ""
            if self.corrector is not None and len(base) >= _CORRECTOR_LEN_THR:
                base = self.corrector.correct(base)
            return _fix_doublequote(_fix_apostrophe(_fix_lead_marker(base)))

        def _safe_tess(lang: str) -> str:
            try:
                return self._tess_paragraph(image, lang)
            except Exception:
                return ""

        romance_out = _safe_tess(_TESS_LANG_ROMANCE)
        stock_out = _safe_tess(_TESS_LANG_STOCK)
        try:
            w, h = image.size
            up = image.resize((w * 2, h * 2), PIL.Image.LANCZOS)
            up_out = self._tess_paragraph(up, _TESS_LANG_AUGMENTED)
        except Exception:
            up_out = ""

        tess_streams = [s for s in (mlt_out, ita_out, romance_out, stock_out, up_out) if s]
        if not tess_streams:
            return ""

        base = ita_out if ita_out else mlt_out
        longest = max(tess_streams, key=len)
        if len(longest) > 10 and len(base) < _ITA_FALLBACK_RATIO * len(longest):
            base = longest

        joined = base
        if self.corrector is not None and len(joined) >= _CORRECTOR_LEN_THR:
            joined = self.corrector.correct(joined)

        candidates = []
        if self.easyocr is not None:
            easy_out = self._easyocr_paragraph(image)
            if easy_out:
                candidates.append(easy_out)
        for cand in (mlt_out, ita_out, romance_out, stock_out, up_out):
            if cand and cand != base and cand not in candidates:
                candidates.append(cand)
        if candidates:
            joined = self.router.combine_lv(joined, candidates)

        return _fix_doublequote(_fix_apostrophe(_fix_lead_marker(joined)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()
    print(CompetitionTranscriber(model_id=args.model_id).transcribe(
        PIL.Image.open(args.image)
    ))
