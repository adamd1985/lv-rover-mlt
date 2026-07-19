"""Line joiner wrapper.

Wraps `malti.line_joiner.RBLineJoiner.join_lines(lines, fix_hyphenated_words=True)`.
The rule-based joiner already preserves structural prefixes (`il-`, `fis-`, ...)
via a builtin denylist. It treats U+002D and U+2014 as no-space-end characters.

The gold inventory has only two dash characters: ASCII hyphen U+002D and em-dash
U+2014. En-dash U+2013 is an image-only glyph - the model may decode it from an
en-dash drawn in the image, but it must never survive into the joined output.
We normalise every U+2013 to U+2014 up front, so en-dash-terminated lines flow
through the same code path as em-dash-terminated lines.
"""
from __future__ import annotations

from typing import List

from malti.line_joiner import RBLineJoiner

_EN_DASH = "–"
_EM_DASH = "—"
_ASCII_HYPHEN = "-"
_SOFT_HYPHEN = "­"

_joiner = RBLineJoiner()


def _normalise_en_dash(lines: List[str]) -> List[str]:
    """En-dash U+2013 is image-only; the canonical label always uses em-dash
    U+2014. Normalise every decoded en-dash before joining."""
    return [s.replace(_EN_DASH, _EM_DASH) for s in lines]


def join_lines(lines: List[str]) -> str:
    """Join OCR'd lines into a single paragraph string.

    Soft hyphens (U+00AD) are stripped before delegating to the malti
    joiner. The joiner's structural-clitic denylist is keyed on bare
    forms like 'għall-', and the last-word regex character class excludes
    U+00AD, so a soft hyphen embedded inside a clitic ('għa­ll-')
    silently falls out of the denylist and the line-break hyphen gets
    over-rejoined. Validation already strips U+00AD on the gold side, so
    pre-stripping here keeps both sides on equal footing.
    """
    if not lines:
        return ""
    clean = [s.replace(_SOFT_HYPHEN, "") for s in lines if s is not None]
    clean = _normalise_en_dash(clean)
    return _joiner.join_lines(clean, fix_hyphenated_words=True)


def join_paragraph(text: str) -> str:
    """Accepts either a multi-line raw decode or already-joined text."""
    if "\n" not in text:
        return text.strip()
    return join_lines(text.split("\n"))
