"""117-character round-trip gate for the submission post-processing chain.

Every character in competition_files/char_set.json must pass through the
deterministic normalisation + label-convention chain that transcribe() applies
to its base output, without crashing or being mangled. A single unhandled glyph
that corrupts a paragraph costs more CER than any single-version gain.

The chain under test is exactly:

    _fix_doublequote(_fix_apostrophe(_fix_lead_marker(_normalise(s))))

transcribe() is NOT called: it needs Tesseract and is per-image slow. The
corrector / router only mutate out-of-lexicon words against bundled lexicon
data and are not part of the per-glyph normalisation contract, so they are out
of scope here.

Two characters are intentionally transformed by a documented label rule and are
expected, not failures:
  - ASCII apostrophe U+0027 -> curly U+2018/U+2019 (positional, _fix_apostrophe)
  - ASCII double quote U+0022 -> curly U+201C/U+201D (positional, _fix_doublequote)
Both targets are themselves in char_set.json.
"""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest

import competition_transcriber as ct

REPO_ROOT = Path(__file__).resolve().parents[1]
_ORGANISER = REPO_ROOT / "competition_files" / "char_set.json"
CHAR_SET_PATH = _ORGANISER if _ORGANISER.exists() else REPO_ROOT / "data" / "char_set.json"

# Documented intentional normalisations. char -> set of acceptable outputs.
# ASCII quotes map to curly per the organiser gold convention (v17/v18/v20).
_EXPECTED_NORMALISED = {
    "'": {"‘", "’"},
    '"': {"“", "”"},
}

# Maltese diacritic letters that MUST NOT fold to their ASCII base.
_MUST_SURVIVE_DIAC = list("ĊċĠġĦħŻż")  # Ċċ Ġġ Ħħ Żż
_ASCII_FOLD = {
    "Ċ": "C", "ċ": "c", "Ġ": "G", "ġ": "g",
    "Ħ": "H", "ħ": "h", "Ż": "Z", "ż": "z",
}

# Grave-accented vowels: lowercase only, per char_set.json.
_GRAVE_VOWELS = list("àìòù")  # à ì ò ù
# Test-set-only glyphs absent from dev gold but present in char_set.json.
_TEST_SET_ONLY = list("=ô•♢")  # = ô • ♢


def _chain(s: str) -> str:
    return ct._fix_doublequote(ct._fix_apostrophe(ct._fix_lead_marker(ct._normalise(s))))


def _load_charset() -> list[str]:
    return json.loads(CHAR_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def charset() -> list[str]:
    cs = _load_charset()
    assert len(cs) == 117, f"expected 117 chars, char_set.json has {len(cs)}"
    return cs


def _roundtrip(char: str) -> tuple[bool, str, str]:
    """Run one char through the chain. Returns (ok, output, reason)."""
    s = f"test {char} word"
    try:
        out = _chain(s)
    except Exception as exc:  # crash = hard fail
        return False, "", f"crashed: {type(exc).__name__}: {exc}"
    if char in _EXPECTED_NORMALISED:
        targets = _EXPECTED_NORMALISED[char]
        hit = next((t for t in targets if t in out), None)
        if hit is None:
            return False, out, f"expected normalise to one of {targets}, got {out!r}"
        if char in out:
            return False, out, f"ASCII form {char!r} leaked through; should be curly"
        return True, out, f"normalised -> U+{ord(hit):04X}"
    if char not in out:
        return False, out, f"char dropped/mangled: {out!r}"
    return True, out, "preserved"


def test_charset_roundtrip_all(charset: list[str]) -> None:
    failures: list[dict] = []
    passed = 0
    for char in charset:
        ok, out, reason = _roundtrip(char)
        if ok:
            passed += 1
        else:
            failures.append(
                {"char": char, "codepoint": f"U+{ord(char):04X}",
                 "name": unicodedata.name(char, "?"), "reason": reason, "output": out}
            )
    assert not failures, (
        f"{len(failures)}/{len(charset)} chars failed round-trip:\n"
        + "\n".join(f"  {f['codepoint']} {f['name']}: {f['reason']}" for f in failures)
    )


def test_maltese_diacritics_never_fold(charset: list[str]) -> None:
    for ch in _MUST_SURVIVE_DIAC:
        assert ch in charset, f"{ch!r} missing from char_set.json"
        out = _chain(f"test {ch} word")
        assert ch in out, f"{ch!r} (U+{ord(ch):04X}) was dropped"
        assert _ASCII_FOLD[ch] not in out or out.count(_ASCII_FOLD[ch]) == 0, (
            f"{ch!r} folded to ASCII {_ASCII_FOLD[ch]!r}: {out!r}"
        )


def test_ghie_digraph_survives() -> None:
    for word in ("għadha", "Għadha", "għall", "ie", "biex"):
        out = _chain(f"test {word} word")
        assert word in out, f"digraph context {word!r} mangled: {out!r}"


def test_grave_vowels_survive() -> None:
    for ch in _GRAVE_VOWELS:
        out = _chain(f"libertà {ch} Ġesù")
        assert ch in out, f"grave vowel {ch!r} (U+{ord(ch):04X}) dropped: {out!r}"
    # No uppercase grave and no è in the inventory; chain must still not invent them.
    out = _chain("test è word")
    assert "è" in out  # passthrough: chain does not strip unknown accents


def test_test_set_only_glyphs_no_crash() -> None:
    for ch in _TEST_SET_ONLY:
        out = _chain(f"test {ch} word")
        assert ch in out, f"test-set-only {ch!r} (U+{ord(ch):04X}) dropped: {out!r}"


def test_en_dash_normalised_to_em_dash() -> None:
    # U+2013 is image-only; gold maps it to em-dash U+2014. Not in char_set.
    out = _chain("test – word")
    assert "—" in out, f"en-dash not mapped to em-dash: {out!r}"
    assert "–" not in out, f"en-dash leaked through: {out!r}"


def test_soft_hyphen_dropped() -> None:
    out = _chain("fis­ seħħ")  # soft hyphen removed
    assert "­" not in out, f"soft hyphen leaked: {out!r}"


# --- Ralph-loop adversarial challenges (positional / context, not per-glyph) ---


def test_quote_positions_in_real_paragraph() -> None:
    # C1: the single-char template cannot catch opening/closing position errors.
    # Opening double quote precedes an alnum after a space -> U+201C; closing
    # follows an alnum -> U+201D. Clitic apostrophes (ta', x') are closing U+2019.
    out = _chain('Il-kelb tiegħu, "dak il-ġuvni", ċempel lill-ħabib tiegħu.')
    assert "“dak" in out, f"opening double quote misplaced: {out!r}"
    assert "ġuvni”" in out, f"closing double quote misplaced: {out!r}"
    assert '"' not in out, f"ASCII double quote leaked: {out!r}"
    assert "ġuvni" in out and "ħabib" in out, f"diacritics mangled: {out!r}"

    clitic = _chain("ta' Pawlu u x'jiġri")
    assert clitic == "ta’ Pawlu u x’jiġri", f"clitic apostrophe wrong: {clitic!r}"

    quoted = _chain("Hu qal 'iva' lilha")
    assert quoted == "Hu qal ‘iva’ lilha", f"quoted-word apostrophes wrong: {quoted!r}"


def test_lead_marker_no_false_positive_on_ranges() -> None:
    # C2: a numeric range or N-il ordinal at paragraph start is NOT a clause
    # marker. A tight ASCII hyphen with no adjacent space must be preserved.
    for s in ("19-20 fil-parlament", "1-2 darba", "12-il sena", "5-darbiet"):
        out = _chain(s)
        assert out == s, f"lead-marker false positive on {s!r}: {out!r}"
    # Genuine markers must still be normalised to "N — word".
    assert _chain("0 - Għadha mhux fis-seħħ") == "0 — Għadha mhux fis-seħħ"
    assert _chain("3- Fis-seħħ") == "3 — Fis-seħħ"
    assert _chain("1—Ippjanata") == "1 — Ippjanata"
    # Em-dash marker before an il- word (real collision) still fires.
    assert _chain("1 — il-bniedem") == "1 — il-bniedem"


def test_lead_marker_real_dev_markers() -> None:
    # C4: the non-digit lookahead in the tightened _LEAD_MARKER must not miss
    # any genuine clause marker present in dev gold. Every gold paragraph that
    # starts "digit + dash + ..." must survive the chain unchanged (gold is
    # already canonical "N — word", so the chain must be idempotent on it).
    import glob
    paths = glob.glob(str(REPO_ROOT / "competition_files" / "**" / "texts.json"),
                      recursive=True)
    if not paths:
        pytest.skip("dev texts.json not found")
    data = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else list(data.values())
    texts = [(it.get("text") if isinstance(it, dict) else it) for it in items]
    import re
    fam = re.compile(r"^\s*\d{1,2}\s*[-–—]")
    markers = [t for t in texts if t and fam.match(t)]
    assert markers, "no marker-family paragraphs found in dev gold"
    corrupted = [(t, _chain(t)) for t in markers if _chain(t) != t]
    assert not corrupted, (
        "chain corrupted a real dev-gold clause marker:\n"
        + "\n".join(f"  {t!r} -> {o!r}" for t, o in corrupted)
    )


def test_router_never_strips_diacritic() -> None:
    # C3: plurality vote and the v19 diacritic-restoration guard must never
    # replace a diacritic letter with its ASCII base.
    lex = {"ġabra", "ġesù", "tiegħu", "jagħmel"}
    r = ct._CrossEngineRouter(lex, max_swap_dist=2)

    # 2 streams carry ġ, 1 carries g; out-of-lex anchor -> plurality ġ wins.
    out = r.combine_lv("jagmel ġabra",
                       ["jagħmel ġabra", "jagħmel ġabra", "jagmel gabra"])
    assert "jagħmel" in out and "ġabra" in out, f"plurality diacritic lost: {out!r}"

    # Correct anchor must survive even when 2 streams carry the ASCII form.
    out = r.combine_lv("ġabra tiegħu",
                       ["gabra tiegħu", "gabra tieghu", "ġabra tiegħu"])
    assert "ġabra" in out and "gabra" not in out, f"diacritic stripped: {out!r}"

    # _decide is asymmetric: never drops a diacritic, only restores one.
    assert r._decide("ġabra", "gabra") == "ġabra"   # candidate poorer -> keep anchor
    assert r._decide("gabra", "ġabra") == "ġabra"   # candidate richer -> restore


def test_lead_marker_survives_post_router_output() -> None:
    # C6: transcribe applies _fix_lead_marker AFTER combine_lv (lines 578-579),
    # so the C2 fix must hold on whatever token-0 the router emits. _fix_lead_marker
    # is the same call in both the router and no-router paths, so testing it
    # directly on a router-style token-0 proves the ordering is safe.
    assert ct._fix_lead_marker("19-20 fil-parlament") == "19-20 fil-parlament"
    assert ct._fix_lead_marker("1-2 darba") == "1-2 darba"
    assert ct._fix_lead_marker("0 — Għadha mhux fis-seħħ") == "0 — Għadha mhux fis-seħħ"
    assert ct._fix_lead_marker("0 - Għadha mhux fis-seħħ") == "0 — Għadha mhux fis-seħħ"


