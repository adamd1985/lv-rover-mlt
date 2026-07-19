import pytest

from src.joiner.joiner import join_lines, join_paragraph


@pytest.mark.parametrize("lines,expected", [
    (["Għadha mhux fis-", "seħħ"], "Għadha mhux fis-seħħ"),
    (["Il-kelb qed", "jiġri"], "Il-kelb qed jiġri"),
    (["id-dar", "il-kbira"], "id-dar il-kbira"),
    (["multi-", "lingwi"], "multilingwi"),
    (["plain", "lines"], "plain lines"),
])
def test_join_lines(lines, expected):
    assert join_lines(lines) == expected


def test_empty():
    assert join_lines([]) == ""


def test_ascii_hyphen_line_break_rejoin():
    assert join_lines(["multi-", "lingwi"]) == "multilingwi"


def test_en_dash_line_end_normalised_then_joined():
    # En-dash at line end is normalised to em-dash, then the malti joiner
    # suppresses the inter-line space (em-dash is a no-space-end char).
    assert join_lines(["en–", "dash"]) == "en—dash"


def test_en_dash_normalised_to_em_dash():
    # En-dash U+2013 is image-only; the joiner normalises it to em-dash U+2014.
    out = join_lines(["0 – Għadha", "mhux"])
    assert "–" not in out
    assert out == "0 — Għadha mhux"


def test_structural_il_survives():
    assert join_lines(["il-kelb", "qed jiġri"]) == "il-kelb qed jiġri"


def test_structural_fis_survives_within_word_break():
    assert join_lines(["Għadha mhux fis-", "seħħ"]) == "Għadha mhux fis-seħħ"


def test_en_dash_normalised_in_worked_example():
    # Worked example from the spec: image en-dash maps to em-dash in the label,
    # structural fis- hyphen preserved, soft hyphen removed and word rejoined.
    out = join_lines(["0 – Għadha mhux fis-", "seħħ"])
    assert out == "0 — Għadha mhux fis-seħħ"


def test_em_dash_preserved():
    out = join_lines(["foo — bar", "baz"])
    assert "—" in out


def test_url_dash_not_rejoined():
    out = join_lines(["see http://a.com/x-", "y"])
    assert "x-y" in out


def test_number_dash_not_rejoined():
    out = join_lines(["page 123-", "456"])
    assert "123-456" in out


def test_join_paragraph_single_line():
    assert join_paragraph("Il-kelb") == "Il-kelb"


def test_join_paragraph_multi_line():
    assert join_paragraph("Għadha mhux fis-\nseħħ") == "Għadha mhux fis-seħħ"


def test_soft_hyphen_inside_clitic_preserves_structural_dash():
    # Renderer wrap may inject U+00AD inside a structural clitic, e.g.
    # 'għa­ll-' followed by 'pubbliku'. Without pre-stripping U+00AD, the
    # malti last-word regex captures only 'll-', misses the denylist, and
    # rejoins the line as 'għallpubbliku' (hyphen lost). With pre-strip
    # the clitic survives intact.
    out = join_lines(["infetaħ għa­ll-", "pubbliku bħala mużew"])
    assert "għall-pubbliku" in out


def test_soft_hyphen_stripped_from_output():
    # Soft hyphens never survive into the joined output; the joiner
    # mirrors the validation contract.
    out = join_lines(["po­litiku", "ieħor"])
    assert "­" not in out
