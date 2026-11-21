"""Scripts, romanisation and morphology: pure functions, no tokenizers needed."""

from __future__ import annotations

import pytest

from tokentax import morphology, romanize
from tokentax.languages import get
from tokentax.scripts import Style, profile

SAMPLES = {
    "kan": "ನಾನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ.",
    "tam": "நான் நாளை கல்லூரிக்குச் செல்வேன்.",
    "tel": "నేను రేపు కాలేజీకి వెళ్తాను.",
    "mal": "ഞാൻ നാളെ കോളേജിൽ പോകും.",
    "hin": "मैं कल कॉलेज जाऊँगा।",
    "ben": "আমি কাল কলেজে যাব।",
    "guj": "હું કાલે કોલેજ જઈશ.",
}


@pytest.mark.parametrize("code", sorted(SAMPLES))
def test_lossless_romanisation_round_trips(code: str) -> None:
    lang = get(code)
    assert romanize.roundtrip_ok(SAMPLES[code], lang.script)


def test_casual_spellings_look_like_real_typing() -> None:
    assert romanize.casual(SAMPLES["kan"], get("kan").script) == (
        "naanu naale bengaloorige hoguttene."
    )
    assert romanize.casual(SAMPLES["tam"], get("tam").script) == (
        "naan naalai kalloorikkuch chelven."
    )
    hindi = get("hin")
    assert romanize.casual(SAMPLES["hin"], hindi.script, hindi.hindi_like_spelling) == (
        "main kal kolej jaaoongaa."
    )
    assert romanize.casual(SAMPLES["mal"], get("mal").script).endswith("pokum.")


def test_unsupported_scripts_pass_through() -> None:
    urdu = get("urd")
    assert romanize.casual("یہ ایک جملہ ہے", urdu.script) == "یہ ایک جملہ ہے"


@pytest.mark.parametrize(
    ("text", "style"),
    [
        ("ನಾಳೆ ಬನ್ನಿ", Style.NATIVE),
        ("naale banni", Style.ROMAN),
        ("ನಾಳೆ meeting ಇದೆ please ಬನ್ನಿ", Style.MIXED),
        ("123 !!", Style.OTHER),
    ],
)
def test_style_detection(text: str, style: Style) -> None:
    assert profile(text).style is style


@pytest.mark.parametrize(
    ("word", "lang", "pieces"),
    [
        ("ಮನೆಗಳಲ್ಲಿ", "kan_Knda", ("ಮನೆ", "ಗಳ", "ಲ್ಲಿ")),
        ("ಹೋಗುತ್ತೇನೆ", "kan_Knda", ("ಹೋಗು", "ತ್ತೇನೆ")),
        ("வீடுகளில்", "tam_Taml", ("வீடு", "கள", "ில்")),
        ("घरों", "hin_Deva", ("घर", "ों")),
        ("ನಾನು", "kan_Knda", ("ನಾನು",)),
    ],
)
def test_morphology(word: str, lang: str, pieces: tuple[str, ...]) -> None:
    seg = morphology.segment(word, lang)
    assert seg.pieces == pieces
    assert "".join(seg.pieces) == word


def test_morphology_boundaries_are_offsets() -> None:
    seg = morphology.segment("ಮನೆಗಳಲ್ಲಿ", "kan_Knda")
    assert seg.boundaries == frozenset({3, 5})


def test_malayalam_is_not_pro_drop() -> None:
    assert get("kan").pro_drop and get("tam").pro_drop
    assert not get("mal").pro_drop and not get("hin").pro_drop
