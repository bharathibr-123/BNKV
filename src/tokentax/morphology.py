"""A light, rule-based suffix segmenter for South Asian languages.

It strips known inflectional suffixes from the end of a word, longest first,
repeatedly, and returns the surface pieces ("ಮನೆಗಳಲ್ಲಿ" -> "ಮನೆ", "ಗಳ", "ಲ್ಲಿ").
Boundaries are character positions in the written word, which is what we need
to compare them with token boundaries. It is a heuristic, not a full analyser:
it does not undo sandhi and it can over-strip short words, so stems shorter than
``MIN_STEM`` characters are never cut.

Suffix lists cover common plural, case, and verb-agreement endings.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_STEM = 2

SUFFIXES: dict[str, tuple[str, ...]] = {
    "kan_Knda": (
        # plural and case
        "ಗಳು", "ಗಳ", "ನ್ನು", "ಲ್ಲಿ", "ಲ್ಲೇ", "ಕ್ಕೆ", "ಗೆ", "ದಿಂದ", "ಯಿಂದ", "ನಿಂದ", "ರಿಂದ",
        "ವರೆಗೆ", "ಕ್ಕಾಗಿ", "ಗಾಗಿ", "ಯನ್ನು", "ವನ್ನು", "ದ", "ನ", "ಯ", "ರ",
        # verb endings
        "ತ್ತೇನೆ", "ತ್ತೀಯ", "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತದೆ", "ತ್ತೇವೆ", "ತ್ತೀರಿ", "ತ್ತಾರೆ",
        "ತ್ತಿದ್ದಾರೆ", "ತ್ತಿದ್ದಾನೆ", "ತ್ತಿದ್ದೇನೆ", "ತ್ತಿದೆ", "ದನು", "ದಳು", "ದರು", "ದೆನು", "ದೆ",
        "ಲು", "ಬೇಕು", "ಬಹುದು", "ತ್ತಿ",
    ),
    "tam_Taml": (
        "கள்", "கள", "ில்", "ிடம்", "ுக்கு", "க்கு", "ுடன்", "ோடு",
        "ால்", "ிலிருந்து", "ுடைய", "ின்", "ை",
        "கிறேன்", "கிறாய்", "கிறான்", "கிறாள்", "கிறது", "கிறோம்", "கிறார்கள்",
        "ந்தேன்", "த்தேன்", "ந்தான்", "த்தான்", "வேன்", "ப்பேன்", "வோம்", "ேன்", "ான்",
        "ாள்", "ோம்", "ார்கள்", "ும்",
    ),
    "tel_Telu": (
        "లు", "లలో", "లకు", "లను", "లో", "కి", "కు", "ని", "ను", "తో", "నుండి", "యొక్క",
        "తాను", "తావు", "తాడు", "తుంది", "తాము", "తారు", "ాను", "ావు", "ాడు", "ింది",
        "ాము", "ారు",
    ),
    "mal_Mlym": (
        "കൾ", "കള", "ിൽ", "ൽ", "ിന്", "ക്ക്", "ുടെ", "ോട്", "ിൽനിന്ന്", "െ",
        "ുന്നു", "ും", "ുമോ", "ാൻ",
    ),
    "hin_Deva": (
        "ियों", "ाओं", "ों", "ें", "ाएँ", "ाएं", "ता", "ती", "ते", "ना", "ने", "कर",
        "ेगा", "ेगी", "ेंगे", "ूँगा", "ूंगा", "ऊँगा", "ऊंगा", "ाएगा", "ाएगी",
    ),
}


@dataclass(frozen=True)
class Segmentation:
    word: str
    pieces: tuple[str, ...]

    @property
    def stem(self) -> str:
        return self.pieces[0]

    @property
    def suffixes(self) -> tuple[str, ...]:
        return self.pieces[1:]

    @property
    def boundaries(self) -> frozenset[int]:
        """Character offsets inside the word where one piece ends and the next begins."""
        out: set[int] = set()
        pos = 0
        for piece in self.pieces[:-1]:
            pos += len(piece)
            out.add(pos)
        return frozenset(out)


def segment(word: str, language: str) -> Segmentation:
    suffixes = sorted(SUFFIXES.get(language, ()), key=len, reverse=True)
    stem = word
    found: list[str] = []
    changed = True
    while changed and suffixes:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf) and len(stem) - len(suf) >= MIN_STEM:
                found.insert(0, suf)
                stem = stem[: -len(suf)]
                changed = True
                break
    return Segmentation(word, (stem, *found))


def supported(language: str) -> bool:
    return language in SUFFIXES
