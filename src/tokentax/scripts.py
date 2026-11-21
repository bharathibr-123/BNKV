"""Character-level script analysis: which scripts a text uses and how it mixes them."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from tokentax.languages import INDIC_SCRIPTS, LATIN, Script


class Style(str, Enum):
    """How a piece of text is written."""

    NATIVE = "native"  # only the language's own script
    ROMAN = "roman"  # only Latin letters
    MIXED = "mixed"  # both, e.g. "ನಾಳೆ meeting ಇದೆ"
    OTHER = "other"


def script_of(ch: str) -> Script | None:
    """Return the script a letter or mark belongs to, or None for spaces, digits, punctuation."""
    if ch.isascii():
        return LATIN if ch.isalpha() else None
    for script in INDIC_SCRIPTS:
        if script.contains(ch):
            cat = unicodedata.category(ch)
            return script if cat[0] in ("L", "M") else None
    if LATIN.contains(ch) and ch.isalpha():
        return LATIN
    return None


@dataclass(frozen=True)
class ScriptProfile:
    counts: dict[str, int]
    letters: int

    @property
    def dominant(self) -> str | None:
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda kv: kv[1])[0]

    def share(self, name: str) -> float:
        return self.counts.get(name, 0) / self.letters if self.letters else 0.0

    @property
    def style(self) -> Style:
        if not self.letters:
            return Style.OTHER
        latin = self.share(LATIN.name)
        if latin >= 0.95:
            return Style.ROMAN
        if latin <= 0.05:
            return Style.NATIVE
        return Style.MIXED


def profile(text: str) -> ScriptProfile:
    counts: Counter[str] = Counter()
    for ch in text:
        script = script_of(ch)
        if script is not None:
            counts[script.name] += 1
    return ScriptProfile(dict(counts), sum(counts.values()))


def is_combining_mark(ch: str) -> bool:
    """Vowel signs (matras), viramas and other marks that attach to a letter."""
    return unicodedata.category(ch).startswith("M")
