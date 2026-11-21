"""Text normalisation and a script-independent "canonical" form.

``normalize`` is a candidate transformation: Unicode NFC, one space between
words, and the danda (। ॥) written as a full stop. It never removes zero-width
joiners, because in Kannada and Malayalam they change how letters are drawn.

``canonical`` is used only for *checking* meaning, never sent to a model: it
writes every word in the same casual Latin spelling, so the original and a
romanised rewrite can be compared word for word.
"""

from __future__ import annotations

import re
import unicodedata

from tokentax import romanize
from tokentax.languages import Language

_SPACES = re.compile(r"[ \t ]+")
_COMBINING = re.compile(r"[̀-ͯ]")
_INDIC_DIGITS = {
    **{chr(0x0966 + i): str(i) for i in range(10)},  # Devanagari
    **{chr(0x09E6 + i): str(i) for i in range(10)},  # Bengali
    **{chr(0x0A66 + i): str(i) for i in range(10)},  # Gurmukhi
    **{chr(0x0AE6 + i): str(i) for i in range(10)},  # Gujarati
    **{chr(0x0B66 + i): str(i) for i in range(10)},  # Odia
    **{chr(0x0BE6 + i): str(i) for i in range(10)},  # Tamil
    **{chr(0x0C66 + i): str(i) for i in range(10)},  # Telugu
    **{chr(0x0CE6 + i): str(i) for i in range(10)},  # Kannada
    **{chr(0x0D66 + i): str(i) for i in range(10)},  # Malayalam
}
_DIGITS = str.maketrans(_INDIC_DIGITS)


def normalize(text: str) -> str:
    out = unicodedata.normalize("NFC", text)
    out = out.replace("।", ".").replace("॥", ".")
    return _SPACES.sub(" ", out).strip()


def ascii_digits(text: str) -> str:
    return text.translate(_DIGITS)


def _flatten_latin(word: str, hindi_like: bool, iso: bool) -> str:
    """Bring a Latin word to the casual spelling.

    ``iso``: the text uses ISO 15919 (c = ch-sound, ch = aspirated), so apply the
    same rules ``romanize.casual`` applies. Otherwise the word is already casual
    Latin (or English) and is only lower-cased.
    """
    out = word.lower()
    if iso:
        out = romanize._FINAL_ANUSVARA.sub("n" if hindi_like else "m", out)
        out = out.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")
    if not out.isascii():
        for src, dst in romanize._FLATTEN:
            out = out.replace(src, dst)
        out = unicodedata.normalize("NFC", _COMBINING.sub("", unicodedata.normalize("NFD", out)))
    return romanize._FINAL_SCHWA.sub("", out) if hindi_like else out


_ISO_LETTERS = set("āīūēōṛṝḷḹṁṃḥṅñṭḍṇśṣẏṟṉḻ")


def _looks_iso(text: str, lang: Language) -> bool:
    """True if the text uses ISO 15919 letters (ā, ṭ, ś ...), not just any accent."""
    return any(c in _ISO_LETTERS for c in unicodedata.normalize("NFC", text.lower()))


def canonical_words(text: str, lang: Language) -> list[str]:
    """Words of ``text`` in one casual Latin spelling, digits as ASCII, no punctuation."""
    words: list[str] = []
    iso = _looks_iso(text, lang)
    # no NFC here: the romaniser must see the same code points the rewrite saw
    light = _SPACES.sub(" ", text.replace("।", ".").replace("॥", ".")).strip()
    for raw in ascii_digits(light).split(" "):
        if not raw:
            continue
        if lang.script.sanscript is not None and any(
                lang.script.contains(c) and unicodedata.category(c)[0] == "L" for c in raw):
            # romanise only the native runs; a Latin run inside the word ("QVCಯಲ್ಲಿ")
            # is treated like an English word
            parts = re.findall(r"[^\x00-\x7f]+|[\x00-\x7f]+", raw)
            word = "".join(
                romanize.casual(part, lang.script, lang.hindi_like_spelling)
                if any(lang.script.contains(c) for c in part)
                else _flatten_latin(part, False, False)
                for part in parts)
        elif raw.isascii() or any(c.isalpha() for c in raw):
            # ISO output is lower-case, so a capitalised word was typed in English
            english = any(c.isupper() for c in raw)
            word = _flatten_latin(raw, lang.hindi_like_spelling and not english,
                                  iso and not english)
        else:
            word = raw.lower()
        word = re.sub(r"[^\w]+", "", word)
        if word:
            words.append(word)
    return words


def canonical(text: str, lang: Language) -> str:
    return " ".join(canonical_words(text, lang))
