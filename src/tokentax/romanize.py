"""Convert Indic text to Latin script in two ways.

* ``casual``: the way people actually type on WhatsApp ("naanu naale baruttene").
  No diacritics, so it is lossy: "ಟ" and "ತ" both become "t".
* ``lossless``: ISO 15919. It uses diacritics ("nānu nāḷe") but converts back to
  exactly the same native text, so meaning cannot change.

Tamil goes through Devanagari (the Unicode blocks share a layout), because the
direct Tamil scheme in indic_transliteration mis-reads plain consonants.
Scripts that indic_transliteration does not support (Arabic, Ol Chiki, Sinhala)
are returned unchanged.
"""

from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from tokentax.languages import DEVANAGARI, TAMIL, Script

_TAMIL_OFFSET = TAMIL.start - DEVANAGARI.start

# Malayalam "chillu" letters are atomic forms of consonant + virama.
_CHILLU = str.maketrans({
    "ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
    "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്",
})

# ISO 15919 -> casual spelling. Order matters: longer sequences first.
_FLATTEN: tuple[tuple[str, str], ...] = (
    ("ṁp", "mp"), ("ṁb", "mb"), ("ṁm", "mm"), ("m̐", "n"),
    ("ā", "aa"), ("ī", "ee"), ("ū", "oo"), ("ē", "e"), ("ō", "o"), ("ô", "o"), ("ê", "e"),
    ("r̥̄", "roo"), ("r̥", "ru"), ("l̥", "lu"),
    ("ṁ", "n"), ("ḥ", "h"), ("ṅ", "n"), ("ñ", "n"), ("ṇ", "n"), ("ṉ", "n"),
    ("ṭ", "t"), ("ḍ", "d"), ("ṟ", "r"), ("ṛ", "r"), ("ṯ", "t"),
    ("ś", "sh"), ("ṣ", "sh"), ("ḷ", "l"), ("ḻ", "zh"), ("ẏ", "y"), ("ġ", "g"), ("ḵ", "k"),
)
_COMBINING = re.compile(r"[̀-ͯ‌‍]")
_FINAL_ANUSVARA = re.compile(r"ṁ(?=\W|$)")
_FINAL_SCHWA = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b")


def _nukta_letters() -> dict[str, str]:
    """base + nukta -> precomposed letter, for every Indic block (from Unicode data)."""
    out: dict[str, str] = {}
    for cp in range(0x0900, 0x0D80):
        parts = unicodedata.decomposition(chr(cp)).split()
        if len(parts) == 2 and not parts[0].startswith("<"):
            base, mark = (chr(int(x, 16)) for x in parts)
            if unicodedata.name(mark, "").endswith("NUKTA"):
                out[base + mark] = chr(cp)
    return out


_NUKTA_LETTERS = _nukta_letters()
_CANDRA = {"\u0ac5": "\u0ac7", "\u0ac9": "\u0acb", "\u0945": "\u0947", "\u0949": "\u094b"}


def _prepare(text: str, script: Script) -> tuple[str, str]:
    """Return (text, scheme) ready for indic_transliteration."""
    text = unicodedata.normalize("NFC", text).translate(_CHILLU)
    # NFC splits these Bengali letters into base + nukta, which the converter does
    # not know; give it the single code points it does know.
    text = text.replace("য়", "য়").replace("ড়", "ড়").replace("ঢ়", "ঢ়")
    if script is TAMIL:
        shifted = "".join(
            chr(ord(ch) - _TAMIL_OFFSET) if TAMIL.contains(ch) else ch for ch in text
        )
        return shifted, sanscript.DEVANAGARI
    assert script.sanscript is not None
    return text, script.sanscript


def lossless(text: str, script: Script) -> str:
    """Reversible romanisation (ISO 15919)."""
    if script.sanscript is None:
        return text
    prepared, scheme = _prepare(text, script)
    out = str(transliterate(prepared, scheme, sanscript.ISO))
    left = {c for c in out if 0x0900 <= ord(c) < 0x0D80 and c not in "।॥"}
    if left:
        # signs the converter does not know (Gurmukhi nukta, Gujarati candra
        # vowels ...): use the nearest plain letter. The result no longer
        # converts back exactly, and ``roundtrip_ok`` reports that.
        plain = "".join(_CANDRA.get(c, "" if c in left else c) for c in prepared)
        out = str(transliterate(plain, scheme, sanscript.ISO))
    return out


def to_native(text: str, script: Script) -> str:
    """Convert ISO 15919 back to the native script."""
    if script.sanscript is None:
        return text
    if script is TAMIL:
        deva = str(transliterate(text, sanscript.ISO, sanscript.DEVANAGARI))
        return "".join(
            chr(ord(ch) + _TAMIL_OFFSET) if DEVANAGARI.contains(ch) else ch for ch in deva
        )
    return str(transliterate(text, sanscript.ISO, script.sanscript))


def roundtrip_ok(text: str, script: Script) -> bool:
    """True if lossless romanisation converts back to the same text."""
    original = unicodedata.normalize("NFC", text).translate(_CHILLU)
    back = unicodedata.normalize("NFC", to_native(lossless(text, script), script))
    return back == original


def casual(text: str, script: Script, indo_aryan: bool = False) -> str:
    """Informal romanisation, close to how Kanglish or Hinglish is typed.

    ``indo_aryan`` applies two spelling habits of Hindi-like languages: the silent
    final "a" is dropped ("kala" -> "kal") and a final nasal is written "n"
    ("maiṁ" -> "main"). Dravidian languages keep both ("pōkuṁ" -> "pokum").
    """
    if script.sanscript is None:
        return text
    out = _FINAL_ANUSVARA.sub("n" if indo_aryan else "m", lossless(text, script).lower())
    out = out.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")
    for src, dst in _FLATTEN:
        out = out.replace(src, dst)
    out = _COMBINING.sub("", unicodedata.normalize("NFD", out))
    out = unicodedata.normalize("NFC", out).replace("।", ".").replace("॥", ".")
    # stray script marks the converter left (e.g. Bengali nukta in "য়") are dropped
    out = "".join(c for c in out
                  if not (script.contains(c) and unicodedata.category(c) == "Mn"))
    if indo_aryan:
        out = _FINAL_SCHWA.sub("", out)
    return out
