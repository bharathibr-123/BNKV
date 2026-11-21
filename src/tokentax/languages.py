"""The languages we study and the scripts they are written in."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Script:
    name: str
    start: int
    end: int
    sanscript: str | None  # indic_transliteration scheme name, None if unsupported

    def contains(self, ch: str) -> bool:
        return self.start <= ord(ch) <= self.end


DEVANAGARI = Script("Devanagari", 0x0900, 0x097F, "devanagari")
BENGALI = Script("Bengali", 0x0980, 0x09FF, "bengali")
GURMUKHI = Script("Gurmukhi", 0x0A00, 0x0A7F, "gurmukhi")
GUJARATI = Script("Gujarati", 0x0A80, 0x0AFF, "gujarati")
ODIA = Script("Odia", 0x0B00, 0x0B7F, "oriya")
TAMIL = Script("Tamil", 0x0B80, 0x0BFF, "tamil")
TELUGU = Script("Telugu", 0x0C00, 0x0C7F, "telugu")
KANNADA = Script("Kannada", 0x0C80, 0x0CFF, "kannada")
MALAYALAM = Script("Malayalam", 0x0D00, 0x0D7F, "malayalam")
SINHALA = Script("Sinhala", 0x0D80, 0x0DFF, None)
ARABIC = Script("Arabic", 0x0600, 0x06FF, None)
OL_CHIKI = Script("Ol Chiki", 0x1C50, 0x1C7F, None)
LATIN = Script("Latin", 0x0041, 0x024F, None)

INDIC_SCRIPTS: tuple[Script, ...] = (
    DEVANAGARI, BENGALI, GURMUKHI, GUJARATI, ODIA, TAMIL, TELUGU, KANNADA, MALAYALAM,
    SINHALA, ARABIC, OL_CHIKI,
)


@dataclass(frozen=True)
class Language:
    code: str  # FLORES-200 code, e.g. kan_Knda
    name: str
    script: Script
    family: str
    # Verbs carry person agreement, so a subject pronoun is often redundant.
    pro_drop: bool = False

    @property
    def short(self) -> str:
        return self.code.split("_")[0]

    @property
    def hindi_like_spelling(self) -> bool:
        """Romanised casually with a silent final "a" (Hindi "kal", not "kala")."""
        return self.family == "Indo-Aryan" and self.code != "san_Deva"


ENGLISH = Language("eng_Latn", "English", LATIN, "Germanic")

SOUTH_ASIAN: tuple[Language, ...] = (
    Language("hin_Deva", "Hindi", DEVANAGARI, "Indo-Aryan"),
    Language("ben_Beng", "Bengali", BENGALI, "Indo-Aryan"),
    Language("mar_Deva", "Marathi", DEVANAGARI, "Indo-Aryan"),
    Language("guj_Gujr", "Gujarati", GUJARATI, "Indo-Aryan"),
    Language("pan_Guru", "Punjabi", GURMUKHI, "Indo-Aryan"),
    Language("ory_Orya", "Odia", ODIA, "Indo-Aryan"),
    Language("asm_Beng", "Assamese", BENGALI, "Indo-Aryan"),
    Language("npi_Deva", "Nepali", DEVANAGARI, "Indo-Aryan"),
    Language("mai_Deva", "Maithili", DEVANAGARI, "Indo-Aryan"),
    Language("urd_Arab", "Urdu", ARABIC, "Indo-Aryan"),
    Language("snd_Arab", "Sindhi", ARABIC, "Indo-Aryan"),
    Language("san_Deva", "Sanskrit", DEVANAGARI, "Indo-Aryan"),
    Language("kan_Knda", "Kannada", KANNADA, "Dravidian", pro_drop=True),
    Language("tam_Taml", "Tamil", TAMIL, "Dravidian", pro_drop=True),
    Language("tel_Telu", "Telugu", TELUGU, "Dravidian", pro_drop=True),
    # Malayalam verbs do not mark person, so dropping a pronoun loses information.
    Language("mal_Mlym", "Malayalam", MALAYALAM, "Dravidian", pro_drop=False),
    Language("sat_Olck", "Santali", OL_CHIKI, "Austroasiatic"),
)

ALL_LANGUAGES: tuple[Language, ...] = (ENGLISH, *SOUTH_ASIAN)
BY_CODE: dict[str, Language] = {lang.code: lang for lang in ALL_LANGUAGES}
BY_SHORT: dict[str, Language] = {lang.short: lang for lang in ALL_LANGUAGES}


def get(code: str) -> Language:
    """Look up a language by FLORES code (kan_Knda) or short code (kan)."""
    if code in BY_CODE:
        return BY_CODE[code]
    if code in BY_SHORT:
        return BY_SHORT[code]
    raise KeyError(f"unknown language: {code}")
