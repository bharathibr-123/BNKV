"""Module 03: explain why a word costs many tokens.

Gap addressed: audits report how many tokens a language costs, and morphology
tools segment words, but nothing attributes a word's token count to causes.

For each word we split its tokens into:

* ``base``         1 token: the ideal, a word stored whole.
* ``vowel_split``  extra tokens forced because the pre-tokenizer cuts the word
                   at vowel signs before BPE runs (pieces cut - pieces with marks kept).
* ``vocab_gap``    the rest: the vocabulary has no merge for this word.
                   ``byte_pieces`` counts how many tokens are raw UTF-8 bytes.

We also report how well token boundaries line up with morpheme boundaries
(``grammar_alignment``): 1.0 means every cut the tokenizer makes is at a real
morpheme boundary; low values mean cuts fall inside meaningful parts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tokentax import morphology, registry
from tokentax.metrics import is_fragment
from tokentax.scripts import profile


@dataclass(frozen=True)
class WordCost:
    word: str
    tokens: int
    pieces: list[str]
    byte_flags: list[bool]
    base: int
    vowel_split: int
    vocab_gap: int
    byte_pieces: int
    morphemes: list[str]
    grammar_alignment: float | None
    # What grammar_alignment would be if the same number of cuts fell at random.
    alignment_chance: float | None

    @property
    def main_cause(self) -> str:
        if self.tokens <= max(1, len(self.morphemes)):
            return "efficient"
        if self.vowel_split >= self.vocab_gap and self.vowel_split > 0:
            return "vowel_split"
        if self.byte_pieces * 2 >= self.tokens:
            return "missing_script"
        return "vocab_gap"

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        d["main_cause"] = self.main_cause
        return d


def _token_pieces(key: str, word: str) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    enc = registry.load(key).encode(word, add_special_tokens=False)
    pieces = [word[a:b] for a, b in enc.offsets]
    return pieces, list(enc.ids), list(enc.offsets)


def explain_word(key: str, word: str, language: str) -> WordCost:
    pieces, ids, offsets = _token_pieces(key, word)
    n = len(ids)
    cut = len(registry.pretokens(key, word))
    kept = len(registry.pretokens(key, word, fixed=True))
    vowel = max(0, cut - kept)
    seg = morphology.segment(word, language)
    alignment: float | None = None
    chance: float | None = None
    token_cuts = {a for a, _ in offsets[1:] if a > 0}
    if morphology.supported(language) and token_cuts:
        alignment = len(token_cuts & seg.boundaries) / len(token_cuts)
        positions = max(len(word.strip()) - 1, 1)
        chance = min(1.0, len(seg.boundaries) / positions)
    return WordCost(
        word=word,
        tokens=n,
        pieces=pieces,
        byte_flags=[is_fragment(key, i) for i in ids],
        base=min(1, n),
        vowel_split=min(vowel, max(0, n - 1)),
        vocab_gap=max(0, n - 1 - min(vowel, max(0, n - 1))),
        byte_pieces=sum(is_fragment(key, i) for i in ids),
        morphemes=list(seg.pieces),
        grammar_alignment=alignment,
        alignment_chance=chance,
    )


@dataclass(frozen=True)
class TextExplanation:
    tokenizer: str
    language: str
    words: list[WordCost]

    @property
    def totals(self) -> dict[str, int]:
        return {
            "tokens": sum(w.tokens for w in self.words),
            "base": sum(w.base for w in self.words),
            "vowel_split": sum(w.vowel_split for w in self.words),
            "vocab_gap": sum(w.vocab_gap for w in self.words),
            "byte_pieces": sum(w.byte_pieces for w in self.words),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer": self.tokenizer,
            "language": self.language,
            "totals": self.totals,
            "words": [w.to_dict() for w in self.words],
        }


def explain(key: str, text: str, language: str | None = None) -> TextExplanation:
    """Explain the cost of every word.

    Byte tokens of one character share that character's text in ``pieces``;
    ``byte_flags`` marks them. Word costs are measured in isolation,
    so their sum can differ slightly from the token count of the full sentence."""
    lang = language or guess_language(text)
    words = [w for w in text.split() if w]
    return TextExplanation(key, lang, [explain_word(key, " " + w, lang) for w in words])


_SCRIPT_TO_LANG = {
    "Kannada": "kan_Knda", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym", "Devanagari": "hin_Deva", "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr", "Gurmukhi": "pan_Guru", "Odia": "ory_Orya",
    "Arabic": "urd_Arab", "Ol Chiki": "sat_Olck", "Latin": "eng_Latn",
}


def guess_language(text: str) -> str:
    """Pick a language from the dominant non-Latin script (English if none)."""
    prof = profile(text)
    native = {k: v for k, v in prof.counts.items() if k != "Latin"}
    if native:
        return _SCRIPT_TO_LANG.get(max(native.items(), key=lambda kv: kv[1])[0], "eng_Latn")
    return "eng_Latn"
