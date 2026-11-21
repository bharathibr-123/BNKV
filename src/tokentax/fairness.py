"""Module 02: the Language Fairness Index (LFI).

Gap addressed: tokenizer metrics are usually reported one at a time, and a
per-character or per-word unit can flip a conclusion between scripts. The LFI
combines five ratios, each measured against English on the same content, into
one number per language and tokenizer. English scores exactly 1.00; lower is
less fair. Every part is also reported on its own so the average hides nothing.

Parts (each in [0, 1], 1 = as good as English):

1. cost_parity        English tokens / language tokens on parallel FLORES text.
                      API price, context use and generation time all scale with it.
2. typing_parity      the same, for casually romanised text (how many people type).
3. word_integrity     share of common words kept as one token, relative to English.
4. char_integrity     1 - share of tokens that are raw byte pieces of a character.
5. pretoken_integrity pre-tokenizer pieces with marks kept / pieces actually cut.
                      Below 1 means vowel signs are split off before BPE starts.

Model answer quality is not part of the index: it needs model runs, not
tokenizers, and is reported separately when available.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

from tokentax import corpus, metrics, registry
from tokentax.languages import ENGLISH, SOUTH_ASIAN, Language

FLOOR = 0.01
PARTS = ("cost_parity", "typing_parity", "word_integrity", "char_integrity", "pretoken_integrity")


@dataclass(frozen=True)
class FairnessScore:
    tokenizer: str
    language: str
    cost_parity: float
    typing_parity: float | None
    word_integrity: float
    char_integrity: float
    pretoken_integrity: float

    @property
    def index(self) -> float:
        """Geometric mean of the available parts (one weak part pulls it down).

        Parts are floored at 0.01 so a single zero does not erase the others.
        """
        parts = [getattr(self, p) for p in PARTS if getattr(self, p) is not None]
        return math.exp(sum(math.log(max(p, FLOOR)) for p in parts) / len(parts))

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        d["index"] = self.index
        return d


def is_word(token: str) -> bool:
    """True if every character is a letter or a combining mark (no digits or punctuation)."""
    return bool(token) and all(unicodedata.category(ch)[0] in ("L", "M") for ch in token)


def frequent_words(lang: Language, n: int = 500) -> list[str]:
    counts: Counter[str] = Counter()
    for line in corpus.flores(lang.code, "dev"):
        counts.update(w.lower() for w in line.split() if is_word(w))
    return [w for w, _ in counts.most_common(n)]


def pretoken_integrity(key: str, texts: list[str]) -> float:
    words: Counter[str] = Counter(w for t in texts for w in t.split())
    cut = fixed = 0
    for word, n in words.items():
        cut += n * len(registry.pretokens(key, word))
        fixed += n * len(registry.pretokens(key, word, fixed=True))
    return fixed / max(cut, 1)


def _clip(x: float) -> float:
    return min(1.0, x)


def score(key: str, lang: Language, english_tokens: int, english_strr: float) -> FairnessScore:
    native = corpus.flores(lang.code)
    st = metrics.stats(key, native)
    typing: float | None = None
    if lang.script.sanscript is not None:
        roman_tokens = sum(registry.count_batch(key, corpus.roman(lang, native)))
        typing = _clip(english_tokens / max(roman_tokens, 1))
    strr = metrics.single_token_rate(key, frequent_words(lang))
    return FairnessScore(
        tokenizer=key,
        language=lang.code,
        cost_parity=_clip(english_tokens / max(st.tokens, 1)),
        typing_parity=typing,
        word_integrity=_clip(strr / max(english_strr, 1e-9)),
        char_integrity=1.0 - st.fragment_rate,
        pretoken_integrity=_clip(pretoken_integrity(key, native)),
    )


def all_scores(keys: list[str] | None = None) -> list[FairnessScore]:
    keys = keys or [s.key for s in registry.available()]
    english = corpus.flores(ENGLISH.code)
    eng_words = frequent_words(ENGLISH)
    out: list[FairnessScore] = []
    for key in keys:
        eng_tokens = metrics.stats(key, english).tokens
        eng_strr = metrics.single_token_rate(key, eng_words)
        out.extend(score(key, lang, eng_tokens, eng_strr) for lang in SOUTH_ASIAN)
    return out
