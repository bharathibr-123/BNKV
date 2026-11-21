"""Tokenizer cost metrics.

Every comparison across languages is made on *parallel* text (the same
sentences translated), so a ratio of token counts compares the cost of the same
meaning. Per-character or per-word rates are reported too, but they are not used
to rank languages, because their units differ between scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

from tokentax import registry


@dataclass(frozen=True)
class TokenStats:
    texts: int
    chars: int
    words: int
    tokens: int
    byte_fragments: int

    @property
    def fertility(self) -> float:
        """Tokens per whitespace word (lower is better)."""
        return self.tokens / max(self.words, 1)

    @property
    def chars_per_token(self) -> float:
        return self.chars / max(self.tokens, 1)

    @property
    def fragment_rate(self) -> float:
        """Share of tokens that are pieces of a character (raw UTF-8 bytes)."""
        return self.byte_fragments / max(self.tokens, 1)

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {k: float(v) for k, v in asdict(self).items()}
        d.update(
            fertility=self.fertility,
            chars_per_token=self.chars_per_token,
            fragment_rate=self.fragment_rate,
        )
        return d


@lru_cache(maxsize=1_000_000)
def is_fragment(key: str, token_id: int) -> bool:
    """True if the token is a raw byte piece of a character, not a whole character."""
    return "�" in registry.load(key).decode([token_id])


def stats(key: str, texts: list[str]) -> TokenStats:
    encs = registry.load(key).encode_batch(texts, add_special_tokens=False)
    tokens = sum(len(e.ids) for e in encs)
    frags = sum(is_fragment(key, i) for e in encs for i in e.ids)
    return TokenStats(
        texts=len(texts),
        chars=sum(len(t) for t in texts),
        words=sum(len(t.split()) for t in texts),
        tokens=tokens,
        byte_fragments=frags,
    )


def premium(lang_tokens: int, english_tokens: int) -> float:
    """How many times more tokens a language needs than English for the same text."""
    return lang_tokens / max(english_tokens, 1)


def single_token_rate(key: str, words: list[str]) -> float:
    """Share of words the tokenizer keeps as one token (STRR, higher is better).

    Words are tested with a leading space, the way they appear mid-sentence.
    """
    if not words:
        return 0.0
    counts = registry.count_batch(key, [" " + w for w in words])
    return sum(c == 1 for c in counts) / len(words)
