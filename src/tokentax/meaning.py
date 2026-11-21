"""Meaning checks for rewritten text.

Three levels, from strongest to weakest:

1. ``exact``: the rewrite converts back to the original character for
   character (lossless romanisation), so meaning cannot change.
2. ``embedding``: cosine similarity of LaBSE sentence embeddings, if the
   optional ``sentence-transformers`` package and model are installed.
3. ``spelling_ambiguity``: for casual romanisation, the share of words whose
   Latin spelling is shared by two or more different native words in a
   reference vocabulary (e.g. "kale" could be ಕಲೆ or ಕಳೆ). Ambiguous words are
   where a reader, or a model, may misread the rewrite.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache

from tokentax import corpus, romanize
from tokentax.languages import Language, get

Embedder = Callable[[list[str]], list[list[float]]]


@lru_cache(maxsize=32)
def _casual_index(code: str) -> dict[str, frozenset[str]]:
    """casual spelling -> set of native words that produce it (FLORES dev vocabulary)."""
    lang = get(code)
    index: dict[str, set[str]] = defaultdict(set)
    if lang.script.sanscript is None:
        return {}
    for line in corpus.flores(code, "dev"):
        for word in line.split():
            word = word.strip(".,!?;:\"'()[]।॥")
            if word and any(lang.script.contains(ch) for ch in word):
                index[romanize.casual(word, lang.script, lang.hindi_like_spelling)].add(word)
    return {k: frozenset(v) for k, v in index.items()}


def spelling_ambiguity(native_text: str, lang: Language) -> float:
    """Share of native-script words whose casual spelling is ambiguous."""
    index = _casual_index(lang.code)
    words = [w.strip(".,!?;:\"'()[]।॥") for w in native_text.split()]
    words = [w for w in words if w and any(lang.script.contains(ch) for ch in w)]
    if not words or not index:
        return 0.0
    hits = 0
    for w in words:
        spelled = romanize.casual(w, lang.script, lang.hindi_like_spelling)
        others = index.get(spelled, frozenset()) - {w}
        hits += bool(others)
    return hits / len(words)


_EMBEDDER: Embedder | None = None


def embedder() -> Embedder | None:
    """LaBSE via sentence-transformers if installed, else None."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer("sentence-transformers/LaBSE")

    def embed(texts: list[str]) -> list[list[float]]:
        vecs = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

    _EMBEDDER = embed
    return _EMBEDDER


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def similarity(original: str, rewrite: str, embed: Embedder | None = None) -> float | None:
    """Embedding similarity, or None when no embedder is available."""
    fn = embed or embedder()
    if fn is None:
        return None
    a, b = fn([original, rewrite])
    return cosine(a, b)
