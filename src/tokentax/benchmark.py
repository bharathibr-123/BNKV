"""Module 01: the token tax for every language, writing style and tokenizer.

Gap addressed: published audits measure native script only; romanised and
code-mixed text, the way most people type, is rarely measured, and never across
many languages and tokenizers at once.

Two kinds of comparison:

* FLORES-200 parallel sentences: native script and casual romanisation are
  compared against the same English sentences, so the ratio is a cost of the
  same meaning.
* Real code-mixed comments: each comment is compared with itself rewritten
  fully in Latin script, so the comparison holds the content fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tokentax import corpus, metrics, registry, romanize
from tokentax.languages import ENGLISH, SOUTH_ASIAN, Language
from tokentax.scripts import Style


@dataclass(frozen=True)
class Cell:
    tokenizer: str
    language: str
    style: str
    tokens: int
    english_tokens: int
    premium: float
    fertility: float
    fragment_rate: float


def flores_matrix(keys: list[str] | None = None) -> list[Cell]:
    keys = keys or [s.key for s in registry.available()]
    english = corpus.flores(ENGLISH.code)
    cells: list[Cell] = []
    for key in keys:
        eng = metrics.stats(key, english).tokens
        cells.append(Cell(key, ENGLISH.code, "native", eng, eng, 1.0, 0.0, 0.0))
        for lang in SOUTH_ASIAN:
            native = corpus.flores(lang.code)
            variants: list[tuple[str, list[str]]] = [("native", native)]
            if lang.script.sanscript is not None:
                variants.append(("roman", corpus.roman(lang, native)))
            for style, texts in variants:
                st = metrics.stats(key, texts)
                cells.append(
                    Cell(key, lang.code, style, st.tokens, eng,
                         metrics.premium(st.tokens, eng), st.fertility, st.fragment_rate)
                )
    return cells


def cheaper_style(cells: list[Cell]) -> dict[str, dict[str, str]]:
    """For each tokenizer and language: which style costs fewer tokens."""
    table: dict[tuple[str, str], dict[str, int]] = {}
    for c in cells:
        table.setdefault((c.tokenizer, c.language), {})[c.style] = c.tokens
    out: dict[str, dict[str, str]] = {}
    for (key, lang), styles in table.items():
        if len(styles) == 2:
            out.setdefault(key, {})[lang] = min(styles, key=lambda s: styles[s])
    return out


@dataclass(frozen=True)
class CodemixRow:
    tokenizer: str
    language: str
    style: str
    comments: int
    tokens: int
    tokens_all_latin: int

    @property
    def latin_saving(self) -> float:
        """Fraction of tokens saved by rewriting the comments fully in Latin script."""
        return 1 - self.tokens_all_latin / max(self.tokens, 1)


def to_latin(text: str, lang: Language) -> str:
    """Rewrite only the native-script words in Latin, keeping English words as typed."""
    out = []
    for word in text.split(" "):
        if any(lang.script.contains(ch) for ch in word):
            out.append(romanize.casual(word, lang.script, lang.hindi_like_spelling))
        else:
            out.append(word)
    return " ".join(out)


def codemix_rows(keys: list[str] | None = None, limit: int = 2000) -> list[CodemixRow]:
    keys = keys or [s.key for s in registry.available()]
    rows: list[CodemixRow] = []
    for code in corpus.DCM_FILES.values():
        lang = next(lang for lang in SOUTH_ASIAN if lang.code == code)
        for style in (Style.NATIVE, Style.MIXED, Style.ROMAN):
            texts = corpus.codemix(code, style)[:limit]
            if not texts:
                continue
            latin = [to_latin(t, lang) for t in texts]
            for key in keys:
                rows.append(
                    CodemixRow(key, code, style.value, len(texts),
                               sum(registry.count_batch(key, texts)),
                               sum(registry.count_batch(key, latin)))
                )
    return rows


def as_dicts(items: list[Cell] | list[CodemixRow]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for it in items:
        d: dict[str, object] = dict(asdict(it))
        if isinstance(it, CodemixRow):
            d["latin_saving"] = it.latin_saving
        out.append(d)
    return out
