"""Same sentence, every language: a per-sentence comparison table.

FLORES-200 devtest contains the same 1,012 sentences translated by professional
translators into every language we cover, so the token difference between two
rows is caused by the language and the tokenizer, not by the content. We do not
machine-translate the user's own text: a poor translation would make the
comparison unfair. Instead the user picks or searches a FLORES sentence.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from tokentax import corpus, metrics, optimizer, pricing, registry
from tokentax.languages import ENGLISH, SOUTH_ASIAN, Language

FEATURED = [
    (123, "Numbers and a 'not'"),
    (181, "Earthquake at a time"),
    (59, "Short news sentence"),
    (100, "Health"),
    (233, "Names and culture"),
    (112, "Disease report"),
]
_PUNCT = re.compile(r"^[^\w]+|[^\w\u0900-\u0dff\u1c50-\u1c7f]+$")


def _words(text: str) -> set[str]:
    """Lower-cased words; vowel signs stay inside Indic words."""
    out = {_PUNCT.sub("", w).lower() for w in unicodedata.normalize("NFC", text).split()}
    return {w for w in out if len(w) > 1}


def languages() -> list[Language]:
    return [ENGLISH, *SOUTH_ASIAN]


@lru_cache(maxsize=32)
def _texts(code: str) -> tuple[str, ...]:
    return tuple(corpus.flores(code))


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).strip().lower()


@lru_cache(maxsize=1)
def _exact_index() -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for lang in languages():
        for i, t in enumerate(_texts(lang.code)):
            out.setdefault(_norm(t), (i, lang.code))
    return out


def find(text: str) -> tuple[int, str] | None:
    """Index and language of a FLORES sentence equal to ``text`` (any language)."""
    return _exact_index().get(_norm(text))


def search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Sentences sharing the most words with ``query``, searched in every language."""
    q = _words(query)
    if not q:
        return []
    best: dict[int, tuple[float, str]] = {}
    for lang in languages():
        for i, t in enumerate(_texts(lang.code)):
            words = _words(t)
            # a query word also matches an inflected form ("ಭೂಕಂಪ" in "ಭೂಕಂಪವೊಂದು")
            hit = sum(w in words or any(x.startswith(w) for x in words) for w in q)
            if hit:
                score = hit / len(q) + hit / (len(words) + 5)
                if score > best.get(i, (0.0, ""))[0]:
                    best[i] = (score, lang.code)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:limit]
    return [{"index": i, "english": _texts("eng_Latn")[i], "matched_in": code}
            for i, (_, code) in top]


def featured() -> list[dict[str, Any]]:
    eng = _texts("eng_Latn")
    return [{"index": i, "title": title, "english": eng[i]} for i, title in FEATURED]


def compare(index: int, key: str) -> dict[str, Any]:
    """Every language's version of FLORES sentence ``index`` on tokenizer ``key``."""
    book = pricing.load()
    price = book.models[key]
    eng_tokens = registry.count(key, _texts("eng_Latn")[index])
    rows: list[dict[str, Any]] = []
    for lang in languages():
        text = _texts(lang.code)[index]
        ids = registry.load(key).encode(text, add_special_tokens=False).ids
        tokens = len(ids)
        row: dict[str, Any] = {
            "code": lang.code, "language": lang.name, "script": lang.script.name,
            "text": text, "characters": len(text), "words": len(text.split()),
            "tokens": tokens, "chars_per_token": len(text) / max(tokens, 1),
            "byte_pieces": sum(metrics.is_fragment(key, t) for t in ids),
            "x_english": tokens / max(eng_tokens, 1),
            "cost_inr_per_lakh": book.cost_inr(key, tokens * 100_000),
            "best": "original", "best_text": text, "best_tokens": tokens, "saving": 0.0,
            "rejected": 0,
        }
        if lang.code != ENGLISH.code:
            plan = optimizer.plan(text, keys=[key], language=lang.code)
            ch = plan.choices[0]
            row.update(best=ch.best, best_text=ch.text, best_tokens=ch.tokens,
                       saving=ch.saving,
                       rejected=sum(not c.safe for c in plan.candidates))
        row["best_cost_inr_per_lakh"] = book.cost_inr(key, row["best_tokens"] * 100_000)
        rows.append(row)
    return {
        "index": index, "tokenizer": key, "label": registry.BY_KEY[key].label,
        "price_input_usd_per_million": price.input, "usd_to_inr": book.usd_to_inr,
        "english_tokens": eng_tokens, "rows": rows,
        "source": "FLORES-200 devtest (professional translations, CC BY-SA 4.0)",
    }
