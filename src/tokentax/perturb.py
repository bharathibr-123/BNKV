"""Controlled perturbations for testing the meaning verifier.

Each function returns a rewrite of a native-script sentence plus a label:
*safe* rewrites keep the meaning (script changes, spacing, a pronoun drop the
grammar allows) and *unsafe* ones change it on purpose (add a negation, change
a number, delete or replace a word, turn a statement into a question). We know
the true label because we made the change, so we can measure how often the
verifier accepts safe rewrites and catches unsafe ones, per check.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from tokentax import compression, romanize
from tokentax.languages import Language
from tokentax.normalize import normalize
from tokentax.optimizer import casual_roman, iso_roman
from tokentax.verify import MARKERS

NEGATION_WORD = {code: m["neg"][0] for code, m in MARKERS.items()}
_END = re.compile(r"([.!?।॥]*)$")


@dataclass(frozen=True)
class Variant:
    kind: str
    safe: bool
    text: str
    removals: tuple[str, ...] = ()


def _words(text: str) -> list[str]:
    return text.split(" ")


def _content_index(words: list[str], lang: Language, rng: random.Random) -> int | None:
    idx = [i for i, w in enumerate(words) if sum(lang.script.contains(c) for c in w) >= 3]
    return rng.choice(idx) if idx else None


def negation_insert(text: str, lang: Language, rng: random.Random) -> str | None:
    neg = NEGATION_WORD.get(lang.code)
    if neg is None:
        return None
    m = _END.search(text)
    end = m.group(1) if m else ""
    return text[: len(text) - len(end)].rstrip() + " " + neg + end


def number_change(text: str, lang: Language, rng: random.Random) -> str | None:
    digits = [i for i, c in enumerate(text) if c.isdecimal()]
    if not digits:
        return None
    i = rng.choice(digits)
    new = str((int(text[i]) + rng.randint(1, 8)) % 10)
    return text[:i] + new + text[i + 1:]


def word_delete(text: str, lang: Language, rng: random.Random) -> str | None:
    words = _words(text)
    i = _content_index(words, lang, rng)
    if i is None or len(words) < 4:
        return None
    return " ".join(words[:i] + words[i + 1:])


def word_replace(text: str, lang: Language, rng: random.Random, vocab: list[str]) -> str | None:
    words = _words(text)
    i = _content_index(words, lang, rng)
    if i is None or not vocab:
        return None
    other = rng.choice([v for v in vocab if v != words[i]] or vocab)
    return " ".join([*words[:i], other, *words[i + 1:]])


def question_flip(text: str, lang: Language, rng: random.Random) -> str | None:
    if "?" in text:
        return None
    return _END.sub("", text).rstrip() + "?"


def variants(text: str, lang: Language, rng: random.Random, vocab: list[str]) -> list[Variant]:
    out: list[Variant] = [
        Variant("safe:iso_roman", True, iso_roman(text, lang)),
        Variant("safe:casual_roman", True, casual_roman(text, lang)),
        Variant("safe:spacing", True, text.replace(" ", "  ", 2)),
    ]
    norm = normalize(text)
    if norm != text:
        out.append(Variant("safe:normalized", True, norm))
    comp = compression.compress(text, ["gpt4o"], lang.code, max_saving=1.0)
    if comp.text != text:
        out.append(Variant("safe:pro_drop", True, comp.text,
                           tuple(e.removed for e in comp.edits if e.removed)))
    unsafe: dict[str, Callable[[], str | None]] = {
        "unsafe:negation_added": lambda: negation_insert(text, lang, rng),
        "unsafe:number_changed": lambda: number_change(text, lang, rng),
        "unsafe:word_deleted": lambda: word_delete(text, lang, rng),
        "unsafe:word_replaced": lambda: word_replace(text, lang, rng, vocab),
        "unsafe:question_added": lambda: question_flip(text, lang, rng),
    }
    for kind, make in unsafe.items():
        changed = make()
        if changed is not None and changed != text:
            out.append(Variant(kind, False, changed))
    # an unsafe change hidden inside a safe-looking script change
    neg = negation_insert(text, lang, rng)
    if neg is not None:
        out.append(Variant("unsafe:negation_added+roman", False,
                           romanize.casual(neg, lang.script, lang.hindi_like_spelling)))
    return out
