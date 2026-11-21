"""Back end for the Live Demos page.

* ``conversation``: the same conversation (consecutive FLORES sentences, which
  come from the same articles) typed by an English user and by a user of another
  language, with and without BhashaToken, so the context window can be watched
  filling up turn by turn.
* ``batch``: audit a list of real messages (one per line) on one tokenizer.
* ``mutate``: one-click edits for the "Can you fool the verifier?" demo; each
  edit is labelled safe or unsafe because we know what it changes.
"""

from __future__ import annotations

import random
import re
from typing import Any

from tokentax import compression, corpus, optimizer, pricing, registry
from tokentax.explainer import guess_language
from tokentax.languages import get
from tokentax.perturb import negation_insert, number_change, question_flip, word_delete

MAX_TURNS = 60
MAX_LINES = 200
MAX_LINE_CHARS = 1000


def conversation(language: str, key: str, start: int = 0, turns: int = 40) -> dict[str, Any]:
    turns = max(1, min(turns, MAX_TURNS))
    start = max(0, min(start, 1012 - turns))
    eng = corpus.flores("eng_Latn")[start:start + turns]
    native = corpus.flores(language)[start:start + turns]
    rows = []
    for i, (e, n) in enumerate(zip(eng, native, strict=True)):
        plan = optimizer.plan(n, keys=[key], language=language)
        ch = plan.choices[0]
        rows.append({
            "turn": i + 1, "english": e, "native": n, "optimized": ch.text,
            "tokens_english": registry.count(key, e), "tokens_native": ch.original_tokens,
            "tokens_optimized": ch.tokens, "chosen": ch.best,
        })
    return {"language": language, "tokenizer": key, "label": registry.BY_KEY[key].label,
            "start": start, "turns": rows,
            "source": "consecutive FLORES-200 devtest sentences (professional translations)"}


def batch(text: str, key: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:MAX_LINES]
    book = pricing.load()
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line[:MAX_LINE_CHARS]
        lang = guess_language(line)
        plan = optimizer.plan(line, keys=[key], language=lang)
        ch = plan.choices[0]
        rows.append({
            "text": line, "language": lang, "characters": len(line),
            "tokens": ch.original_tokens, "after": ch.tokens, "chosen": ch.best,
            "sent": ch.text, "saving": ch.saving,
            "blocked": [c.name for c in plan.candidates if not c.safe],
        })
    before = sum(r["tokens"] for r in rows)
    after = sum(r["after"] for r in rows)
    return {
        "tokenizer": key, "label": registry.BY_KEY[key].label, "rows": rows,
        "truncated": len([ln for ln in text.splitlines() if ln.strip()]) > MAX_LINES,
        "totals": {
            "messages": len(rows), "tokens": before, "after": after,
            "saving": 1 - after / before if before else 0.0,
            "inr_per_lakh_before": book.cost_inr(key, before * 100_000 / max(len(rows), 1)),
            "inr_per_lakh_after": book.cost_inr(key, after * 100_000 / max(len(rows), 1)),
            "blocked_rewrites": sum(len(r["blocked"]) for r in rows),
            "languages": sorted({r["language"] for r in rows}),
        },
    }


MUTATIONS: dict[str, tuple[str, bool]] = {
    "roman": ("Write it in English letters", True),
    "iso": ("Exact romanisation (ISO)", True),
    "spaces": ("Add extra spaces", True),
    "pronoun": ("Drop a pronoun the verb already shows", True),
    "negation": ("Add or remove a 'not'", False),
    "number": ("Change a number", False),
    "question": ("Turn it into a question", False),
    "delete": ("Delete a word", False),
}


# negative verb endings -> positive ending (Kannada): "ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ" (don't take)
# -> "ತೆಗೆದುಕೊಳ್ಳಿ" (take), "ಬರುವುದಿಲ್ಲ" (won't come) -> "ಬರುತ್ತೇನೆ" (will come)
_NEG_SUFFIX = {"kan_Knda": [("ಬೇಡಿ", "ಿ"), ("ಬೇಡ", "ಿ"), ("ವುದಿಲ್ಲ", "ತ್ತೇನೆ"), ("ಬಾರದು", "ಬೇಕು")],
               "tam_Taml": [("ாதீர்கள்", "ுங்கள்"), ("ாதே", "ு")]}


def _remove_negation(text: str, code: str) -> str | None:
    from tokentax.verify import MARKERS

    words = text.split(" ")
    for i, w in enumerate(words):
        core = re.sub(r"[.!?,।]+$", "", w)
        for neg, pos in _NEG_SUFFIX.get(code, []):
            if core.endswith(neg) and len(core) > len(neg) + 1:
                words[i] = core[: -len(neg)] + pos + w[len(core):]
                return " ".join(words)
    for neg in MARKERS.get(code, {}).get("neg", []):
        for i, w in enumerate(words):
            core = re.sub(r"[.!?,।]+$", "", w)
            if core == neg:
                tail = w[len(core):]
                rest = [*words[:i], *words[i + 1:]]
                if tail and rest:
                    rest[-1] = re.sub(r"[.!?,।]*$", "", rest[-1]) + tail
                return " ".join(rest)
    return None


def mutate(text: str, kind: str, language: str | None = None, seed: int = 0) -> dict[str, Any]:
    if kind not in MUTATIONS:
        raise KeyError(kind)
    lang = get(language or guess_language(text))
    rng = random.Random(seed)
    out: str | None
    if kind == "roman":
        out = optimizer.casual_roman(text, lang)
    elif kind == "iso":
        out = optimizer.iso_roman(text, lang)
    elif kind == "spaces":
        out = text.replace(" ", "  ", 2)
    elif kind == "pronoun":
        out = compression.compress(text, ["gpt4o"], lang.code, max_saving=1.0).text
    elif kind == "negation":
        out = _remove_negation(text, lang.code) or negation_insert(text, lang, rng)
    elif kind == "number":
        out = number_change(text, lang, rng)
    elif kind == "question":
        out = question_flip(text, lang, rng)
    else:
        out = word_delete(text, lang, rng)
    label, safe = MUTATIONS[kind]
    applied = out is not None and out != text
    return {"kind": kind, "label": label, "expected_safe": safe, "applied": applied,
            "text": out if applied else text, "language": lang.code}


CROWD_LANGS = ("kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "mal_Mlym", "ben_Beng")
CROWD_TOKENIZER = "llama3"


def crowd_pair(language: str, seed: int) -> dict[str, Any]:
    """A short FLORES sentence and the rewrite BhashaToken would send (Llama 3).

    Only rewrites that differ from the original are shown, so raters judge real
    decisions. Sentences are short so they can be read on a phone."""
    texts = corpus.flores(language)
    short = [i for i, t in enumerate(texts) if 25 <= len(t) <= 110]
    rng = random.Random(f"crowd:{language}:{seed}")
    for _ in range(25):
        i = rng.choice(short)
        plan = optimizer.plan(texts[i], keys=[CROWD_TOKENIZER], language=language)
        ch = plan.choices[0]
        if ch.best != "original" and ch.text != texts[i]:
            return {"pair_id": f"{language}:{i}:{ch.best}", "language": language,
                    "index": i, "original": texts[i], "rewrite": ch.text,
                    "candidate": ch.best, "english": corpus.flores("eng_Latn")[i],
                    "tokens_before": ch.original_tokens, "tokens_after": ch.tokens}
    raise LookupError(language)
