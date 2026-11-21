"""Adaptive, meaning-safe format optimizer (Modules D + E).

For one input we build several rewrites, count their tokens on every tokenizer,
verify each against the original (``verify.verify``) and pick, per tokenizer,
the cheapest rewrite that passed. Every step is recorded in a decision trace,
so a reviewer can see what was tried, what it cost and why anything was
rejected.

Candidates (all rewrites of the text the user typed, never of the model):

* ``original``          as typed.
* ``normalized``        Unicode NFC, single spaces, danda written as ".".
* ``iso_roman``         lossless ISO 15919 romanisation (converts back exactly).
* ``casual_roman``      native words in casual Latin spelling; English kept.
* ``word_best``         each word in native or casual Latin, whichever is cheaper
                        for this tokenizer (so it differs per tokenizer).
* ``compressed``        grammar-rule compression (``compression``: spacing and
                        pronoun drop only where the verb already marks the subject).
* ``compressed+word_best``  both.

Important: this changes the *text sent*, not any model's tokenizer. Token
counts are exact for the open tokenizer files we load; for hosted APIs the
billed count can differ slightly (special tokens, chat formatting).
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache

from tokentax import compression, meaning, registry, romanize
from tokentax.explainer import guess_language
from tokentax.languages import Language, get
from tokentax.normalize import normalize
from tokentax.verify import Verification, verify

VERSION = "bhashatoken-3.0"
LOSSY = {"casual_roman", "word_best", "compressed+word_best"}


@dataclass(frozen=True)
class Candidate:
    name: str
    text: str
    meaning_check: str
    meaning_score: float
    safe: bool
    tokens: dict[str, int] = field(default_factory=dict)
    per_model_text: dict[str, str] = field(default_factory=dict)
    transforms: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    verification: dict[str, object] = field(default_factory=dict)
    build_ms: float = 0.0


@dataclass(frozen=True)
class Choice:
    tokenizer: str
    best: str
    tokens: int
    original_tokens: int
    text: str = ""
    why: str = ""

    @property
    def saving(self) -> float:
        return 1 - self.tokens / max(self.original_tokens, 1)


@dataclass(frozen=True)
class Plan:
    language: str
    candidates: list[Candidate]
    choices: list[Choice]
    trace: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "candidates": [asdict(c) for c in self.candidates],
            "choices": [dict(asdict(c), saving=c.saving) for c in self.choices],
            "trace": self.trace,
        }


@lru_cache(maxsize=16)
def tokenizer_hash(key: str) -> str:
    """First 12 hex digits of the SHA-256 of the tokenizer file (its exact version)."""
    return hashlib.sha256(registry.BY_KEY[key].path.read_bytes()).hexdigest()[:12]


def _native_words(text: str, lang: Language) -> list[bool]:
    return [any(lang.script.contains(ch) for ch in w) for w in text.split(" ")]


_RUNS = re.compile(r"[^\x00-\x7f]+|[\x00-\x7f]+")


def _native_runs(word: str, lang: Language, convert: Callable[[str], str]) -> str:
    """Convert only the native-script parts of a word; Latin parts ("QVC" in
    "QVCಯಲ್ಲಿ") stay exactly as typed."""
    return "".join(convert(p) if any(lang.script.contains(c) for c in p) else p
                   for p in _RUNS.findall(word))


def _casual_word(word: str, lang: Language) -> str:
    return _native_runs(word, lang,
                        lambda p: romanize.casual(p, lang.script, lang.hindi_like_spelling))


def casual_roman(text: str, lang: Language) -> str:
    return " ".join(
        _casual_word(w, lang) if f else w
        for w, f in zip(text.split(" "), _native_words(text, lang), strict=True)
    )


def iso_roman(text: str, lang: Language) -> str:
    return " ".join(
        _native_runs(w, lang, lambda p: romanize.lossless(p, lang.script)) if f else w
        for w, f in zip(text.split(" "), _native_words(text, lang), strict=True)
    )


def word_best(text: str, lang: Language, key: str) -> str:
    """Per word, keep native script or switch to casual Latin, whichever is cheaper."""
    out: list[str] = []
    for word, native in zip(text.split(" "), _native_words(text, lang), strict=True):
        if not native:
            out.append(word)
            continue
        latin = _casual_word(word, lang)
        cheaper = registry.count(key, " " + latin) < registry.count(key, " " + word)
        out.append(latin if cheaper else word)
    return " ".join(out)


def plan(text: str, keys: list[str] | None = None, language: str | None = None,
         max_ambiguity: float = 0.25, use_embeddings: bool = False,
         candidates: set[str] | None = None, verify_meaning: bool = True) -> Plan:
    """Build, verify and choose. ``candidates`` limits which rewrites are tried
    (used by the ablation study); ``verify_meaning=False`` turns the safety layer
    off (only for the ablation that shows why it is needed)."""
    t0 = time.perf_counter()
    keys = keys or [s.key for s in registry.available()]
    lang = get(language or guess_language(text))
    embed = meaning.embedder() if use_embeddings else None
    is_indic = lang.script.sanscript is not None and any(_native_words(text, lang))
    want = candidates or {"original", "normalized", "iso_roman", "casual_roman", "word_best",
                          "compressed", "compressed+word_best"}
    ambiguity = meaning.spelling_ambiguity(text, lang) if is_indic else 0.0

    built: list[Candidate] = []

    def add(name: str, rewrite: str, transforms: list[str], per_model: dict[str, str] | None = None,
            removals: list[str] | None = None, exact: bool = False, started: float = 0.0) -> None:
        texts = per_model or dict.fromkeys(keys, rewrite)
        tokens = {k: registry.count(k, texts[k]) for k in keys}
        reasons: list[str] = []
        ver: Verification | None = None
        if name == "original":
            check, score, safe = "identity", 1.0, True
        elif exact:
            check, score, safe = "exact round-trip", 1.0, True
        else:
            # verify the version for the first tokenizer and every distinct per-model text
            results = [verify(text, t, lang.code, removals or [], embed)
                       for t in dict.fromkeys(texts.values())]
            ver = min(results, key=lambda v: (v.passed, v.similarity))
            check, score = ver.similarity_method, ver.similarity
            safe = all(v.passed for v in results)
            for v in results:
                reasons.extend(r for r in v.reasons if r not in reasons)
            if name in LOSSY and ambiguity > max_ambiguity:
                safe = False
                reasons.append(f"spelling_clarity: {ambiguity:.0%} of words have a casual "
                               f"spelling shared by another word (limit {max_ambiguity:.0%})")
            if not verify_meaning:
                safe, reasons = True, [f"(verification off) {r}" for r in reasons]
        built.append(Candidate(
            name, rewrite, check, score, safe, tokens, per_model or {}, transforms, reasons,
            ver.to_dict() if ver else {}, round((time.perf_counter() - started) * 1000, 2)))

    add("original", text, [], started=time.perf_counter())
    if "normalized" in want:
        s = time.perf_counter()
        norm = normalize(text)
        if norm != text:
            add("normalized", norm, ["normalize"], started=s)
    if is_indic:
        if "iso_roman" in want:
            s = time.perf_counter()
            iso = iso_roman(text, lang)
            exact = all(romanize.roundtrip_ok(w, lang.script)
                        for w, n in zip(text.split(" "), _native_words(text, lang), strict=True)
                        if n)
            add("iso_roman", iso, ["iso15919"], exact=exact, started=s)
        if "casual_roman" in want:
            s = time.perf_counter()
            add("casual_roman", casual_roman(text, lang), ["casual_latin"], started=s)
        if "word_best" in want:
            s = time.perf_counter()
            wb = {k: word_best(text, lang, k) for k in keys}
            add("word_best", wb[keys[0]], ["per_word_hybrid"], wb, started=s)
    if {"compressed", "compressed+word_best"} & want:
        s = time.perf_counter()
        comp = compression.compress(text, keys, lang.code, max_saving=1.0)
        removed = [e.removed for e in comp.edits if e.removed]
        rules = sorted({e.rule for e in comp.edits})
        if comp.text != text and "compressed" in want:
            add("compressed", comp.text, rules, removals=removed, started=s)
        if comp.text != text and is_indic and "compressed+word_best" in want:
            s = time.perf_counter()
            cwb = {k: word_best(comp.text, lang, k) for k in keys}
            add("compressed+word_best", cwb[keys[0]], [*rules, "per_word_hybrid"], cwb,
                removals=removed, started=s)

    choices: list[Choice] = []
    original = built[0]
    for k in keys:
        safe = [c for c in built if c.safe]
        best = min(safe, key=lambda c: (c.tokens[k], c.name != "original"))
        rejected_cheaper = [c.name for c in built if not c.safe and c.tokens[k] < best.tokens[k]]
        why = (f"cheapest verified rewrite ({best.tokens[k]} vs {original.tokens[k]} tokens)"
               if best.name != "original" else "no verified rewrite was cheaper")
        if rejected_cheaper:
            why += f"; rejected cheaper but unsafe: {', '.join(rejected_cheaper)}"
        choices.append(Choice(k, best.name, best.tokens[k], original.tokens[k],
                              best.per_model_text.get(k, best.text), why))

    trace: dict[str, object] = {
        "request_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "system_version": VERSION,
        "language_detected": lang.code,
        "language_given": language,
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "tokenizers": {k: tokenizer_hash(k) for k in keys},
        "spelling_ambiguity": round(ambiguity, 4),
        "verification": "on" if verify_meaning else "OFF (ablation)",
        "similarity_backend": "LaBSE" if embed else "chrF on canonical spelling",
        "candidates_tried": [c.name for c in built],
        "rejected": {c.name: c.reasons for c in built if not c.safe},
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return Plan(lang.code, built, choices, trace)
