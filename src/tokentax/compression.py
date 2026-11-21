"""Module 07: meaning-safe compression with grammar rules.

Gap addressed: a 2026 cross-lingual audit of prompt compressors found that
compressors trained on English data damage non-English text, that safe
compression budgets outside English are about 1.3x (versus about 2x in
English), and that rule-based methods did not show this gap. Hindi was the only
Indian language tested. We build a rule-based compressor for South Asian
languages that removes only what the grammar makes redundant, records why each
edit was made, and stops before an unsafe budget.

Rules, in two tiers:

safe
  * ``spacing``       repeated spaces and punctuation ("!!!" -> "!").
  * ``pro_drop``      drop a subject pronoun when the verb ending already says
                      who it is (ನಾನು ... ಹೋಗುತ್ತೇನೆ -> ... ಹೋಗುತ್ತೇನೆ). Only for
                      languages whose verbs agree with the subject (Kannada,
                      Tamil, Telugu), never for Malayalam or Hindi.
aggressive (off by default)
  * ``shorter_word``  replace a long formal word with a common shorter one.
  * ``filler``        drop English discourse fillers in code-mixed chat.
  * ``repeat``        collapse a word repeated for emphasis.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from tokentax import meaning, registry
from tokentax.explainer import guess_language
from tokentax.languages import get

# pronoun -> verb endings that agree with it
PRO_DROP: dict[str, dict[str, tuple[str, ...]]] = {
    "kan_Knda": {
        "ನಾನು": ("ೇನೆ", "ೆನು"),
        "ನಾವು": ("ೇವೆ", "ೆವು"),
        "ನೀನು": ("ೀಯ", "ೀಯಾ"),
        "ನೀವು": ("ೀರಿ", "ೀರ", "ೀರಾ"),
    },
    "tam_Taml": {
        "நான்": ("ேன்",),
        "நாங்கள்": ("ோம்",),
        "நீ": ("ாய்",),
        "நீங்கள்": ("ீர்கள்",),
    },
    "tel_Telu": {
        "నేను": ("ాను", "ాను."),
        "మేము": ("ాము",),
        "నువ్వు": ("ావు",),
    },
}

SHORTER: dict[str, dict[str, str]] = {
    "kan_Knda": {
        "ಪೂರ್ಣಗೊಳಿಸು": "ಮುಗಿಸು",
        "ಪೂರ್ಣಗೊಳಿಸಿ": "ಮುಗಿಸಿ",
        "ಕಾರ್ಯನಿರ್ವಹಿಸು": "ಕೆಲಸಮಾಡು",
    },
    "hin_Deva": {
        "के द्वारा": "द्वारा",
        "के लिए कि": "ताकि",
    },
}

FILLERS = ("actually", "basically", "literally", "umm", "hmm", "like,")

_SENT = re.compile(r"(?<=[.!?।॥])\s+")


@dataclass(frozen=True)
class Edit:
    rule: str
    removed: str
    added: str
    reason: str


@dataclass(frozen=True)
class Compressed:
    language: str
    original: str
    text: str
    edits: list[Edit]
    tokens_before: dict[str, int]
    tokens_after: dict[str, int]
    similarity: float | None
    within_budget: bool

    def saving(self, key: str) -> float:
        return 1 - self.tokens_after[key] / max(self.tokens_before[key], 1)

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        d["saving"] = {k: self.saving(k) for k in self.tokens_before}
        return d


def _spacing(text: str, edits: list[Edit]) -> str:
    out = re.sub(r"([!?.,।])\1+", r"\1", text)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if out != text:
        edits.append(Edit("spacing", "", "", "repeated spaces or punctuation"))
    return out


def _pro_drop(text: str, language: str, edits: list[Edit]) -> str:
    rules = PRO_DROP.get(language)
    if not rules or not get(language).pro_drop:
        return text
    sentences = _SENT.split(text)
    out_sents: list[str] = []
    for sent in sentences:
        words = sent.split(" ")
        bare = [w.strip(".,!?;:।\"'") for w in words]
        for pronoun, endings in rules.items():
            if bare.count(pronoun) != 1:
                continue
            agreeing = [w for w in bare if w != pronoun and w.endswith(endings)]
            if len(agreeing) != 1:
                continue  # zero or several candidate verbs: not safe to infer the subject
            i = bare.index(pronoun)
            if words[i] != pronoun:
                continue  # attached punctuation or particle: keep it
            del words[i]
            del bare[i]
            edits.append(Edit("pro_drop", pronoun, "",
                              f"verb '{agreeing[0]}' already marks the subject"))
        out_sents.append(" ".join(words))
    return " ".join(out_sents)


def _shorter(text: str, language: str, edits: list[Edit]) -> str:
    for long, short in SHORTER.get(language, {}).items():
        if long in text:
            text = text.replace(long, short)
            edits.append(Edit("shorter_word", long, short, "common shorter word, same meaning"))
    return text


def _fillers(text: str, edits: list[Edit]) -> str:
    words = text.split(" ")
    kept = [w for w in words if w.lower() not in FILLERS]
    for w in words:
        if w.lower() in FILLERS:
            edits.append(Edit("filler", w, "", "discourse filler"))
    return " ".join(kept)


def _repeats(text: str, edits: list[Edit]) -> str:
    words = text.split(" ")
    out: list[str] = []
    for w in words:
        if out and w == out[-1] and any(ch.isalpha() for ch in w):
            edits.append(Edit("repeat", w, "", "word repeated for emphasis"))
            continue
        out.append(w)
    return " ".join(out)


def compress(text: str, keys: list[str] | None = None, language: str | None = None,
             aggressive: bool = False, max_saving: float = 0.25,
             min_similarity: float = 0.90, use_embeddings: bool = False) -> Compressed:
    """Compress ``text``. If the token saving exceeds ``max_saving`` (the safe
    budget) and no embedding check confirms the meaning, the original is kept."""
    keys = keys or [s.key for s in registry.available()]
    lang = language or guess_language(text)
    edits: list[Edit] = []
    out = _spacing(text, edits)
    out = _pro_drop(out, lang, edits)
    if aggressive:
        out = _shorter(out, lang, edits)
        out = _fillers(out, edits)
        out = _repeats(out, edits)
    before = {k: registry.count(k, text) for k in keys}
    after = {k: registry.count(k, out) for k in keys}
    sim = meaning.similarity(text, out) if use_embeddings and out != text else None
    biggest = max((1 - after[k] / max(before[k], 1) for k in keys), default=0.0)
    within = biggest <= max_saving or (sim is not None and sim >= min_similarity)
    if not within:
        return Compressed(lang, text, text, [], before, before, sim, False)
    return Compressed(lang, text, out, edits, before, after, sim, True)
