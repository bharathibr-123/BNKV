"""Meaning-safety verification for a rewritten text (Module E).

A rewrite is accepted only if every *critical* check passes and the similarity
score clears a threshold that was chosen on validation data
(see ``experiments.run_verification``). Similarity alone is never enough: two
sentences that differ only by "not" can be 95% similar.

Layers
  1. similarity   LaBSE cosine if available, else chrF on the canonical form
                  (every word in one Latin spelling, so scripts are comparable).
  2. critical     numbers, dates/times, negation, question form, modality
                  (must / may / should), protected strings (URLs, @handles,
                  capitalised names), and every content word, unless its removal
                  was declared and justified by a grammar rule.
  3. structure    sentence count; length ratio.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from tokentax.languages import ENGLISH, Language, get
from tokentax.normalize import ascii_digits, canonical, canonical_words, normalize
from tokentax.paths import RESULTS

Embedder = Callable[[list[str]], list[list[float]]]

# Marker words in native script; converted to canonical spelling at load time.
MARKERS: dict[str, dict[str, list[str]]] = {
    "kan_Knda": {
        "neg": ["ಇಲ್ಲ", "ಅಲ್ಲ", "ಬೇಡ", "ಬೇಡಿ", "ಕೂಡದು", "ಬಾರದು"],
        "neg_suffix": ["ಇಲ್ಲ", "ಕೂಡದು", "ಬಾರದು", "ಬೇಡ", "ಬೇಡಿ"],
        "question": ["ಏನು", "ಯಾಕೆ", "ಏಕೆ", "ಎಲ್ಲಿ", "ಯಾವ", "ಯಾರು", "ಹೇಗೆ", "ಎಷ್ಟು", "ಯಾವಾಗ",
                     "ಯಾವುದು"],
        "modal": ["ಬೇಕು", "ಬಹುದು", "ಬೇಕಾಗಿದೆ"],
        "modal_suffix": ["ಬೇಕು", "ಬಹುದು"],
    },
    "hin_Deva": {
        "neg": ["नहीं", "न", "मत", "ना"],
        "question": ["क्या", "क्यों", "कहाँ", "कहां", "कौन", "कैसे", "कब", "कितना", "कितनी", "कितने"],
        "modal": ["चाहिए", "सकता", "सकती", "सकते", "पड़ेगा", "पड़ेगी"],
    },
    "tam_Taml": {
        "neg": ["இல்லை", "வேண்டாம்", "அல்ல"],
        "neg_suffix": ["இல்லை", "ாதே", "ாதீர்கள்"],
        "question": ["என்ன", "ஏன்", "எங்கே", "யார்", "எப்படி", "எப்போது", "எவ்வளவு", "எந்த"],
        "modal": ["வேண்டும்", "முடியும்"],
    },
    "tel_Telu": {
        "neg": ["లేదు", "కాదు", "వద్దు"],
        "neg_suffix": ["లేదు", "వద్దు", "కూడదు"],
        "question": ["ఏమి", "ఎందుకు", "ఎక్కడ", "ఎవరు", "ఎలా", "ఎప్పుడు", "ఎంత"],
        "modal": ["కావాలి", "వచ్చు"],
    },
    "mal_Mlym": {
        "neg": ["ഇല്ല", "അല്ല", "വേണ്ട"],
        "neg_suffix": ["ഇല്ല"],
        "question": ["എന്ത്", "എന്തുകൊണ്ട്", "എവിടെ", "ആര്", "എങ്ങനെ", "എപ്പോൾ", "എത്ര"],
        "modal": ["വേണം", "കഴിയും"],
    },
    "ben_Beng": {
        "neg": ["না", "নয়", "নেই"],
        "question": ["কি", "কী", "কেন", "কোথায়", "কে", "কীভাবে", "কখন"],
    },
}
ENGLISH_MARKERS: dict[str, set[str]] = {
    "neg": {"no", "not", "never", "none", "nothing", "nobody", "neither", "nor", "dont",
            "doesnt", "didnt", "cant", "cannot", "wont", "isnt", "arent", "wasnt", "werent",
            "havent", "hasnt", "shouldnt", "wouldnt", "couldnt", "mustnt"},
    "question": {"what", "why", "where", "who", "whom", "how", "when", "which"},
    "modal": {"must", "should", "may", "might", "can", "could", "shall"},
}

_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
# Short month names count only when capitalised ("Jan"), because romanised
# Indian words are lower-case and "jan", "mar", "may" are common words there.
_MONTH = (r"(?:(?i:january|february|march|april|june|july|august|september|october|november|"
          r"december)|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)")
_DATE = re.compile(
    r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?!\d)|(?<!\d)\d{1,2}:\d{2}(?!\d)"
    rf"|(?<!\d)\d{{1,2}}\s+{_MONTH}\b|\b{_MONTH}\s+\d{{1,2}}(?!\d)")
# native-script handles are covered by the content-word check
_PROTECTED = re.compile(r"https?://\S+|\b\S+@\S+\.\w+\b|[@#][A-Za-z0-9_]+")
_SENT_END = re.compile(r"[.!?।॥]+")


@dataclass(frozen=True)
class Check:
    name: str
    layer: str  # similarity | critical | structure | warning
    passed: bool
    detail: str


@dataclass(frozen=True)
class Verification:
    passed: bool
    similarity: float
    similarity_method: str
    threshold: float
    threshold_source: str
    checks: list[Check] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks
                if not c.passed and c.layer != "warning"]

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        d["reasons"] = self.reasons
        return d


@lru_cache(maxsize=32)
def _markers(code: str) -> dict[str, frozenset[str]]:
    lang = get(code)
    out: dict[str, set[str]] = {k: set(v) for k, v in ENGLISH_MARKERS.items()}
    out.update({"neg_suffix": set(), "modal_suffix": set()})
    for kind, words in MARKERS.get(code, {}).items():
        for w in words:
            out.setdefault(kind, set()).update(canonical_words(w, lang))
    return {k: frozenset(v) for k, v in out.items()}


def _count(words: Iterable[str], exact: frozenset[str], suffixes: frozenset[str]) -> int:
    return sum(w in exact or any(w.endswith(s) and w != s for s in suffixes) for w in words)


def chrf(a: str, b: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF), whitespace removed, in [0, 1]."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a and not b:
        return 1.0
    scores: list[float] = []
    for n in range(1, max_n + 1):
        ga = Counter(a[i:i + n] for i in range(len(a) - n + 1))
        gb = Counter(b[i:i + n] for i in range(len(b) - n + 1))
        if not ga or not gb:
            continue
        overlap = sum((ga & gb).values())
        p, r = overlap / sum(gb.values()), overlap / sum(ga.values())
        scores.append(0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r))
    return sum(scores) / len(scores) if scores else 0.0


@lru_cache(maxsize=1)
def thresholds() -> dict[str, object]:
    """Similarity thresholds chosen on validation data, or a labelled default."""
    path = RESULTS / "thresholds.json"
    if path.exists():
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return {"chrf": 0.85, "embedding": 0.90, "source": "default (not calibrated yet)"}


def _multiset_diff(a: list[str], b: list[str]) -> list[str]:
    return list((Counter(a) - Counter(b)).elements())


def verify(original: str, candidate: str, language: str | None = None,
           declared_removals: Iterable[str] = (), embed: Embedder | None = None,
           threshold: float | None = None) -> Verification:
    """Check whether ``candidate`` keeps the meaning of ``original``.

    ``declared_removals`` are words a grammar rule removed on purpose (for example a
    pronoun the verb already expresses); they are excused from the content check.
    """
    lang: Language = get(language) if language else ENGLISH
    marks = _markers(lang.code)
    o_words = canonical_words(original, lang)
    c_words = canonical_words(candidate, lang)
    removed = [w for r in declared_removals for w in canonical_words(r, lang)]
    o_kept = list(o_words)
    for w in removed:
        if w in o_kept:
            o_kept.remove(w)
    checks: list[Check] = []

    def add(name: str, layer: str, ok: bool, detail: str) -> None:
        checks.append(Check(name, layer, ok, detail))

    o_raw, c_raw = ascii_digits(normalize(original)), ascii_digits(normalize(candidate))
    o_num, c_num = sorted(_NUMBER.findall(o_raw)), sorted(_NUMBER.findall(c_raw))
    add("numbers", "critical", o_num == c_num,
        "all numbers kept" if o_num == c_num else f"{o_num} became {c_num}")
    o_date = sorted(m.lower() for m in _DATE.findall(o_raw))
    c_date = sorted(m.lower() for m in _DATE.findall(c_raw))
    add("dates_times", "critical", o_date == c_date,
        "dates and times kept" if o_date == c_date else f"{o_date} became {c_date}")
    for kind, label in (("neg", "negation"), ("modal", "modality")):
        n_o = _count(o_words, marks[kind], marks[f"{kind}_suffix"])
        n_c = _count(c_words, marks[kind], marks[f"{kind}_suffix"])
        add(label, "critical", n_o == n_c,
            f"{n_o} marker(s) kept" if n_o == n_c else f"{n_o} marker(s) became {n_c}")
    q_o = ("?" in original, _count(o_words, marks["question"], frozenset()))
    q_c = ("?" in candidate, _count(c_words, marks["question"], frozenset()))
    add("question_form", "critical", q_o == q_c,
        "question form kept" if q_o == q_c else "question became a statement or vice versa")
    protected = _PROTECTED.findall(original) + [
        w for i, w in enumerate(original.split()) if i > 0 and w[:1].isupper() and w.isascii()
    ]
    lost = [p for p in protected if p.lower().strip(".,!?") not in candidate.lower()]
    add("names_and_links", "critical", not lost,
        "names, links and handles kept" if not lost else f"missing {lost}")
    missing = _multiset_diff(o_kept, c_words)
    add("content_words", "critical", not missing,
        "every word accounted for" if not missing
        else f"{len(missing)} word(s) dropped without a rule: {missing[:5]}")

    th = thresholds()
    if embed is not None:
        a, b = embed([original, candidate])
        num = sum(x * y for x, y in zip(a, b, strict=True))
        den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        sim, method = (num / den if den else 0.0), "embedding (LaBSE)"
        limit = float(threshold if threshold is not None else th["embedding"])  # type: ignore[arg-type]
    else:
        sim = chrf(" ".join(o_kept), " ".join(c_words))
        method = "chrF on canonical spelling"
        limit = float(threshold if threshold is not None else th["chrf"])  # type: ignore[arg-type]
    add("similarity", "similarity", sim >= limit, f"{sim:.3f} vs threshold {limit:.3f}")

    o_s = len([s for s in _SENT_END.split(original) if s.strip()])
    c_s = len([s for s in _SENT_END.split(candidate) if s.strip()])
    add("sentence_count", "structure", o_s == c_s, f"{o_s} → {c_s} sentence(s)")
    ratio = len(c_words) / max(len(o_words), 1)
    add("length_ratio", "warning", ratio >= 0.6, f"{ratio:.0%} of the original words remain")

    passed = all(c.passed for c in checks if c.layer in ("critical", "similarity", "structure"))
    return Verification(passed, sim, method, limit, str(th.get("source", "")), checks)


def same_meaning_canonical(original: str, candidate: str, language: str) -> bool:
    """Quick test: identical canonical spelling (true for pure script changes)."""
    lang = get(language)
    return canonical(original, lang) == canonical(candidate, lang)
