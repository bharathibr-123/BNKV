"""Module 04: predict token count, cost and context use before a request is sent.

Gap addressed: token estimators either ship the real tokenizer (not possible for
every API) or use rules like "4 characters per token" or regressions on generic
character classes (Latin, Han, digits). None model the features that drive the
token tax in Indic scripts: vowel signs, viramas and the script itself.

Our model is a ridge regression per tokenizer on script-aware counts. It is
trained on FLORES dev plus half of the real code-mixed comments and tested on
FLORES devtest plus the other half. ``OnlineCalibrator`` keeps improving it from
the token counts an API reports back (recursive least squares).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tokentax.languages import INDIC_SCRIPTS
from tokentax.paths import DATA
from tokentax.scripts import script_of

Vec = NDArray[np.float64]

_VIRAMAS = {"्", "্", "੍", "્", "୍", "்", "్", "್",
            "്", "්"}

FEATURES: tuple[str, ...] = (
    "bias", "chars", "utf8_bytes", "words", "latin", "latin_english", "digit", "punct",
    *(f"{s.name}:{part}" for s in INDIC_SCRIPTS for part in ("letter", "mark", "virama")),
)
_INDEX = {name: i for i, name in enumerate(FEATURES)}
# The generic baseline (like character-class estimators) sees scripts but not marks.
GENERIC: tuple[str, ...] = (
    "bias", "chars", "utf8_bytes", "words", "latin", "digit", "punct",
    *(f"{s.name}:letter" for s in INDIC_SCRIPTS),
)


_LATIN_WORD = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def english_words() -> frozenset[str]:
    """Common English words (built from FLORES English *dev*, never the test set)."""
    path = DATA / "english_words.txt"
    if not path.exists():
        return frozenset()
    return frozenset(path.read_text(encoding="utf-8").split())


def features(text: str) -> Vec:
    v = np.zeros(len(FEATURES))
    v[_INDEX["bias"]] = 1.0
    v[_INDEX["chars"]] = len(text)
    v[_INDEX["utf8_bytes"]] = len(text.encode("utf-8"))
    v[_INDEX["words"]] = len(text.split())
    vocab = english_words()
    # Latin letters that belong to English words cost far less than romanised Indic.
    v[_INDEX["latin_english"]] = sum(
        len(m.group(0)) for m in _LATIN_WORD.finditer(text) if m.group(0).lower() in vocab
    )
    for ch in text:
        if ch.isdigit():
            v[_INDEX["digit"]] += 1
            continue
        script = script_of(ch)
        if script is None:
            if not ch.isspace():
                v[_INDEX["punct"]] += 1
        elif script.name == "Latin":
            v[_INDEX["latin"]] += 1
        elif ch in _VIRAMAS:
            v[_INDEX[f"{script.name}:virama"]] += 1
        elif unicodedata.category(ch).startswith("M"):
            v[_INDEX[f"{script.name}:mark"]] += 1
        else:
            v[_INDEX[f"{script.name}:letter"]] += 1
    return v


def _generic(x: Vec) -> Vec:
    """Collapse marks and viramas into letters, as a character-class estimator would."""
    out = np.zeros(len(GENERIC))
    for i, name in enumerate(GENERIC):
        out[i] = x[_INDEX[name]]
        if name.endswith(":letter"):
            base = name.split(":")[0]
            out[i] += x[_INDEX[f"{base}:mark"]] + x[_INDEX[f"{base}:virama"]]
    return out


GROUPS: tuple[str, ...] = ("english", "romanised", "native")


def group_of(x: Vec) -> int:
    """Which sub-model handles a text: mostly-English Latin, other Latin, or Indic script."""
    latin = x[_INDEX["latin"]]
    letters = latin + sum(x[_INDEX[f"{s.name}:{p}"]] for s in INDIC_SCRIPTS
                          for p in ("letter", "mark", "virama"))
    if letters and latin / letters > 0.95:
        return 0 if x[_INDEX["latin_english"]] / max(latin, 1.0) > 0.6 else 1
    return 2


def _ridge(x: NDArray[np.float64], y: Vec, lam: float) -> Vec:
    reg = lam * np.eye(x.shape[1])
    reg[0, 0] = 0.0  # do not shrink the bias
    return np.asarray(np.linalg.solve(x.T @ x + reg, x.T @ y), dtype=np.float64)


@dataclass
class TokenPredictor:
    """Token-count model for one tokenizer.

    The script-aware model keeps one linear model per text group (English,
    romanised Indic, Indic script), because a Latin letter costs about twice as
    much in romanised Kannada as in English. The generic baseline is a single
    linear model on character classes.
    """

    tokenizer: str
    weights: list[Vec]
    generic: bool = False

    @classmethod
    def fit(cls, tokenizer: str, texts: list[str], tokens: list[int],
            lam: float = 1.0, generic: bool = False) -> TokenPredictor:
        x = np.stack([features(t) for t in texts])
        y = np.asarray(tokens, dtype=np.float64)
        if generic:
            gx = np.stack([_generic(r) for r in x])
            return cls(tokenizer, [_ridge(gx, y, lam)], True)
        groups = np.asarray([group_of(r) for r in x])
        overall = _ridge(x, y, lam)
        weights = [
            _ridge(x[groups == g], y[groups == g], lam) if (groups == g).sum() >= 50 else overall
            for g in range(len(GROUPS))
        ]
        return cls(tokenizer, weights, False)

    def predict(self, text: str) -> float:
        x = features(text)
        if self.generic:
            return max(1.0, float(_generic(x) @ self.weights[0]))
        return max(1.0, float(x @ self.weights[group_of(x)]))

    def to_json(self) -> dict[str, object]:
        names = GENERIC if self.generic else FEATURES
        return {"tokenizer": self.tokenizer, "generic": self.generic,
                "weights": [dict(zip(names, w.tolist(), strict=True)) for w in self.weights]}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> TokenPredictor:
        generic = bool(data.get("generic", False))
        names = GENERIC if generic else FEATURES
        ws = data["weights"]
        assert isinstance(ws, list)
        return cls(str(data["tokenizer"]),
                   [np.asarray([float(w[n]) for n in names]) for w in ws], generic)


def chars_over_four(text: str) -> float:
    """The common rule of thumb: one token per four characters."""
    return max(1.0, len(text) / 4)


@dataclass
class OnlineCalibrator:
    """Recursive least squares on top of a trained predictor.

    Each time an API reports the real token count for a text, ``update`` nudges
    the weights toward it, with a forgetting factor so recent traffic counts more.
    """

    weights: Vec
    forget: float = 0.999
    p: NDArray[np.float64] = field(default_factory=lambda: np.eye(len(FEATURES)) * 10.0)

    @classmethod
    def start(cls, base: TokenPredictor, group: int, forget: float = 0.999) -> OnlineCalibrator:
        """Calibrate one group's sub-model (e.g. 2 = Indic script)."""
        if base.generic:
            raise ValueError("calibrate the script-aware predictor")
        return cls(base.weights[group].copy(), forget)

    def predict(self, text: str) -> float:
        return max(1.0, float(features(text) @ self.weights))

    def update(self, text: str, actual_tokens: int) -> float:
        """Learn from one observation and return the error before the update."""
        x = features(text)
        err = actual_tokens - float(x @ self.weights)
        px = self.p @ x
        gain = px / (self.forget + float(x @ px))
        self.weights = self.weights + gain * err
        self.p = (self.p - np.outer(gain, px)) / self.forget
        return err


def save(predictors: dict[str, TokenPredictor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: p.to_json() for k, p in predictors.items()}, indent=1),
                    encoding="utf-8")


def load(path: Path) -> dict[str, TokenPredictor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: TokenPredictor.from_json(v) for k, v in raw.items()}
