"""Build every text split used in the study, with no test leakage.

Written to ``data/``:

  flores/<code>.txt      FLORES-200 devtest, 1,012 parallel sentences per language
  flores_dev/<code>.txt  FLORES-200 dev (997 sentences), used only for training
  roman/<code>.txt       devtest in casual romanisation (machine-made)
  codemix/<lang>_<style>.txt
                         real YouTube comments (DravidianCodeMix) for Kannada,
                         Tamil and Malayalam, split by writing style
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokentax import romanize
from tokentax.languages import ALL_LANGUAGES, ENGLISH, SOUTH_ASIAN, Language
from tokentax.paths import DATA, RAW
from tokentax.scripts import Style, profile

FLORES_DIR = RAW / "ctq" / "dataset"
DCM_DIR = RAW / "dcm" / "x" / "DravidianCodeMix"

# DravidianCodeMix file -> FLORES code. Lines are "label<TAB>text".
DCM_FILES: dict[str, str] = {
    "kannada_sentiment.csv": "kan_Knda",
    "tamil_sentiment_full.csv": "tam_Taml",
    "mal_full_sentiment.tsv": "mal_Mlym",
}
_DCM_LABELS = {
    "positive", "negative", "mixed_feelings", "unknown_state", "not-kannada",
    "not-tamil", "not-malayalam", "not_offensive", "offensive",
}


def _read(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(lines)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def flores(code: str, split: str = "devtest") -> list[str]:
    """Read a FLORES split from data/ (after build) or the raw download."""
    built = DATA / ("flores" if split == "devtest" else "flores_dev") / f"{code}.txt"
    if built.exists():
        return _read(built)
    sub = "test" if split == "devtest" else "train"
    return _read(FLORES_DIR / sub / f"{code}.{split}")


def roman(lang: Language, texts: Iterable[str]) -> list[str]:
    return [romanize.casual(t, lang.script, lang.hindi_like_spelling) for t in texts]


def _dcm_comments(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2:
            continue
        labels = [p for p in parts if p.lower() in _DCM_LABELS]
        texts = [p for p in parts if p and p.lower() not in _DCM_LABELS]
        if not labels or not texts or labels[0].lower().startswith("not-"):
            continue
        text = max(texts, key=len)
        if len(text.split()) >= 3:
            yield text


def codemix(code: str, style: Style) -> list[str]:
    path = DATA / "codemix" / f"{code}_{style.value}.txt"
    return _read(path) if path.exists() else []


def build_english_words(n: int = 5000) -> int:
    """Most frequent English words in FLORES *dev* (used by the predictor)."""
    from collections import Counter

    words: Counter[str] = Counter()
    for line in flores(ENGLISH.code, "dev"):
        words.update(w.lower() for w in re.findall(r"[A-Za-z]+", line))
    top = [w for w, _ in words.most_common(n)]
    return _write(DATA / "english_words.txt", top)


def build() -> dict[str, int]:
    counts: dict[str, int] = {}
    for lang in ALL_LANGUAGES:
        test = flores(lang.code, "devtest")
        dev = flores(lang.code, "dev")
        counts[f"flores/{lang.code}"] = _write(DATA / "flores" / f"{lang.code}.txt", test)
        _write(DATA / "flores_dev" / f"{lang.code}.txt", dev)
    for lang in SOUTH_ASIAN:
        if lang.script.sanscript is None:
            continue
        test = flores(lang.code, "devtest")
        out = DATA / "roman" / f"{lang.code}.txt"
        counts[f"roman/{lang.code}"] = _write(out, roman(lang, test))
        _write(DATA / "roman_dev" / f"{lang.code}.txt", roman(lang, flores(lang.code, "dev")))
    for fname, code in DCM_FILES.items():
        buckets: dict[Style, list[str]] = {Style.NATIVE: [], Style.ROMAN: [], Style.MIXED: []}
        seen: set[str] = set()
        for text in _dcm_comments(DCM_DIR / fname):
            style = profile(text).style
            if style in buckets and text not in seen:
                seen.add(text)
                buckets[style].append(text)
        for style, rows in buckets.items():
            counts[f"codemix/{code}_{style.value}"] = _write(
                DATA / "codemix" / f"{code}_{style.value}.txt", rows
            )
    counts["english_words"] = build_english_words()
    (DATA / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts
