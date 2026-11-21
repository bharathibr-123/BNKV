"""The tokenizers we compare, loaded from each model's official tokenizer.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex
from tokenizers import Tokenizer

from tokentax.paths import TOKENIZERS


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    vendor: str
    open_weights: bool

    @property
    def path(self) -> Path:
        return TOKENIZERS / f"{self.key}.json"


SPECS: tuple[TokenizerSpec, ...] = (
    TokenizerSpec("gpt4", "GPT-4 (cl100k)", "OpenAI", False),
    TokenizerSpec("gpt4o", "GPT-4o (o200k)", "OpenAI", False),
    TokenizerSpec("llama3", "Llama 3", "Meta", True),
    TokenizerSpec("gemma3", "Gemma 3", "Google", True),
    TokenizerSpec("qwen3", "Qwen3", "Alibaba", True),
    TokenizerSpec("deepseek_v4", "DeepSeek V4", "DeepSeek", True),
    TokenizerSpec("mistral_nemo", "Mistral NeMo", "Mistral", True),
    # optional: download with `python scripts/get_sarvam.py` (Sarvam non-commercial licence)
    TokenizerSpec("sarvam1", "Sarvam-1", "Sarvam AI", True),
)
BY_KEY: dict[str, TokenizerSpec] = {s.key: s for s in SPECS}


def available() -> tuple[TokenizerSpec, ...]:
    """Specs whose tokenizer file is present on disk."""
    return tuple(s for s in SPECS if s.path.exists())


@lru_cache(maxsize=16)
def load(key: str) -> Tokenizer:
    return Tokenizer.from_file(str(BY_KEY[key].path))


def count(key: str, text: str) -> int:
    return len(load(key).encode(text, add_special_tokens=False).ids)


def count_batch(key: str, texts: list[str]) -> list[int]:
    encs = load(key).encode_batch(texts, add_special_tokens=False)
    return [len(e.ids) for e in encs]


# --- pre-tokenizer inspection ------------------------------------------------

def split_patterns(key: str) -> list[str]:
    """Regex patterns used by the tokenizer's pre-tokenizer (empty if none)."""
    spec = json.loads(BY_KEY[key].path.read_text(encoding="utf-8"))
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "Split":
                pat = node.get("pattern", {})
                if isinstance(pat, dict) and "Regex" in pat:
                    found.append(str(pat["Regex"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec.get("pre_tokenizer"))
    return found


def has_matra_wall(key: str) -> bool:
    """True if some word class matches letters but not combining marks."""
    for pat in split_patterns(key):
        for cls in regex.findall(r"\[[^\]]*\]|\\p\{L\}\+", pat):
            if "\\p{L}" in cls and "\\p{M}" not in cls and "^" not in cls:
                return True
    return False


@lru_cache(maxsize=16)
def _pre(key: str) -> list[regex.Pattern[str]]:
    return [regex.compile(p) for p in split_patterns(key)]


@lru_cache(maxsize=16)
def _pre_fixed(key: str) -> list[regex.Pattern[str]]:
    return [regex.compile(_fix_classes(p)) for p in split_patterns(key)]


def _fix_classes(pattern: str) -> str:
    """Add \\p{M} next to every positive \\p{L} so marks join the word."""
    out = regex.sub(r"(?<!\^)\\p\{L\}", r"\\p{L}\\p{M}", pattern)
    # "\p{L}\p{M}+" outside a class must become a class.
    out = regex.sub(r"(?<![\[\w])\\p\{L\}\\p\{M\}([+*]{1,2})", r"[\\p{L}\\p{M}]\1", out)
    return str(out)


@lru_cache(maxsize=16)
def load_fixed(key: str) -> Tokenizer | None:
    """The same tokenizer with only its pre-tokenizer fixed so vowel signs stay
    inside words (vocabulary and merges unchanged). None if nothing to fix.

    This is a what-if: the model itself was trained with the original
    pre-tokenizer, so a real fix would also need retraining.
    """
    spec = json.loads(BY_KEY[key].path.read_text(encoding="utf-8"))
    changed = False

    def walk(node: object) -> None:
        nonlocal changed
        if isinstance(node, dict):
            pat = node.get("pattern") if node.get("type") == "Split" else None
            if isinstance(pat, dict) and "Regex" in pat:
                fixed = _fix_classes(str(pat["Regex"]))
                if fixed != pat["Regex"]:
                    pat["Regex"] = fixed
                    changed = True
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec.get("pre_tokenizer"))
    return Tokenizer.from_str(json.dumps(spec)) if changed else None


def count_fixed(key: str, text: str) -> int:
    """Tokens if the pre-tokenizer kept vowel signs inside words (what-if)."""
    tok = load_fixed(key)
    return count(key, text) if tok is None else len(tok.encode(text, add_special_tokens=False).ids)


def pretokens(key: str, word: str, fixed: bool = False) -> list[str]:
    """Pieces the pre-tokenizer cuts a word into, before BPE runs.

    BPE can never merge across these cuts, so their count is a floor on tokens.
    """
    pats = _pre_fixed(key) if fixed else _pre(key)
    if not pats:
        return [word]
    pieces = [word]
    for pat in pats:
        nxt: list[str] = []
        for piece in pieces:
            pos = 0
            for m in pat.finditer(piece):
                if m.start() > pos:
                    nxt.append(piece[pos : m.start()])
                if m.group(0):
                    nxt.append(m.group(0))
                pos = m.end()
            if pos < len(piece):
                nxt.append(piece[pos:])
        pieces = nxt
    return pieces
