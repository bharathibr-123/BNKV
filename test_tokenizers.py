"""Tests that use the real tokenizer files in vendor/tokenizers."""

from __future__ import annotations

import pytest

from tokentax import compression, explainer, fairness, metrics, optimizer, registry
from tokentax.languages import get

pytestmark = pytest.mark.skipif(not registry.available(), reason="tokenizer files missing")

KANNADA = "ನಾನು ನಾಳೆ ಕಾಲೇಜಿಗೆ ಹೋಗಿ ನನ್ನ ಪ್ರಾಜೆಕ್ಟ್ ಪೂರ್ಣಗೊಳಿಸುತ್ತೇನೆ."


def test_vowel_sign_split_is_detected() -> None:
    assert registry.has_matra_wall("llama3")
    assert registry.has_matra_wall("gpt4")
    assert not registry.has_matra_wall("gpt4o")
    assert not registry.has_matra_wall("gemma3")


def test_pretokens_split_at_vowel_signs_and_fix_joins_them() -> None:
    word = "ಬೆಂಗಳೂರಿಗೆ"
    assert len(registry.pretokens("llama3", word)) == 6
    assert registry.pretokens("llama3", word, fixed=True) == [word]
    assert registry.pretokens("gpt4o", word) == [word]


def test_pretokens_keep_every_character() -> None:
    for word in ("abc123def", "ಬೆಂಗಳೂರಿಗೆ", "hello, world"):
        assert "".join(registry.pretokens("qwen3", word)) == word


def test_byte_pieces_are_counted() -> None:
    st = metrics.stats("llama3", ["ನಾನು"])
    assert st.tokens == st.byte_fragments == 8
    assert metrics.stats("gpt4o", ["ನಾನು"]).byte_fragments == 0


def test_explainer_parts_add_up() -> None:
    ex = explainer.explain("llama3", KANNADA, "kan_Knda")
    for w in ex.words:
        assert w.base + w.vowel_split + w.vocab_gap == w.tokens
        assert len(w.byte_flags) == len(w.pieces) == w.tokens
    assert ex.totals["vowel_split"] > 0


def test_guess_language() -> None:
    assert explainer.guess_language(KANNADA) == "kan_Knda"
    assert explainer.guess_language("मैं कल") == "hin_Deva"
    assert explainer.guess_language("hello there") == "eng_Latn"


def test_optimizer_never_costs_more_than_original() -> None:
    plan = optimizer.plan(KANNADA, keys=["llama3", "gpt4o", "gemma3"])
    original = next(c for c in plan.candidates if c.name == "original")
    assert original.safe
    for ch in plan.choices:
        assert ch.tokens <= original.tokens[ch.tokenizer]
    llama = next(c for c in plan.choices if c.tokenizer == "llama3")
    assert llama.best in ("casual_roman", "word_best", "compressed+word_best")
    assert llama.saving > 0.5


def test_optimizer_leaves_english_alone() -> None:
    plan = optimizer.plan("Please come on time tomorrow.", keys=["gpt4o"])
    assert [c.name for c in plan.candidates] == ["original"]


def test_iso_candidate_is_exact() -> None:
    plan = optimizer.plan(KANNADA, keys=["gpt4o"])
    iso = next(c for c in plan.candidates if c.name == "iso_roman")
    assert iso.safe and iso.meaning_check == "exact round-trip"


def test_pro_drop_only_when_verb_agrees() -> None:
    c = compression.compress("ನಾನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ.", ["gpt4o"], "kan_Knda")
    assert c.text == "ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ."
    assert [e.rule for e in c.edits] == ["pro_drop"]
    # third-person verb: the pronoun is not redundant
    kept = compression.compress("ನಾನು ಹೇಳಿದ ಮಾತು ಅವನು ಕೇಳುತ್ತಾನೆ.", ["gpt4o"], "kan_Knda")
    assert "ನಾನು" in kept.text


def test_no_pro_drop_in_malayalam() -> None:
    c = compression.compress("ഞാൻ നാളെ കോളേജിൽ പോകും.", ["gpt4o"], "mal_Mlym")
    assert c.text == "ഞാൻ നാളെ കോളേജിൽ പോകും."


def test_compression_budget_guard_keeps_original() -> None:
    c = compression.compress("ನಾನು ಹೋಗುತ್ತೇನೆ.", ["gpt4o"], "kan_Knda", max_saving=0.01)
    assert not c.within_budget
    assert c.text == c.original and not c.edits


def test_aggressive_rules_are_opt_in() -> None:
    text = "Sir actually tumba tumba chennagi explain madidri!!!"
    safe = compression.compress(text, ["gpt4o"], "kan_Knda")
    assert "actually" in safe.text and "tumba tumba" in safe.text
    agg = compression.compress(text, ["gpt4o"], "kan_Knda", aggressive=True, max_saving=0.9)
    assert "actually" not in agg.text and "tumba tumba" not in agg.text


def test_fairness_index_is_one_for_perfect_parts() -> None:
    perfect = fairness.FairnessScore("x", "y", 1.0, 1.0, 1.0, 1.0, 1.0)
    assert perfect.index == pytest.approx(1.0)
    weak = fairness.FairnessScore("x", "y", 0.5, None, 0.0, 1.0, 1.0)
    assert 0.0 < weak.index < 0.5


def test_is_word() -> None:
    assert fairness.is_word("ಮನೆಗಳಲ್ಲಿ")
    assert fairness.is_word("house")
    assert not fairness.is_word("2024")
    assert not fairness.is_word("house,")
    assert get("kan").script.contains("ಕ")
