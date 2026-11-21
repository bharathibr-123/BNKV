"""Tests for normalisation, the meaning verifier, the adaptive optimizer, the audit
ledger and the cost simulator (master-prompt modules C, E, F, G)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tokentax import audit, normalize, optimizer, perturb, pricing, romanize, simulator, verify
from tokentax.languages import get

KAN = "ನಾನು ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ. ಟಿಕೆಟ್ ಬೆಲೆ ಎಷ್ಟು?"
MED = "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ."
KEYS = ["gpt4o", "llama3"]


# --- normalisation ------------------------------------------------------------

def test_normalize_spaces_and_danda() -> None:
    assert normalize.normalize("  कल   आना।  ") == "कल आना."


def test_normalize_keeps_zero_width_joiner() -> None:
    text = "ಬುಷ್‌ ಬಂದರು"
    assert "‌" in normalize.normalize(text)


def test_indic_digits_become_ascii() -> None:
    assert normalize.ascii_digits("೨೦೨೬ और २५") == "2026 और 25"


@pytest.mark.parametrize("code", ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "mal_Mlym",
                                  "ben_Beng"])
def test_canonical_form_ignores_script(code: str) -> None:
    from tokentax import corpus

    lang = get(code)
    texts = corpus.flores(code)[:100]
    agree = 0
    for text in texts:
        base = normalize.canonical_words(text, lang)
        agree += (normalize.canonical_words(optimizer.casual_roman(text, lang), lang) == base
                  and normalize.canonical_words(optimizer.iso_roman(text, lang), lang) == base)
    # rare misses are mixed words such as "Erdoğan-এর"; they cause a (safe) rejection
    assert agree >= 0.97 * len(texts)


def test_bengali_nukta_letters_roundtrip() -> None:
    ben = get("ben_Beng").script
    assert romanize.roundtrip_ok("হয়েছিল", ben)
    assert "়" not in romanize.casual("হয়েছিল", ben, True)


# --- verifier -----------------------------------------------------------------

def test_script_change_is_safe() -> None:
    lang = get("kan_Knda")
    for rewrite in (optimizer.casual_roman(KAN, lang), optimizer.iso_roman(KAN, lang)):
        assert verify.verify(KAN, rewrite, "kan_Knda").passed


def test_negation_removed_is_rejected() -> None:
    v = verify.verify(MED, "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ.", "kan_Knda")
    assert not v.passed
    assert any(r.startswith("negation") for r in v.reasons)


def test_negation_added_is_rejected_even_when_similar() -> None:
    v = verify.verify("मैं कल आऊँगा।", "मैं कल नहीं आऊँगा।", "hin_Deva")
    assert not v.passed and any(r.startswith("negation") for r in v.reasons)


def test_number_change_is_rejected() -> None:
    v = verify.verify(KAN, KAN.replace("10", "11"), "kan_Knda")
    assert not v.passed and any(r.startswith("numbers") for r in v.reasons)


def test_date_and_time_are_checked() -> None:
    v = verify.verify("Meet on 12 March at 10:30", "Meet on 13 March at 10:30")
    assert any(r.startswith("dates_times") for r in v.reasons)
    # a time glued to a word after romanisation still counts as kept
    lang = get("kan_Knda")
    t = "ಸೋಮವಾರ ರಾತ್ರಿ10:08 ಕ್ಕೆ ಭೂಕಂಪ ಆಯಿತು."
    assert verify.verify(t, optimizer.casual_roman(t, lang), "kan_Knda").passed


def test_question_turned_into_statement_is_rejected() -> None:
    v = verify.verify("क्या कल स्कूल बंद रहेगा?", "कल स्कूल बंद रहेगा।", "hin_Deva")
    assert any(r.startswith("question_form") for r in v.reasons)


def test_name_and_link_must_survive() -> None:
    v = verify.verify("Please email Ravi at https://example.org today",
                      "Please email at today")
    assert any(r.startswith("names_and_links") for r in v.reasons)


def test_declared_pronoun_drop_is_allowed() -> None:
    dropped = KAN.replace("ನಾನು ", "", 1)
    assert not verify.verify(KAN, dropped, "kan_Knda").passed
    assert verify.verify(KAN, dropped, "kan_Knda", declared_removals=["ನಾನು"]).passed


def test_chrf_bounds() -> None:
    assert verify.chrf("abc def", "abc def") == pytest.approx(1.0)
    assert verify.chrf("abc", "xyz") == 0.0


def test_perturbation_labels_match_verifier_on_sample() -> None:
    lang = get("tam_Taml")
    rng = random.Random(1)
    text = "நான் நாளை 5 மணிக்கு சென்னைக்கு போவேன்."
    for var in perturb.variants(text, lang, rng, ["வீடு", "பள்ளி"]):
        assert verify.verify(text, var.text, lang.code, var.removals).passed == var.safe, var.kind


# --- adaptive optimizer -------------------------------------------------------

def test_plan_never_chooses_unsafe_candidate() -> None:
    p = optimizer.plan(KAN, keys=KEYS, language="kan_Knda")
    safe = {c.name for c in p.candidates if c.safe}
    assert all(ch.best in safe for ch in p.choices)
    assert all(ch.tokens <= ch.original_tokens for ch in p.choices)


def test_plan_picks_minimum_safe_tokens() -> None:
    p = optimizer.plan(KAN, keys=KEYS, language="kan_Knda")
    for ch in p.choices:
        best = min(c.tokens[ch.tokenizer] for c in p.candidates if c.safe)
        assert ch.tokens == best


def test_trace_records_versions_and_candidates() -> None:
    p = optimizer.plan(KAN, keys=KEYS, language="kan_Knda")
    t = p.trace
    assert set(t["tokenizers"]) == set(KEYS)  # type: ignore[arg-type]
    assert all(len(h) == 12 for h in t["tokenizers"].values())  # type: ignore[union-attr]
    assert "original" in t["candidates_tried"]  # type: ignore[operator]
    assert float(t["latency_ms"]) > 0  # type: ignore[arg-type]


def test_turning_verification_off_can_choose_unsafe() -> None:
    text = "ಯು.ಎಸ್ ಅಧ್ಯಕ್ಷ ಜಾರ್ಜ್‌ ಡಬ್ಲ್ಯೂ. ಬುಷ್‌ ಸ್ವಾಗತ ಕೋರಿದರು."
    on = optimizer.plan(text, keys=["llama3"], language="kan_Knda")
    off = optimizer.plan(text, keys=["llama3"], language="kan_Knda", verify_meaning=False)
    assert on.trace["rejected"]
    assert off.choices[0].tokens <= on.choices[0].tokens


def test_english_input_is_left_alone() -> None:
    p = optimizer.plan("Please come on time tomorrow.", keys=["gpt4o"])
    assert p.choices[0].best == "original"


def test_long_input_handled() -> None:
    text = " ".join([KAN] * 20)
    p = optimizer.plan(text, keys=["gpt4o"], language="kan_Knda")
    assert p.choices[0].tokens <= p.choices[0].original_tokens


def test_plan_is_deterministic_except_ids() -> None:
    a = optimizer.plan(KAN, keys=KEYS, language="kan_Knda").to_dict()
    b = optimizer.plan(KAN, keys=KEYS, language="kan_Knda").to_dict()
    assert a["choices"] == b["choices"]
    assert a["trace"]["input_sha256"] == b["trace"]["input_sha256"]  # type: ignore[index]


# --- audit ledger -------------------------------------------------------------

def test_audit_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "DB_PATH", tmp_path / "a.sqlite")
    plan = optimizer.plan(KAN, keys=["gpt4o"], language="kan_Knda").to_dict()
    audit.record(KAN, plan)
    rid = plan["trace"]["request_id"]  # type: ignore[index]
    item = audit.get(rid)
    assert item is not None and item["input"] == KAN
    audit.add_review(rid, "word_best", "tester", 5, 4, True, "fine")
    assert audit.review_summary()["total"] == 1
    assert "tester" in audit.export_csv()
    with pytest.raises(KeyError):
        audit.add_review("missing-id", "x", "t", 3, 3, False)


# --- simulator ----------------------------------------------------------------

def _rates() -> simulator.Rates:
    return simulator.Rates(en_tokens_per_word=1.3, premium=4.0, premium_optimized=2.0,
                           premium_compressed=1.9)


def test_simulator_costs_scale_and_price_override() -> None:
    book = pricing.load()
    base = simulator.run(simulator.Scenario(), _rates(), book)
    assert base.annual_inr == pytest.approx(base.daily_inr * 365)
    assert base.monthly_inr == pytest.approx(base.annual_inr / 12)
    cheap = simulator.run(simulator.Scenario(input_price_usd=0.0, output_price_usd=0.0),
                          _rates(), book)
    assert cheap.annual_inr == 0.0


def test_simulator_input_only_by_default() -> None:
    book = pricing.load()
    inp = simulator.run(simulator.Scenario(), _rates(), book)
    both = simulator.run(simulator.Scenario(reply_optimized=True), _rates(), book)
    assert 0 < inp.saved_by_optimizer_annual_inr < both.saved_by_optimizer_annual_inr


def test_simulator_context() -> None:
    r = simulator.run(simulator.Scenario(context_window=100_000), _rates(), pricing.load())
    assert r.effective_context_vs_english == pytest.approx(0.25)
    assert r.messages_in_context_optimized > r.messages_in_context_native
