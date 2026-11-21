"""Predictor, calibrator and simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokentax import predictor, simulator
from tokentax.pricing import Price, PriceBook


def _fake_count(text: str) -> int:
    """A made-up tokenizer: 1 token per Latin letter pair, 3 per Kannada letter or mark."""
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    kannada = sum(0x0C80 <= ord(ch) <= 0x0CFF for ch in text)
    return max(1, latin // 2 + 3 * kannada)


TEXTS = [
    "ನಾನು ನಾಳೆ ಬರುತ್ತೇನೆ", "ಮನೆಗಳಲ್ಲಿ ಮಳೆ", "naanu naale baruttene", "come tomorrow please",
    "ನಾಳೆ meeting ಇದೆ", "ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ", "the quick brown fox", "nivu tumba chennagi",
] * 10


def test_features_count_marks_and_viramas() -> None:
    x = predictor.features("ಕನ್ನಡ")  # ಕ ನ ್ ನ ಡ
    idx = {n: i for i, n in enumerate(predictor.FEATURES)}
    assert x[idx["Kannada:letter"]] == 4
    assert x[idx["Kannada:virama"]] == 1
    assert x[idx["Kannada:mark"]] == 0
    assert x[idx["chars"]] == 5


def test_predictor_learns_a_linear_tokenizer() -> None:
    y = [_fake_count(t) for t in TEXTS]
    model = predictor.TokenPredictor.fit("fake", TEXTS, y, lam=1e-6)
    errors = [abs(model.predict(t) - _fake_count(t)) / _fake_count(t) for t in TEXTS]
    assert max(errors) < 0.15


def test_predictor_json_round_trip(tmp_path: Path) -> None:
    y = [_fake_count(t) for t in TEXTS]
    model = predictor.TokenPredictor.fit("fake", TEXTS, y)
    predictor.save({"fake": model}, tmp_path / "p.json")
    loaded = predictor.load(tmp_path / "p.json")["fake"]
    assert loaded.predict("ನಾಳೆ ಬನ್ನಿ") == pytest.approx(model.predict("ನಾಳೆ ಬನ್ನಿ"))


def test_calibrator_reduces_error() -> None:
    zero = predictor.TokenPredictor("fake", [np.zeros(len(predictor.FEATURES))] * 3)
    cal = predictor.OnlineCalibrator.start(zero, group=2)
    native = [t for t in TEXTS if any(0x0C80 <= ord(c) <= 0x0CFF for c in t)]
    first = abs(cal.update(native[0], _fake_count(native[0])))
    for t in native * 3:
        cal.update(t, _fake_count(t))
    assert abs(cal.predict(native[0]) - _fake_count(native[0])) < first


def test_chars_over_four() -> None:
    assert predictor.chars_over_four("abcdefgh") == 2.0


BOOK = PriceBook(usd_to_inr=100.0, context_window=1000,
                 models={"m": Price(input=1.0, output=4.0, list_price=False)})


def test_simulator_english_costs_no_tax() -> None:
    rates = simulator.Rates(en_tokens_per_word=1.0, premium=1.0, premium_optimized=1.0,
                            premium_compressed=1.0)
    res = simulator.run(simulator.Scenario(model="m"), rates, BOOK)
    assert res.extra_annual_inr == pytest.approx(0.0)
    assert res.annual_inr == pytest.approx(res.english_only_annual_inr)


def test_simulator_matches_hand_calculation() -> None:
    scen = simulator.Scenario(daily_users=1000, messages_per_user=1, regional_share=1.0,
                              input_words=10, output_words=10, model="m")
    rates = simulator.Rates(en_tokens_per_word=1.0, premium=2.0, premium_optimized=1.5,
                            premium_compressed=1.5)
    res = simulator.run(scen, rates, BOOK)
    # 1000 msgs/day * (20 in * $1 + 20 out * $4) / 1e6 * ₹100 * 365
    assert res.annual_inr == pytest.approx(1000 * (20 + 80) / 1e6 * 100 * 365)
    # the optimizer rewrites questions only: 20 -> 15 input tokens, replies unchanged
    assert res.saved_by_optimizer_annual_inr == pytest.approx(1000 * 5 * 1.0 / 1e6 * 100 * 365)
    both = simulator.run(simulator.Scenario(**{**scen.__dict__, "reply_optimized": True}),
                         rates, BOOK)
    assert both.saved_by_optimizer_annual_inr == pytest.approx(res.annual_inr * 0.25)
    assert res.saved_by_compression_annual_inr == pytest.approx(0.0)
