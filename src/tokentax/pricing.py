"""Prices, currency and context-window settings from config/prices.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tokentax.paths import CONFIG


@dataclass(frozen=True)
class Price:
    input: float  # USD per 1M tokens
    output: float
    list_price: bool


@dataclass(frozen=True)
class PriceBook:
    usd_to_inr: float
    context_window: int
    models: dict[str, Price]

    def cost_inr(self, key: str, input_tokens: float, output_tokens: float = 0.0) -> float:
        p = self.models[key]
        usd = (input_tokens * p.input + output_tokens * p.output) / 1_000_000
        return usd * self.usd_to_inr


@lru_cache(maxsize=4)
def load(path: Path = CONFIG / "prices.json") -> PriceBook:
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = {
        k: Price(float(v["input"]), float(v["output"]), bool(v.get("list_price", False)))
        for k, v in raw["models"].items()
    }
    return PriceBook(float(raw["usd_to_inr"]), int(raw["context_window"]), models)
