"""Language-aware model router (idea borrowed from LLM gateways).

Gateways route requests by price or quality but count cost per token, so they
miss that the same message costs a different number of tokens on each model.
The router counts the real tokens of *this* message on every candidate model,
applies the format optimizer, and picks the cheapest (model, format) pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tokentax import optimizer, pricing, registry


@dataclass(frozen=True)
class Route:
    model: str
    format: str
    text: str
    input_tokens: int
    cost_inr: float


@dataclass(frozen=True)
class Decision:
    best: Route
    options: list[Route]

    def to_dict(self) -> dict[str, object]:
        return {"best": asdict(self.best), "options": [asdict(r) for r in self.options]}


def route(text: str, models: list[str] | None = None, expected_output_ratio: float = 3.0,
          language: str | None = None) -> Decision:
    """Pick the cheapest model and safe format for ``text``.

    ``expected_output_ratio`` is the reply length in tokens relative to the prompt.
    """
    book = pricing.load()
    have = {s.key for s in registry.available()}
    models = models or [m for m in book.models if m in have]
    plan = optimizer.plan(text, keys=models, language=language)
    by_name = {c.name: c for c in plan.candidates}
    options: list[Route] = []
    for choice in plan.choices:
        cand = by_name[choice.best]
        text_out = cand.per_model_text.get(choice.tokenizer, cand.text)
        cost = book.cost_inr(choice.tokenizer, choice.tokens,
                             choice.tokens * expected_output_ratio)
        options.append(Route(choice.tokenizer, choice.best, text_out, choice.tokens, cost))
    options.sort(key=lambda r: r.cost_inr)
    return Decision(options[0], options)
