"""Module 05: what a government chatbot pays for the token tax.

The simulator turns measured token premiums into rupees. Inputs are the
settings a state IT department would choose; every rate it uses comes from the
benchmark (results/benchmark.json), so nothing is guessed.

Assumptions, stated so they can be challenged:
* Messages are the same content as English, just in another language, so the
  English token count times the language's premium gives that language's count.
* Output tokens follow the same premium as input (the model replies in the same
  language).
* Savings apply only to the share of chats in the regional language.
* The optimizer rewrites what is *sent*, so by default it lowers input tokens
  only. Replies stay in native script unless ``reply_optimized`` is set (that
  assumes the model is asked to answer in the cheaper format, which changes the
  user experience and should be tested before use).
* These are simulated costs from tokenizer counts, not an API bill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tokentax.pricing import PriceBook

# Average English tokens per word, measured on FLORES English devtest for each
# tokenizer; filled in from results at runtime, with this as a fallback.
DEFAULT_EN_TOKENS_PER_WORD = 1.3


@dataclass(frozen=True)
class Scenario:
    daily_users: int = 100_000
    messages_per_user: float = 4.0
    regional_share: float = 0.7
    input_words: float = 40.0
    output_words: float = 120.0
    model: str = "gpt4o"
    language: str = "kan_Knda"
    input_price_usd: float | None = None  # per 1M tokens; None = price book
    output_price_usd: float | None = None
    context_window: int | None = None
    reply_optimized: bool = False


@dataclass(frozen=True)
class Rates:
    """Measured multipliers for one model and language."""

    en_tokens_per_word: float
    premium: float  # tokens vs English, native script as typed
    premium_optimized: float  # after the format optimizer
    premium_compressed: float  # after optimizer + safe compression


@dataclass(frozen=True)
class Result:
    daily_inr: float
    monthly_inr: float
    annual_inr: float
    english_only_annual_inr: float
    extra_annual_inr: float
    saved_by_optimizer_annual_inr: float
    saved_by_compression_annual_inr: float
    context_share_per_message: float
    tokens_per_message_english: float
    tokens_per_message_native: float
    tokens_per_message_optimized: float
    messages_in_context_english: float
    messages_in_context_native: float
    messages_in_context_optimized: float
    effective_context_vs_english: float
    effective_context_vs_english_optimized: float
    price_input_usd: float
    price_output_usd: float

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self))


def run(s: Scenario, rates: Rates, prices: PriceBook) -> Result:
    msgs_per_day = s.daily_users * s.messages_per_user
    en_in = s.input_words * rates.en_tokens_per_word
    en_out = s.output_words * rates.en_tokens_per_word
    book = prices.models[s.model]
    p_in = book.input if s.input_price_usd is None else s.input_price_usd
    p_out = book.output if s.output_price_usd is None else s.output_price_usd
    window = s.context_window or prices.context_window

    def inr(tok_in: float, tok_out: float) -> float:
        return (tok_in * p_in + tok_out * p_out) / 1_000_000 * prices.usd_to_inr

    def daily(in_premium: float, out_premium: float) -> float:
        regional = s.regional_share * msgs_per_day * inr(en_in * in_premium, en_out * out_premium)
        english = (1 - s.regional_share) * msgs_per_day * inr(en_in, en_out)
        return regional + english

    out_opt = rates.premium_optimized if s.reply_optimized else rates.premium
    out_cmp = rates.premium_compressed if s.reply_optimized else rates.premium
    base = 365 * daily(rates.premium, rates.premium)
    english_only = 365 * daily(1.0, 1.0)
    optimized = 365 * daily(rates.premium_optimized, out_opt)
    compressed = 365 * daily(rates.premium_compressed, out_cmp)
    per_en = en_in + en_out
    per_native = (en_in + en_out) * rates.premium
    per_opt = en_in * rates.premium_compressed + en_out * out_cmp
    return Result(
        daily_inr=base / 365,
        monthly_inr=base / 12,
        annual_inr=base,
        english_only_annual_inr=english_only,
        extra_annual_inr=base - english_only,
        saved_by_optimizer_annual_inr=base - optimized,
        saved_by_compression_annual_inr=optimized - compressed,
        context_share_per_message=per_native / window,
        tokens_per_message_english=per_en,
        tokens_per_message_native=per_native,
        tokens_per_message_optimized=per_opt,
        messages_in_context_english=window / per_en,
        messages_in_context_native=window / per_native,
        messages_in_context_optimized=window / per_opt,
        effective_context_vs_english=per_en / per_native,
        effective_context_vs_english_optimized=per_en / per_opt,
        price_input_usd=p_in,
        price_output_usd=p_out,
    )
