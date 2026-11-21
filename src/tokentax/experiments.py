"""Run every experiment and write the results to results/*.json.

Each function returns plain data (dicts and lists) so results are easy to save,
test and serve to the dashboard.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tokentax import (
    benchmark,
    compression,
    corpus,
    explainer,
    fairness,
    meaning,
    metrics,
    optimizer,
    predictor,
    pricing,
    registry,
    report,
    simulator,
)
from tokentax.languages import ALL_LANGUAGES, ENGLISH, SOUTH_ASIAN, get
from tokentax.paths import RESULTS
from tokentax.scripts import Style

Json = dict[str, Any]


def _save(name: str, data: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def keys() -> list[str]:
    return [s.key for s in registry.available()]


# 01 -------------------------------------------------------------------------
def run_benchmark() -> Json:
    cells = benchmark.flores_matrix()
    english = corpus.flores(ENGLISH.code)
    en_words = sum(len(t.split()) for t in english)
    data: Json = {
        "flores": benchmark.as_dicts(cells),
        "cheaper_style": benchmark.cheaper_style(cells),
        "codemix": benchmark.as_dicts(benchmark.codemix_rows()),
        "english_tokens_per_word": {
            k: metrics.stats(k, english).tokens / en_words for k in keys()
        },
        "matra_wall": {k: registry.has_matra_wall(k) for k in keys()},
    }
    _save("benchmark", data)
    return data


# 02 -------------------------------------------------------------------------
def run_fairness() -> Json:
    scores = fairness.all_scores()
    data: Json = {"parts": list(fairness.PARTS), "scores": [s.to_dict() for s in scores]}
    _save("fairness", data)
    return data


# 03 -------------------------------------------------------------------------
def run_explainer(per_lang: int = 150) -> Json:
    """Where do the extra tokens come from, per tokenizer and language?"""
    rows: list[Json] = []
    langs = [lang for lang in SOUTH_ASIAN if lang.script.sanscript is not None]
    for key in keys():
        for lang in langs:
            totals: dict[str, int] = defaultdict(int)
            aligns: list[float] = []
            chances: list[float] = []
            for text in corpus.flores(lang.code)[:per_lang]:
                ex = explainer.explain(key, text, lang.code)
                for k, v in ex.totals.items():
                    totals[k] += v
                aligns.extend(w.grammar_alignment for w in ex.words
                              if w.grammar_alignment is not None)
                chances.extend(w.alignment_chance for w in ex.words
                               if w.alignment_chance is not None)
            extra = max(totals["tokens"] - totals["base"], 1)
            rows.append({
                "tokenizer": key, "language": lang.code, **totals,
                "vowel_split_share": totals["vowel_split"] / extra,
                "vocab_gap_share": totals["vocab_gap"] / extra,
                "byte_share": totals["byte_pieces"] / max(totals["tokens"], 1),
                "grammar_alignment": statistics.mean(aligns) if aligns else None,
                "alignment_chance": statistics.mean(chances) if chances else None,
            })
    data: Json = {"rows": rows, "sentences_per_language": per_lang}
    _save("explainer", data)
    return data


# 04 -------------------------------------------------------------------------
def _predictor_sets(split: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for lang in ALL_LANGUAGES:
        texts = corpus.flores(lang.code, split)
        out.append((lang.code, texts))
        if lang.script.sanscript is not None:
            out.append((f"{lang.code}/roman", corpus.roman(lang, texts)))
    for code in corpus.DCM_FILES.values():
        for style in (Style.NATIVE, Style.MIXED, Style.ROMAN):
            rows = corpus.codemix(code, style)
            if rows:
                half = rows[0::2] if split == "dev" else rows[1::2]
                out.append((f"{code}/real_{style.value}", half))
    return out


def run_predictor(train_per_set: int = 600, test_per_set: int = 400) -> Json:
    train = _predictor_sets("dev")
    test = _predictor_sets("devtest")
    models: dict[str, predictor.TokenPredictor] = {}
    report: list[Json] = []
    for key in keys():
        x = [t for _, texts in train for t in texts[:train_per_set]]
        y = registry.count_batch(key, x)
        ours = predictor.TokenPredictor.fit(key, x, y)
        generic = predictor.TokenPredictor.fit(key, x, y, generic=True)
        models[key] = ours
        for name, texts in test:
            texts = texts[:test_per_set]
            truth = np.asarray(registry.count_batch(key, texts), dtype=np.float64)

            def mape(preds: list[float], truth: np.ndarray = truth) -> float:
                return float(np.median(np.abs(np.asarray(preds) - truth) / truth) * 100)

            report.append({
                "tokenizer": key, "set": name,
                "ours": mape([ours.predict(t) for t in texts]),
                "generic": mape([generic.predict(t) for t in texts]),
                "chars_over_4": mape([predictor.chars_over_four(t) for t in texts]),
            })
    predictor.save(models, RESULTS / "predictors.json")
    summary: Json = {}
    for key in keys():
        rows = [r for r in report if r["tokenizer"] == key]
        summary[key] = {m: {"mean": statistics.mean(r[m] for r in rows),
                            "worst": max(r[m] for r in rows)}
                        for m in ("ours", "generic", "chars_over_4")}
    data: Json = {"metric": "median absolute % error per sentence",
                  "summary": summary, "rows": report}
    _save("predictor", data)
    return data


# 06 -------------------------------------------------------------------------
def run_optimizer(per_lang: int = 120) -> Json:
    """Per-sentence format choice vs fixed policies, on FLORES."""
    ks = keys()
    rows: list[Json] = []
    langs = [lang for lang in SOUTH_ASIAN if lang.script.sanscript is not None]
    for lang in langs:
        texts = corpus.flores(lang.code)[:per_lang]
        english = corpus.flores(ENGLISH.code)[:per_lang]
        en = {k: sum(registry.count_batch(k, english)) for k in ks}
        totals: dict[str, dict[str, int]] = {k: defaultdict(int) for k in ks}
        ambiguity: list[float] = []
        for text in texts:
            plan = optimizer.plan(text, keys=ks, language=lang.code)
            ambiguity.append(meaning.spelling_ambiguity(text, lang))
            by = {c.name: c for c in plan.candidates}
            for ch in plan.choices:
                t = totals[ch.tokenizer]
                t["native"] += by["original"].tokens[ch.tokenizer]
                t["casual"] += by["casual_roman"].tokens[ch.tokenizer]
                t["word_best"] += by["word_best"].tokens[ch.tokenizer]
                t["optimized"] += ch.tokens
        for k in ks:
            t = totals[k]
            rows.append({
                "tokenizer": k, "language": lang.code,
                "premium_native": t["native"] / en[k],
                "premium_casual": t["casual"] / en[k],
                "premium_word_best": t["word_best"] / en[k],
                "premium_optimized": t["optimized"] / en[k],
                "saving_vs_native": 1 - t["optimized"] / t["native"],
                "saving_vs_best_fixed": 1 - t["optimized"] / min(t["native"], t["casual"]),
            })
        rows.append({"tokenizer": "*", "language": lang.code,
                     "spelling_ambiguity": statistics.mean(ambiguity)})
    data: Json = {"sentences_per_language": per_lang, "rows": rows}
    _save("optimizer", data)
    return data


# 07 -------------------------------------------------------------------------
def run_compression(per_lang: int = 300) -> Json:
    ks = keys()
    rows: list[Json] = []
    examples: list[Json] = []
    for code in ("kan_Knda", "tam_Taml", "tel_Telu", "mal_Mlym", "hin_Deva"):
        texts = corpus.flores(code)[:per_lang]
        before: dict[str, int] = defaultdict(int)
        after: dict[str, int] = defaultdict(int)
        edited = 0
        rules: dict[str, int] = defaultdict(int)
        for text in texts:
            c = compression.compress(text, ks, code)
            edited += bool(c.edits)
            for e in c.edits:
                rules[e.rule] += 1
            for k in ks:
                before[k] += c.tokens_before[k]
                after[k] += c.tokens_after[k]
            if c.edits and len(examples) < 12:
                examples.append({"language": code, "before": c.original, "after": c.text,
                                 "edits": [e.reason for e in c.edits]})
        rows.append({"set": f"flores/{code}", "sentences": len(texts),
                     "edited_share": edited / len(texts), "rules": dict(rules),
                     "saving": {k: 1 - after[k] / max(before[k], 1) for k in ks}})
    for code in corpus.DCM_FILES.values():
        texts = corpus.codemix(code, Style.ROMAN)[:per_lang] + corpus.codemix(
            code, Style.MIXED)[:per_lang]
        before = defaultdict(int)
        after = defaultdict(int)
        edited = 0
        rules = defaultdict(int)
        for text in texts:
            c = compression.compress(text, ks, code, aggressive=True)
            edited += bool(c.edits)
            for e in c.edits:
                rules[e.rule] += 1
            for k in ks:
                before[k] += c.tokens_before[k]
                after[k] += c.tokens_after[k]
        rows.append({"set": f"real_codemix/{code}", "sentences": len(texts), "aggressive": True,
                     "edited_share": edited / max(len(texts), 1), "rules": dict(rules),
                     "saving": {k: 1 - after[k] / max(before[k], 1) for k in ks}})
    data: Json = {"rows": rows, "examples": examples}
    _save("compression", data)
    return data


# 05 -------------------------------------------------------------------------
def rates_for(model: str, language: str, bench: Json, opt: Json, comp: Json) -> simulator.Rates:
    flores_cells = {(c["tokenizer"], c["language"], c["style"]): c for c in bench["flores"]}
    premium = flores_cells[(model, language, "native")]["premium"]
    opt_row = next((r for r in opt["rows"]
                    if r.get("tokenizer") == model and r["language"] == language), None)
    optimized = opt_row["premium_optimized"] if opt_row else premium
    comp_row = next((r for r in comp["rows"] if r["set"] == f"flores/{language}"), None)
    comp_saving = comp_row["saving"][model] if comp_row else 0.0
    return simulator.Rates(
        en_tokens_per_word=bench["english_tokens_per_word"][model],
        premium=premium,
        premium_optimized=optimized,
        premium_compressed=optimized * (1 - comp_saving),
    )


def run_simulator(bench: Json, opt: Json, comp: Json) -> Json:
    book = pricing.load()
    rows: list[Json] = []
    for model in keys():
        for lang in ("kan_Knda", "tam_Taml", "tel_Telu", "hin_Deva", "ben_Beng", "mal_Mlym"):
            scen = simulator.Scenario(model=model, language=lang)
            rates = rates_for(model, lang, bench, opt, comp)
            res = simulator.run(scen, rates, book)
            rows.append({"model": model, "language": lang, "scenario": scen.__dict__,
                         "rates": rates.__dict__, **res.to_dict()})
    data: Json = {"usd_to_inr": book.usd_to_inr, "rows": rows}
    _save("simulator", data)
    return data


def run_all(build_corpus: bool = True) -> dict[str, float]:
    """Re-run every experiment. ``build_corpus=False`` reuses the prepared text in
    ``data/`` (shipped with the project), so the raw downloads are not needed."""
    timings: dict[str, float] = {}
    t = time.time()
    if build_corpus:
        corpus.build()
    timings["corpus"] = time.time() - t
    from tokentax import research  # calibrate the verifier before anything uses it

    t = time.time()
    research.run_verification()
    timings["verification"] = time.time() - t
    steps: list[tuple[str, Any]] = [
        ("benchmark", run_benchmark), ("fairness", run_fairness),
        ("explainer", run_explainer), ("predictor", run_predictor),
        ("optimizer", run_optimizer), ("compression", run_compression),
    ]
    out: dict[str, Json] = {}
    for name, fn in steps:
        t = time.time()
        out[name] = fn()
        timings[name] = time.time() - t
        print(f"{name}: {timings[name]:.0f}s", flush=True)
    run_simulator(out["benchmark"], out["optimizer"], out["compression"])
    for name, fn in (("strategies", research.run_strategies),
                     ("counterfactual", research.run_counterfactual),
                     ("lfi_sensitivity", research.run_lfi_sensitivity),
                     ("efficiency", research.run_efficiency), ("ledger", research.run_ledger)):
        t = time.time()
        fn()
        timings[name] = time.time() - t
        print(f"{name}: {timings[name]:.0f}s", flush=True)
    _save("timings", timings)
    report.write()
    return timings


if __name__ == "__main__":
    import sys

    get("kan")  # fail fast if the language table is broken
    run_all(build_corpus="--no-corpus" not in sys.argv)
