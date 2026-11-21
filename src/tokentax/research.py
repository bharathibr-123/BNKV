"""Research-grade experiments (Module F): verifier calibration, strategy
comparison with confidence intervals and paired tests, ablations, failure
taxonomy, domain shift, LFI weight sensitivity, cost of running, and a
reproducibility ledger.

All numbers come from running the code on FLORES-200 and the DravidianCodeMix
comments; nothing is typed in by hand. Seeds are fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from importlib import metadata
from typing import Any

import numpy as np

from tokentax import corpus, optimizer, perturb, registry, verify
from tokentax.fairness import FLOOR, PARTS
from tokentax.languages import SOUTH_ASIAN, Language, get
from tokentax.paths import DATA, RESULTS
from tokentax.scripts import Style

Json = dict[str, Any]
SEED = 20261120
VERIFIED_LANGS = ("kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "mal_Mlym", "ben_Beng")


def _save(name: str, data: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


def _keys() -> list[str]:
    return [s.key for s in registry.available()]


def _sample(n_total: int, n: int, salt: str) -> list[int]:
    rng = random.Random(f"{SEED}:{salt}")
    return sorted(rng.sample(range(n_total), min(n, n_total)))


# ---------------------------------------------------------------------------
# 1. Verifier calibration (dev) and evaluation (devtest)
# ---------------------------------------------------------------------------

def _vocab(lang: Language) -> list[str]:
    words = {w.strip(".,!?;:\"'()[]।॥") for line in corpus.flores(lang.code, "dev")[:300]
             for w in line.split()}
    return sorted(w for w in words if sum(lang.script.contains(c) for c in w) >= 3)


def _verify_rows(split: str, n: int, threshold: float | None) -> list[Json]:
    rows: list[Json] = []
    for code in VERIFIED_LANGS:
        lang = get(code)
        texts = corpus.flores(code, split)
        vocab = _vocab(lang)
        rng = random.Random(f"{SEED}:{split}:{code}")
        for i in _sample(len(texts), n, f"verify:{split}:{code}"):
            for var in perturb.variants(texts[i], lang, rng, vocab):
                v = verify.verify(texts[i], var.text, code, var.removals, threshold=threshold)
                rows.append({
                    "language": code, "id": i, "kind": var.kind, "safe": var.safe,
                    "similarity": v.similarity, "passed": v.passed,
                    "failed": [c.name for c in v.checks
                               if not c.passed and c.layer != "warning"],
                })
    return rows


def run_verification(n_dev: int = 150, n_test: int = 200) -> Json:
    # Threshold: highest chrF value that still accepts >= 99% of safe rewrites on dev.
    dev = _verify_rows("dev", n_dev, threshold=0.0)
    safe_sims = sorted(r["similarity"] for r in dev if r["safe"])
    threshold = float(np.quantile(safe_sims, 0.01)) if safe_sims else 0.85
    threshold = round(min(threshold, 0.95), 3)
    th = {"chrf": threshold, "embedding": 0.90,
          "source": f"calibrated on FLORES dev ({len(safe_sims)} safe rewrites, 1st percentile)"}
    _save("thresholds", th)
    verify.thresholds.cache_clear()

    test = _verify_rows("devtest", n_test, threshold=threshold)
    kinds = sorted({r["kind"] for r in test})
    by_kind = []
    for k in kinds:
        rs = [r for r in test if r["kind"] == k]
        caught = Counter(name for r in rs for name in r["failed"])
        sim_only = sum(r["similarity"] >= threshold for r in rs) / len(rs)
        by_kind.append({
            "kind": k, "safe": rs[0]["safe"], "n": len(rs),
            "accepted": sum(r["passed"] for r in rs) / len(rs),
            "accepted_similarity_only": sim_only,
            "mean_similarity": statistics.mean(r["similarity"] for r in rs),
            "caught_by": dict(caught.most_common()),
        })
    safe = [r for r in test if r["safe"]]
    unsafe = [r for r in test if not r["safe"]]
    # layer ablation: what would be accepted if a check were removed
    layers = ["numbers", "dates_times", "negation", "modality", "question_form",
              "names_and_links", "content_words", "similarity", "sentence_count"]
    ablation = []
    for drop in [None, *layers]:
        def ok(r: Json, drop: str | None = drop) -> bool:
            return not [f for f in r["failed"] if f != drop]
        ablation.append({
            "removed_check": drop or "(none: full verifier)",
            "unsafe_caught": sum(not ok(r) for r in unsafe) / len(unsafe),
            "safe_accepted": sum(ok(r) for r in safe) / len(safe),
        })
    sim_only = {"unsafe_caught": sum(r["similarity"] < threshold for r in unsafe) / len(unsafe),
                "safe_accepted": sum(r["similarity"] >= threshold for r in safe) / len(safe)}
    data: Json = {
        "threshold": th, "dev_rows": len(dev), "test_rows": len(test),
        "languages": list(VERIFIED_LANGS),
        "summary": {
            "safe_accepted": sum(r["passed"] for r in safe) / len(safe),
            "unsafe_caught": sum(not r["passed"] for r in unsafe) / len(unsafe),
            "similarity_only": sim_only,
        },
        "by_kind": by_kind, "ablation": ablation,
        "missed_examples": [r for r in unsafe if r["passed"]][:15],
        "false_rejections": [r for r in safe if not r["passed"]][:15],
    }
    _save("verification", data)
    return data


# ---------------------------------------------------------------------------
# 2. Strategy comparison, ablation, failure taxonomy, domain shift
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    words = text.split(" ")
    return " ".join(words[: max(1, round(len(words) * 0.75))])


def _collect(texts: list[tuple[str, str, str, int]], keys: list[str]) -> tuple[list[Json], Json]:
    """Plan every text once; keep each candidate's tokens and safety per tokenizer."""
    rows: list[Json] = []
    failures: Counter[tuple[str, str]] = Counter()
    fail_lang: Counter[tuple[str, str]] = Counter()
    latency: list[float] = []
    examples: dict[str, Json] = {}
    for domain, code, text, idx in texts:
        p = optimizer.plan(text, keys=keys, language=code)
        latency.append(float(p.trace["latency_ms"]))  # type: ignore[arg-type]
        cands = {c.name: c for c in p.candidates}
        for c in p.candidates:
            for reason in c.reasons:
                kind = reason.split(":")[0]
                failures[(c.name, kind)] += 1
                fail_lang[(code, kind)] += 1
                examples.setdefault(kind, {"language": code, "candidate": c.name,
                                           "original": text, "rewrite": c.text,
                                           "reason": reason})
        trunc = _truncate(text)
        trunc_ok = verify.verify(text, trunc, code).passed
        for k in keys:
            rows.append({
                "domain": domain, "language": code, "id": idx, "tokenizer": k,
                "cands": {n: (c.tokens[k], c.safe) for n, c in cands.items()},
                "truncate": (registry.count(k, trunc), trunc_ok),
            })
    meta: Json = {
        "failure_by_candidate": [{"candidate": a, "reason": b, "count": n}
                                 for (a, b), n in failures.most_common()],
        "failure_by_language": [{"language": a, "reason": b, "count": n}
                                for (a, b), n in fail_lang.most_common()],
        "failure_examples": list(examples.values()),
        "latency_ms": {"p50": float(np.percentile(latency, 50)),
                       "p95": float(np.percentile(latency, 95)),
                       "mean": statistics.mean(latency), "n": len(latency)},
    }
    return rows, meta


def _pick(row: Json, allowed: set[str], need_safe: bool = True) -> tuple[int, bool]:
    options = [(t, s) for n, (t, s) in row["cands"].items()
               if n in allowed and (s or not need_safe)]
    if not options:
        t, ok = row["cands"]["original"]
        return int(t), bool(ok)
    t, ok = min(options, key=lambda o: o[0])
    return int(t), bool(ok)


def _fixed(row: Json, name: str) -> tuple[int, bool]:
    """Always use one format; if it does not apply to this text, send the original."""
    t, ok = row["cands"].get(name, row["cands"]["original"])
    return int(t), bool(ok)


ALL = {"original", "normalized", "iso_roman", "casual_roman", "word_best", "compressed",
       "compressed+word_best"}


def _strategies(row: Json, learned: dict[tuple[str, str], str]) -> dict[str, tuple[int, bool]]:
    best_pair = learned.get((row["tokenizer"], row["language"]), "original")
    return {
        "original": row["cands"]["original"],
        "always_iso_roman": _fixed(row, "iso_roman"),
        "always_casual_roman": _fixed(row, "casual_roman"),
        "always_compressed": _fixed(row, "compressed"),
        "best_fixed_per_model_language": _fixed(row, best_pair),
        "truncate_25pct": row["truncate"],
        "adaptive_unverified": _pick(row, ALL, need_safe=False),
        "adaptive_verified (ours)": _pick(row, ALL),
    }


ABLATIONS = {
    "full system": ALL,
    "without word_best hybrid": ALL - {"word_best", "compressed+word_best"},
    "without compression": ALL - {"compressed", "compressed+word_best"},
    "without romanisation": {"original", "normalized", "compressed"},
    "without lossless (ISO) option": ALL - {"iso_roman"},
}


def _paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Json:
    """Bootstrap 95% CI of the mean difference and a sign-flip permutation p-value."""
    d = a - b
    boots = rng.choice(d, size=(2000, d.size), replace=True).mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(5000, d.size))
    null = np.abs((flips * d).mean(axis=1))
    p = (np.sum(null >= abs(d.mean())) + 1) / (null.size + 1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                    float(np.percentile(boots, 97.5))],
            "p_value": float(p), "n": int(d.size)}


def _learn_fixed(keys: list[str], langs: list[Language], n: int) -> dict[tuple[str, str], str]:
    """On FLORES dev, which single format is cheapest per (tokenizer, language)?"""
    out: dict[tuple[str, str], str] = {}
    for lang in langs:
        texts = corpus.flores(lang.code, "dev")[:n]
        totals: dict[str, Counter[str]] = {k: Counter() for k in keys}
        for t in texts:
            p = optimizer.plan(t, keys=keys, language=lang.code,
                               candidates={"original", "iso_roman", "casual_roman"})
            for c in p.candidates:
                for k in keys:
                    totals[k][c.name] += c.tokens[k]
        for k in keys:
            out[(k, lang.code)] = min(totals[k], key=lambda name: totals[k][name])
    return out


def run_strategies(per_lang: int = 100, codemix_n: int = 150) -> Json:
    keys = _keys()
    langs = [lang for lang in SOUTH_ASIAN if lang.script.sanscript is not None]
    learned = _learn_fixed(keys, langs, 60)
    texts: list[tuple[str, str, str, int]] = []
    for lang in langs:
        src = corpus.flores(lang.code)
        texts += [("formal (FLORES)", lang.code, src[i], i)
                  for i in _sample(len(src), per_lang, f"strat:{lang.code}")]
    for code in corpus.DCM_FILES.values():
        src = corpus.codemix(code, Style.MIXED)
        texts += [("conversational (code-mixed comments)", code, src[i], i)
                  for i in _sample(len(src), codemix_n, f"cm:{code}")]
    rows, meta = _collect(texts, keys)
    rng = np.random.default_rng(SEED)

    def summarise(subset: list[Json]) -> list[Json]:
        out = []
        names = list(_strategies(subset[0], learned))
        orig = sum(r["cands"]["original"][0] for r in subset)
        for name in names:
            picks = [_strategies(r, learned)[name] for r in subset]
            out.append({"strategy": name,
                        "token_reduction": 1 - sum(t for t, _ in picks) / orig,
                        "meaning_safe_rate": sum(s for _, s in picks) / len(picks)})
        return out

    formal = [r for r in rows if r["domain"].startswith("formal")]
    pareto = summarise(formal)
    per_tokenizer = []
    for k in keys:
        sub = [r for r in formal if r["tokenizer"] == k]
        ours = np.array([1 - _pick(r, ALL)[0] / max(r["cands"]["original"][0], 1) for r in sub])
        base = np.array([1 - _fixed(r, learned.get((k, r["language"]), "original"))[0]
                         / max(r["cands"]["original"][0], 1) for r in sub])
        per_tokenizer.append({"tokenizer": k, "adaptive_saving": float(ours.mean()),
                              "best_fixed_saving": float(base.mean()),
                              "test": _paired(ours, base, rng)})
    ablation = []
    orig_total = sum(r["cands"]["original"][0] for r in formal)
    for name, allowed in ABLATIONS.items():
        picks = [_pick(r, allowed) for r in formal]
        ablation.append({"variant": name,
                         "token_reduction": 1 - sum(t for t, _ in picks) / orig_total})
    unv = [_pick(r, ALL, need_safe=False) for r in formal]
    ablation.append({"variant": "without meaning verification",
                     "token_reduction": 1 - sum(t for t, _ in unv) / orig_total,
                     "unsafe_outputs": 1 - sum(s for _, s in unv) / len(unv)})
    domains = []
    for dom in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == dom]
        for s in summarise(sub):
            if s["strategy"] in ("adaptive_verified (ours)", "always_casual_roman",
                                 "best_fixed_per_model_language", "always_compressed"):
                domains.append({"domain": dom, **s,
                                "sentences": len(sub) // len(keys)})
    wins: Counter[str] = Counter()
    for r in formal:
        best = min((n for n, (_, s) in r["cands"].items() if s),
                   key=lambda n: (r["cands"][n][0], n != "original"))
        wins[best] += 1
    data: Json = {
        "sentences_per_language": per_lang, "languages": [lang.code for lang in langs],
        "learned_fixed_format": {f"{k}|{c}": v for (k, c), v in learned.items()},
        "pareto": pareto, "per_tokenizer": per_tokenizer, "ablation": ablation,
        "domains": domains, "winning_candidate": dict(wins.most_common()),
        **meta,
    }
    _save("strategies", data)
    return data


# ---------------------------------------------------------------------------
# 3. LFI weight sensitivity
# ---------------------------------------------------------------------------

def _weighted(scores: list[Json], w: np.ndarray) -> list[float]:
    out = []
    for s in scores:
        logs = [np.log(max(float(s[p]), FLOOR)) for p in PARTS]
        out.append(float(np.exp(np.dot(w, logs) / w.sum())))
    return out


def _kendall(a: list[float], b: list[float]) -> float:
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / max(conc + disc, 1)


def run_lfi_sensitivity(draws: int = 500) -> Json:
    fair = json.loads((RESULTS / "fairness.json").read_text(encoding="utf-8"))
    scores = [s for s in fair["scores"] if all(s.get(p) is not None for p in PARTS)]
    base = _weighted(scores, np.ones(len(PARTS)))
    rng = np.random.default_rng(SEED)
    taus, top_same = [], 0
    worst = min(range(len(base)), key=lambda i: base[i])
    for _ in range(draws):
        w = rng.dirichlet(np.ones(len(PARTS))) * len(PARTS)
        alt = _weighted(scores, w)
        taus.append(_kendall(base, alt))
        top_same += min(range(len(alt)), key=lambda i: alt[i]) == worst
    leave_one = []
    for p in PARTS:
        w = np.array([0.0 if q == p else 1.0 for q in PARTS])
        leave_one.append({"dropped_part": p, "kendall_tau": _kendall(base, _weighted(scores, w))})
    data: Json = {
        "formula": "LFI = exp( sum_i w_i * ln(max(part_i, 0.01)) / sum_i w_i ), equal weights",
        "parts": list(PARTS), "pairs": len(scores), "draws": draws,
        "kendall_tau": {"mean": statistics.mean(taus), "min": min(taus),
                        "p05": float(np.percentile(taus, 5))},
        "worst_pair": f"{scores[worst]['tokenizer']}|{scores[worst]['language']}",
        "worst_pair_stable": top_same / draws,
        "leave_one_out": leave_one,
    }
    _save("lfi_sensitivity", data)
    return data


# ---------------------------------------------------------------------------
# 4. Cost of running (CPU) and reproducibility ledger
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def run_efficiency() -> Json:
    texts = corpus.flores("kan_Knda")[:500]
    rows = []
    for k in _keys():
        registry.load.cache_clear()
        before = _rss_mb()
        t = time.perf_counter()
        registry.load(k)
        load_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        registry.count_batch(k, texts)
        enc_ms = (time.perf_counter() - t) * 1000
        rows.append({"tokenizer": k, "load_ms": load_ms,
                     "encode_ms_per_sentence": enc_ms / len(texts),
                     "rss_increase_mb": max(_rss_mb() - before, 0.0)})
    optimizer.plan(texts[0], language="kan_Knda")  # warm caches (vocabulary index)
    t = time.perf_counter()
    for s in texts[1:51]:
        optimizer.plan(s, language="kan_Knda")
    plan_ms = (time.perf_counter() - t) * 1000 / 50
    tracemalloc.start()  # measured separately: tracing slows Python down
    for s in texts[51:61]:
        optimizer.plan(s, language="kan_Knda")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    data: Json = {"tokenizers": rows, "plan_ms_per_sentence_all_tokenizers": plan_ms,
                  "plan_python_peak_mb": peak / 1e6, "process_rss_mb": _rss_mb(),
                  "note": "measured on the machine named in ledger.json; CPU only"}
    _save("efficiency", data)
    return data


def _sha(path: os.PathLike[str] | str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def run_ledger() -> Json:
    files = sorted(p for p in DATA.rglob("*") if p.is_file())
    pkgs = {}
    for name in ("tokenizers", "numpy", "regex", "indic-transliteration", "fastapi"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "not installed"
    data: Json = {
        "system_version": optimizer.VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "gpu": "none used",
        "packages": pkgs,
        "tokenizers": {k: optimizer.tokenizer_hash(k) for k in _keys()},
        "datasets": {str(p.relative_to(DATA)): _sha(p) for p in files},
        "sampling": "random.Random(f'{seed}:{salt}').sample, salts listed in research.py",
        "commands": ["python -m tokentax.experiments"],
    }
    _save("ledger", data)
    return data


# ---------------------------------------------------------------------------
# 5. Counterfactual: fix the tokenizer's pre-tokenizer vs fix the text
# ---------------------------------------------------------------------------

def run_counterfactual(n: int = 200) -> Json:
    """Tokens if only the pre-tokenizer kept vowel signs inside words (same
    vocabulary), compared with as typed and with BhashaToken's verified rewrite."""
    keys = _keys()
    langs = [lang for lang in SOUTH_ASIAN if lang.script.sanscript is not None]
    rows: list[Json] = []
    for lang in langs:
        texts = [corpus.flores(lang.code)[i]
                 for i in _sample(len(corpus.flores(lang.code)), n, f"cf:{lang.code}")]
        for k in keys:
            base = sum(registry.count_batch(k, texts))
            fixed = sum(registry.count_fixed(k, t) for t in texts)
            rows.append({"language": lang.code, "tokenizer": k,
                         "has_split": registry.has_matra_wall(k),
                         "tokens": base, "tokens_fixed_pretokenizer": fixed,
                         "saving_fixed_pretokenizer": 1 - fixed / base})
    strat = json.loads((RESULTS / "strategies.json").read_text(encoding="utf-8"))
    ours = {r["tokenizer"]: r["adaptive_saving"] for r in strat["per_tokenizer"]}
    per_tok = []
    for k in keys:
        sub = [r for r in rows if r["tokenizer"] == k]
        per_tok.append({
            "tokenizer": k, "has_split": sub[0]["has_split"],
            "saving_fixed_pretokenizer": 1 - sum(r["tokens_fixed_pretokenizer"] for r in sub)
            / sum(r["tokens"] for r in sub),
            "saving_bhashatoken": ours.get(k, 0.0),
        })
    data: Json = {
        "sentences_per_language": n, "languages": [lang.code for lang in langs],
        "rows": rows, "per_tokenizer": per_tok,
        "note": "The fixed pre-tokenizer keeps the original vocabulary and merges. "
                "A real fix would need a retrained vocabulary and model.",
    }
    _save("counterfactual", data)
    return data


def run_all() -> None:
    for name, fn in (("verification", run_verification), ("strategies", run_strategies),
                     ("counterfactual", run_counterfactual),
                     ("lfi_sensitivity", run_lfi_sensitivity), ("efficiency", run_efficiency),
                     ("ledger", run_ledger)):
        t = time.time()
        fn()
        print(f"{name}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    run_all()
