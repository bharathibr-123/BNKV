# BhashaToken architecture

BhashaToken has three layers: an offline **research pipeline** that produces every number in
`results/`, a **runtime engine** that optimises one message at a time, and a **FastAPI + vanilla JS
dashboard** that shows both. Everything runs on a CPU laptop; no paid API is needed.

## System overview

```mermaid
flowchart LR
  subgraph Data["Data (scripts/fetch_data.sh)"]
    F[FLORES-200<br/>dev 997 · devtest 1,012<br/>18 languages]
    D[DravidianCodeMix<br/>real YouTube comments]
    T[7 tokenizer.json files<br/>GPT-4, GPT-4o, Llama 3, Gemma 3,<br/>Qwen3, DeepSeek V4, Mistral NeMo]
  end
  subgraph Engine["Runtime engine (src/tokentax)"]
    R[registry · metrics<br/>exact token counts]
    X[explainer · morphology<br/>why a word is costly]
    P[predictor<br/>script-aware estimate]
    O[optimizer<br/>7 candidates]
    V[verify<br/>meaning-safety checks]
    A[audit<br/>SQLite ledger + reviews]
  end
  subgraph Research["Research pipeline (experiments.py, research.py)"]
    B[benchmark · fairness LFI]
    C[verifier calibration<br/>perturbation suite]
    S[strategies · ablation · CIs<br/>failure taxonomy · domains]
    L[ledger · efficiency · LFI sensitivity]
  end
  F --> B & C & S
  D --> S & B
  T --> R
  R --> X & O & B
  O --> V --> A
  C -->|thresholds.json| V
  B & C & S & L -->|results/*.json| UI[Dashboard<br/>app/static]
  Engine -->|/api/*| UI
```

## One message, end to end (`POST /api/optimize`)

```mermaid
sequenceDiagram
  participant U as User / client app
  participant API as FastAPI
  participant O as optimizer.plan
  participant V as verify.verify
  participant DB as audit.sqlite
  U->>API: text (+ optional language)
  API->>O: plan(text, 7 tokenizers)
  O->>O: detect language, spelling ambiguity
  O->>O: build candidates (normalized, ISO, casual, word-mix, compressed, combo)
  O->>O: count tokens per candidate per tokenizer
  loop each non-exact candidate
    O->>V: original vs rewrite (+ declared removals)
    V-->>O: checks, similarity, pass/fail, reasons
  end
  O->>O: per tokenizer: cheapest candidate that passed
  O-->>API: Plan (candidates, choices, trace)
  API->>DB: record(request_id, input, choices, trace)
  API-->>U: plan JSON with decision trace
```

## Verifier layers (`verify.py`)

```mermaid
flowchart TB
  IN[original, rewrite] --> CAN[canonical form<br/>every word in one Latin spelling,<br/>digits as ASCII]
  CAN --> C1{critical facts}
  C1 --> N1[numbers] & N2[dates / times] & N3[negation incl. suffixes] & N4[modality] & N5[question form] & N6[names, links, handles] & N7[every content word<br/>unless removal declared by a grammar rule]
  CAN --> SIM{similarity}
  SIM --> E1[LaBSE cosine if installed]
  SIM --> E2[else chrF on canonical form<br/>threshold calibrated on FLORES dev]
  CAN --> ST{structure}
  ST --> S1[sentence count] & S2[length ratio: warning only]
  N1 & N2 & N3 & N4 & N5 & N6 & N7 & E1 & E2 & S1 --> OUT[PASS only if every critical,<br/>similarity and structure check passes]
```

A rewrite in casual Latin letters must also keep **spelling ambiguity** at or below 25%: the share of
words whose casual spelling is shared with another word in the FLORES dev vocabulary (for example
"kale" for both ಕಲೆ and ಕಳೆ).

## Modules

| Module | File | Role |
|---|---|---|
| A Measurement | `registry.py`, `metrics.py`, `benchmark.py` | exact counts, fertility, byte-fragment rate, premium vs English |
| B Explanation | `explainer.py`, `morphology.py` | per-word cause: base, vowel-sign split, vocabulary gap; grammar alignment |
| C Normalisation | `normalize.py` | NFC/space/danda candidate; canonical form for checking |
| D Optimisation | `optimizer.py`, `compression.py`, `romanize.py` | 7 candidates, per-tokenizer choice |
| E Meaning safety | `verify.py`, `perturb.py`, `meaning.py` | layered checks, calibration, perturbation suite |
| F Research | `research.py`, `experiments.py`, `fairness.py` | Pareto, CIs, paired tests, ablations, failures, LFI sensitivity |
| G Product | `simulator.py`, `pricing.py`, `router.py`, `audit.py`, `compare.py`, `demos.py`, `app/` | ₹ and context, routing, ledger, reviews, dashboard, proxy |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/meta` | tokenizers, languages, currency |
| POST | `/api/analyze` | tokens, pieces, byte flags per tokenizer |
| POST | `/api/explain` | per-word causes |
| POST | `/api/optimize` | candidates, choices, decision trace (logged) |
| POST | `/api/verify` | check any rewrite against an original |
| POST | `/api/compare` | same FLORES sentence in 18 languages: characters, tokens, × English, ₹, safe rewrite |
| GET | `/api/compare/search`, `/api/compare/featured` | find parallel sentences by words in any language |
| POST | `/api/demo/conversation` | memory race: per-turn tokens for English, native, optimised |
| POST | `/api/demo/batch` | audit up to 200 messages on one tokenizer |
| GET/POST | `/api/demo/mutations`, `/api/demo/mutate` | one-click safe/unsafe edits checked by the verifier |
| GET/POST | `/rate`, `/api/crowd/*`, `/api/crowd.csv` | live crowd check from phones |
| GET | `/api/health` | `{"ready": true}` once tokenizers are loaded |
| POST | `/api/compress` | grammar-rule compression with edits |
| POST | `/api/predict` | token estimate without the tokenizer |
| POST | `/api/simulate` | daily/monthly/annual ₹, context before/after |
| POST | `/api/route` | cheapest model + format |
| GET | `/api/audit`, `/api/audit/{id}` | ledger |
| POST/GET | `/api/review`, `/api/reviews`, `/api/reviews.csv` | human evaluation |
| GET | `/api/results/{name}` | experiment results (whitelisted names) |
| POST | `/v1/chat/completions` | OpenAI-compatible proxy; dry run unless `TOKENTAX_UPSTREAM` is set |

## Constraints met

* CPU only; peak process memory is reported in `results/efficiency.json`.
* No paid API: tokenizers are local files; LaBSE is optional.
* The audit database is a local file (`var/audit.sqlite`); disable with `BHASHATOKEN_AUDIT=off`.
* Input length is capped at 5,000 characters; result names are whitelisted.
