# BhashaToken · Team BNKV

**Measure the Tax. Preserve the Meaning. Make AI More Inclusive.**

LLM tokenizers are built mostly for English, so the same sentence in a South Asian language is
cut into many more tokens: users pay more, wait longer and fit less into the model's context.
BhashaToken **measures** that tax for 17 South Asian languages on 7 real tokenizers, **explains**
it word by word, and **cuts** it by rewriting each message into its cheapest form *for each model
separately*, sending a rewrite **only if a layered verifier finds the meaning unchanged**. Every
decision is logged in an audit ledger that people can review.

Built for The Big Hack 2.0 (AI Research track, R3 "The Token Tax") by Bharathi B R, Nayana,
Kavitha M and Varsha Malipatil, ATME College of Engineering, Mysuru.

## Headline results (all measured; see `RESULTS.md`)

* **The tax is real and uneven:** 1.2× to 13.7× the tokens of English for the same FLORES-200
  sentences. Kannada: 8.9× on GPT-4 and Llama 3, about 2× on GPT-4o and Gemma 3.
* **No format is always cheapest:** native script beats casual Latin in 12/14 languages on
  Gemma 3 and 11/14 on GPT-4o, but 0/14 on GPT-4, Llama 3 and Qwen3.
* **Meaning-safe savings:** on held-out data (100 sentences × 14 languages × 7 tokenizers) BhashaToken
  removes **51.4%** of tokens with **100%** of outputs verified, versus 49.8% with 98.6% verified for
  the best single format per model and language. Gains of up to +9.0 points per model (p < 0.001).
* **Similarity scores are not enough:** on 6,242 deliberately unsafe changes (negation, numbers,
  questions, deleted or replaced words) in 6 languages, a similarity score alone catches 29.4%;
  our verifier catches 100% while accepting 99.8% of safe rewrites.
* **Real chat too:** 51.7% fewer tokens on real code-mixed Kannada, Tamil and Malayalam comments.
* **Fixing the tokenizer is not enough:** keeping vowel signs inside words with the same vocabulary
  changes token counts by at most 0.2% on all 7 tokenizers (14 languages); BhashaToken's rewrite
  saves 11–64% on the same models today.
* **Honest negatives:** grammar-safe compression saves ≤ 1.1% on formal text; on GPT-4 and Qwen3 the
  best fixed format is 0.3–0.5 points ahead because it also sends rewrites we reject.
* **Cheap to run:** 35 ms per sentence for all 7 tokenizers on a CPU, about 0.6 GB memory.

What is new compared with existing research: [`docs/RESEARCH.md`](docs/RESEARCH.md). We do not
claim to have discovered the token tax or the vowel-sign pre-tokenizer split (prior art, cited).

## Run it (Windows, macOS or Linux; Python 3.10+)

```bash
pip install -r requirements.txt

# start the dashboard (results are included, so this works straight away)
PYTHONPATH=src:. python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000  and press "▶ Judge demo"

# optional: re-run every experiment (about 10 minutes on a laptop CPU)
bash scripts/fetch_data.sh
PYTHONPATH=src python -m tokentax.experiments
```

On Windows PowerShell set the path first: `$env:PYTHONPATH="src;."`

## Optional extras (all free)

**1. Ask a real AI on your laptop (answer-quality check).** Install Ollama from
<https://ollama.com>, then in a terminal:

```bash
ollama pull gemma3:1b      # about 815 MB, once
```

Keep Ollama running and restart BhashaToken. On the Optimizer page, *Ask the AI both versions*
sends the original question and BhashaToken's version to Gemma 3 on your laptop and shows both
answers, with the input tokens the model itself counted. Use `BHASHATOKEN_OLLAMA_MODEL=gemma3:4b`
for better answers if your laptop has 8 GB RAM free.

No Ollama? Two other ways (set these in the terminal **before** starting the server):

* **Google Gemini API (free key, needs internet, nothing to install).** Get a key at
  <https://aistudio.google.com>, then (PowerShell):
  ```powershell
  $env:BHASHATOKEN_AI_URL="https://generativelanguage.googleapis.com/v1beta/openai"
  $env:BHASHATOKEN_AI_KEY="paste-your-key-here"
  $env:BHASHATOKEN_AI_MODEL="gemini-2.5-flash"   # or a model name listed in AI Studio
  ```
  Your test sentences are sent to Google; do not use private text.
* **llamafile (offline, one downloaded file, no installer).** Download a llamafile (for example
  Mozilla's Gemma 3 1B llamafile from Hugging Face), add `.exe` to its name on Windows, run it, and
  set `$env:BHASHATOKEN_AI_URL="http://127.0.0.1:8080/v1"`.

**2. Add Sarvam-1 (an Indian model's tokenizer).**

```bash
python scripts/get_sarvam.py
```

Downloads only the tokenizer file (about 8.5 MB, Sarvam non-commercial licence) into
`vendor/tokenizers/sarvam1.json`. After a restart, Sarvam-1 appears in the Optimizer, Compare
and Live Demos. To include it in the research charts too, re-run the experiments without the raw
data download: `PYTHONPATH=src python -m tokentax.experiments --no-corpus` (20–30 minutes).

**3. One-page handout for judges.** Open <http://127.0.0.1:8000/handout> (or the *Handout*
button) and print / save as PDF; its QR code points to your laptop. A copy without the QR code is
in `docs/BhashaToken_handout.pdf`.

## The dashboard

| Page | What it shows |
|---|---|
| Overview | KPI cards, what existed vs what we add, India map of the tax, every model for one language, causes, Fairness Index |
| Optimizer | Type, paste or **speak** (🎤, Chrome/Edge) a sentence: summary strip, one table per model (tokens, characters, characters per token, broken-letter pieces, ₹ per 1 lakh messages, tokens and ₹ after safe rewrite, saving, tokens if the tokenizer were fixed), Fragmentation Microscope, all 7 versions with verification, decision trace, human rating form, then why / predict / shorten |
| Compare | The same professionally translated sentence in 18 languages side by side: text, characters, tokens, × English, ₹, after safe rewrite; search 1,012 sentences in any language; then heatmap, native vs Latin letters, fairness radar, real code-mixed comments |
| Live Demos | (1) Memory race: the same conversation in English vs another language, watching the AI forget older messages, with and without BhashaToken; (2) Fool the checker: one-click safe and unsafe edits with a live score; (3) Audit your own messages: paste or upload a .txt/.csv, get totals and a CSV report; (4) Drop-in API: what your app sends vs what BhashaToken forwards; (5) Live crowd check: people scan a QR code, rate real rewrites on their phones (`/rate`), results update live; (6) Token Tax receipt: printable monthly receipt |
| Safety Lab | Test any rewrite against an original; calibrated catch rates vs similarity-only; check ablation; human-review summary and CSV export |
| Research | Pareto (saving vs safety), fix-the-tokenizer vs fix-the-text, paired tests with CIs, ablation, failure taxonomy, formal vs conversational, LFI weight sensitivity, winning candidates |
| Simulator | Daily / monthly / annual ₹ with editable prices, context window before and after |
| Evidence | Pipeline, audit ledger, reproducibility ledger (hashes, versions, hardware, speed), findings, limitations, related work, phone QR, team |

**Judge demo mode** walks through 10 steps: a Kannada request on 7 models, real code-mixed chat
optimised safely, one sentence compared across 18 languages, the memory race, and an unsafe
medicine rewrite that is rejected. **Research mode** (top right) adds raw
JSON traces, timings and statistics tables. **Dark/light** theme toggle; works on phones (bottom
tab bar, "Add to Home Screen").

## API (FastAPI; interactive docs at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/optimize` | candidates, per-model choice, decision trace (saved to the ledger) |
| POST | `/api/compare` | `{"index": 123, "tokenizer": "gpt4o"}` or `{"text": "..."}`: same sentence in 18 languages |
| GET | `/api/compare/search?q=...`, `/api/compare/featured` | find parallel sentences in any language |
| POST | `/api/demo/conversation`, `/api/demo/batch`, `/api/demo/mutate` | memory race, message audit, fool-the-checker |
| GET / POST | `/rate`, `/api/crowd/pair`, `/api/crowd/rate`, `/api/crowd/summary`, `/api/crowd.csv` | live crowd check |
| POST | `/api/verify` | check any rewrite: `{"original": "...", "candidate": "..."}` |
| POST | `/api/analyze`, `/api/explain`, `/api/predict`, `/api/compress`, `/api/route` | measurement, causes, estimate, compression, routing |
| POST | `/api/simulate` | cost and context; optional `input_price_usd`, `output_price_usd`, `context_window`, `reply_optimized` |
| GET | `/api/audit`, `/api/audit/{id}` | audit ledger |
| POST / GET | `/api/review`, `/api/reviews`, `/api/reviews.csv` | human evaluation |
| POST | `/v1/chat/completions` | OpenAI-compatible proxy; dry run unless `TOKENTAX_UPSTREAM` is set |

## Documents

| File | Contents |
|---|---|
| `docs/TECHNICAL_REPORT.md` | full report: problem, system, results, failures, limits |
| `docs/METHODOLOGY.md` | data splits, metrics, LFI formula, verifier, statistics, human protocol |
| `docs/ARCHITECTURE.md` | system, data-flow and verifier diagrams (Mermaid), module and API tables |
| `docs/RESEARCH.md` | literature review table, gap, novelty statement, hypotheses |
| `docs/DEMO_SCRIPT.md` | 4-minute demo, step by step |
| `docs/JUDGE_QA.md` | 18 likely questions with honest answers |
| `docs/PRESENTATION.md` | 10-slide outline |
| `RESULTS.md` | every number, generated from `results/*.json` |

## Quality checks

```bash
pip install pytest httpx mypy ruff
PYTHONPATH=src:. python -m pytest -q   # 102 tests
mypy                                   # strict type checking
ruff check src app tests
```

## Data, tokenizers and licences

* **FLORES-200** dev/devtest (CC BY-SA 4.0), via the AI4Bharat CTQScorer repository. Dev is used
  only for calibration; devtest for every reported result.
* **DravidianCodeMix** (Chakravarthi et al., 2022): real Kannada, Tamil and Malayalam comments,
  research use.
* **Tokenizers:** official `tokenizer.json` files from the `@lenml/tokenizer-*` npm packages
  (Apache-2.0), in `vendor/tokenizers/`; their SHA-256 prefixes are in every decision trace.
* **India map:** DataMeet (CC BY 4.0). **Charts:** Chart.js 4 (MIT), vendored for offline use.
* **Prices:** `config/prices.json` (GPT-4o list price, USD→INR 96.5); editable in the Simulator.

## Privacy

The audit ledger is a local SQLite file (`var/audit.sqlite`); set `BHASHATOKEN_AUDIT=off` to turn
it off. No text leaves the computer unless you configure an upstream model.

## Limits

Token counts are exact for the tokenizer files, but costs are simulations; the verifier is
rule-based and was tested on synthetic changes; LaBSE is optional; answer quality per format and
human ratings are not yet measured; Sinhala is not included, and Sarvam-1 is optional (download script provided). Details:
`docs/TECHNICAL_REPORT.md` §8.
