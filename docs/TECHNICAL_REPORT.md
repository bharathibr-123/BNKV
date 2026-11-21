# BhashaToken: Meaning-Safe, Language-Aware Token Efficiency and Fairness for South Asian Languages

**Team BNKV** (Bharathi B R, Nayana, Kavitha M, Varsha Malipatil), ATME College of Engineering,
Mysuru. The Big Hack 2.0, AI Research track, problem R3 "The Token Tax".

*Measure the Tax. Preserve the Meaning. Make AI More Inclusive.*

## Abstract

Large language models are billed and limited by tokens, and the tokenizers behind them cut
South Asian languages into far more tokens than English for the same content. We measure this
tax for 17 South Asian languages on 7 current tokenizers using FLORES-200, explain it word by
word, and build BhashaToken, a system that rewrites each message into its cheapest form **for
each model separately** and sends a rewrite only if a layered verifier finds that numbers, dates,
negation, question form, modality, names and every content word survived. On held-out FLORES
devtest data, BhashaToken removes 51.4% of tokens with 100% of outputs passing verification,
compared with 49.8% (98.6% passing) for the best single format chosen per model and language.
The verifier catches 100% of six kinds of controlled meaning changes across 6 languages while
accepting 99.8% of safe rewrites; a similarity score alone catches 29.4%. Every decision is
written to an audit ledger with the tokenizer's hash, all candidates and rejection reasons, and
can be rated by human reviewers. The system runs on a CPU laptop in about 35 ms per sentence for
all 7 tokenizers.

## 1. Problem

The same question in Kannada needs 8.9 times the tokens of English on GPT-4's and Llama 3's
tokenizers. Tokens set price, the share of the context window a conversation uses, and latency.
A Kannada government chatbot with 1 lakh daily users on GPT-4o pays a simulated ₹1.52 crore a
year more than the same chats in English, and its users get 51% of the conversation memory an
English user gets.

## 2. Related work and gap

See `docs/RESEARCH.md`. In short: the tax is well measured (Petrov 2023; Srivastava 2026, which
also covers Kannada and explains the tax by byte tokens) and tokenizer-side fixes exist, but
nothing gives users of existing hosted models a per-message, per-model rewrite that is checked for
meaning and auditable.

## 3. System

Architecture diagrams: `docs/ARCHITECTURE.md`. Modules:

* **A Measurement:** exact counts with each model's `tokenizer.json`; premium, fertility,
  byte-fragment rate, context effect.
* **B Explanation:** per word, how many tokens come from the one-per-word base, vowel-sign splits by
  the pre-tokenizer, and missing vocabulary; morpheme alignment of token cuts.
* **C Normalisation:** NFC, spaces, danda; a canonical cross-script form used only for checking.
* **D Optimisation:** seven candidates (original, normalized, ISO 15919, casual Latin, per-word
  cheapest mix, grammar-rule compression, compression + mix); per tokenizer, the cheapest verified
  one wins.
* **E Meaning safety:** layered verifier (Methodology §5), spelling-ambiguity limit for casual Latin,
  threshold calibrated on FLORES dev.
* **F Research:** strategy comparison, paired statistics, ablations, failure taxonomy, domain shift,
  LFI sensitivity, efficiency, reproducibility ledger.
* **G Product:** FastAPI service, dashboard with Judge Demo and Research modes, light/dark themes,
  phone install (PWA + QR), a same-sentence comparison across 18 languages, live demos (memory
  race, fool-the-checker, message audit with CSV report, drop-in API), cost and context simulator, audit ledger with human review and CSV
  export, OpenAI-compatible proxy (dry run by default).

## 4. Experimental setup

Methodology: `docs/METHODOLOGY.md`. Calibration uses FLORES **dev** only; all reported numbers use
FLORES **devtest** and DravidianCodeMix comments. Seeds, dataset and tokenizer hashes, versions and
hardware are recorded in `results/ledger.json`.

## 5. Results

### 5.1 The tax (Module A)

| | Result |
|---|---|
| Range of premiums (native script) | 1.21× (Bengali, Gemma 3) to 13.7× (Santali, GPT-4o) |
| Kannada | 8.9× on GPT-4 and Llama 3; 1.9–2.2× on GPT-4o, Gemma 3, Mistral NeMo |
| Casual Latin typing | 1.9–2.5× on every tokenizer |
| Native cheaper than casual Latin | 12/14 languages on Gemma 3, 11/14 on GPT-4o, 0/14 on GPT-4, Llama 3, Qwen3 |

Full tables: `RESULTS.md` §1–3.

### 5.2 Verifier (Module E)

| | Safe rewrites accepted | Unsafe changes caught |
|---|---|---|
| BhashaToken verifier | 99.8% | 100.0% |
| Similarity score only | 99.9% | 29.4% |

Per change type, similarity alone catches 0% of added negations and question marks and 7.3% of
changed numbers. Ablation: without the negation check, catch rate drops to 62.0%; without the
question check to 80.9%; without the content-word check to 90.9%. Threshold: chrF 0.95, the 1st
percentile of 3,074 safe rewrites on dev.

### 5.3 Strategies (Modules D + F)

| Strategy | Token reduction | Meaning-safe outputs |
|---|---|---|
| **BhashaToken (adaptive, verified)** | **51.4%** | **100%** |
| Adaptive without verification | 51.7% | 98.2% |
| Best single format per model and language (chosen on dev) | 49.8% | 98.6% |
| Always casual Latin | 47.9% | 98.1% |
| Truncate last 25% of words | 25.4% | 0% |
| Always ISO 15919 | 22.6% | 99.5% |
| Always grammar compression | 0.1% | 100% |

Paired per-sentence gains over the best single format (95% bootstrap CI, permutation p): GPT-4o
+9.04 [8.65, 9.43], Mistral NeMo +8.02 [7.58, 8.43], Gemma 3 +6.24 [5.92, 6.57], DeepSeek V4
+1.38 [1.14, 1.60], Llama 3 +1.32 [1.08, 1.57], Qwen3 −0.32 [−0.49, −0.17], GPT-4 −0.47
[−0.70, −0.27]; all p < 0.001, n = 1,400 per tokenizer.

Ablation: removing the per-word mix costs 1.6 points; removing compression 0.1; removing ISO 0.4;
removing romanisation leaves 0.2%. Most sentence–model pairs are won by casual Latin (4,573) or the
per-word mix (4,121), then compression + mix (552), the original (409) and ISO (89).

Domain: on real code-mixed comments the system removes 51.7% of tokens (100% verified) against
49.0% for the best single format (99.0%).

### 5.4 Fairness index (Module A/F)

LFI is a geometric mean of five parity parts (Methodology §3). Over 500 random weightings the
ranking of 98 tokenizer–language pairs keeps Kendall τ 0.953 on average (5th percentile 0.899);
the worst pair stays worst in 91% of weightings.

### 5.4b Fix the tokenizer or fix the text?

Patching the pre-tokenizer of each tokenizer so vowel signs stay inside words, with the same
vocabulary and merges, changes token counts by at most 0.2% (200 devtest sentences × 14 languages);
for Hindi on Llama 3 it adds 2.4%. The split exists, but the vocabulary was learned with it, so a fix
needs retraining. Our verified text rewrite saves 10.7% (Gemma 3) to 63.6% (GPT-4) per sentence on
the same tokenizers today.

### 5.5 Cost and context (Module G)

Default scenario (1 lakh users/day, 4 messages, 70% Kannada, 40-word questions, 120-word answers,
GPT-4o list price): ₹3.77 crore a year, of which ₹1.52 crore is the token tax. Because GPT-4o's
tokenizer already handles Kannada fairly well (1.97×) and only questions are rewritten, the
simulated saving there is ₹3.4 lakh a year; on Llama 3's tokenizer (8.9×) the same rewriting
removes 75% of Kannada input tokens. These are simulations, not API bills.

### 5.6 Efficiency

35 ms to build, verify and choose for one sentence on all 7 tokenizers (CPU, warm); median 28 ms
and 95th percentile 75 ms across 1,850 texts; process memory about 0.6 GB.

## 6. Failure analysis

Rejections in the strategy study, by first failed check: spelling clarity (casual Latin with more
than 25% ambiguous words), content words (mostly words mixing Latin and native script, such as
"QVCಯಲ್ಲಿ", where our canonical form differs: false alarms), names and links, question form.
Earlier development runs exposed further false alarms that we fixed and report here: Bengali,
Punjabi and Odia nukta letters, Gujarati candra vowels, "jan" (people) read as January, and native
@handles. Examples are shown in the Research page in Research mode.

## 7. Demo

`docs/DEMO_SCRIPT.md` (3–5 minutes, 9 steps, three examples including an unsafe rejection).
In the app: **▶ Judge demo**.

## 8. Limitations

1. Token counts are exact for the tokenizer files; hosted APIs add formatting tokens and change
   prices. Costs are simulations.
2. The verifier is rule-based. It was evaluated on synthetic changes of the types it checks; a wrong
   synonym or changed word relationship can pass. It errs toward rejection.
3. The default similarity is chrF on a canonical spelling; LaBSE is optional (not installed in our
   runs because model downloads were unavailable), so paraphrases are rejected.
4. We have not benchmarked whether models answer equally well in each format. The app can show it
   live with a small local model (Gemma 3 via Ollama), which is a demonstration, not a measurement.
5. Casual romanisation is machine-made; real users spell variably.
6. Human evaluation: interface, storage and protocol are built and tested; no ratings are reported
   until collected.
7. Sinhala is not included; Sarvam-1's tokenizer is optional (download script, non-commercial
   licence) and is not in the reported results unless the experiments are re-run.

## 9. Future work

Collect native-speaker ratings with the built protocol; add LaBSE and an NLI check; measure answer
quality per format with open models; learn the per-word choice from the predictor so the
tokenizer is not needed at request time; add Sarvam and Sinhala.

## 10. Reproducibility

```
bash scripts/fetch_data.sh
pip install -r requirements.txt
PYTHONPATH=src python -m tokentax.experiments   # writes results/*.json and RESULTS.md
PYTHONPATH=src:. pytest -q                       # 101 tests
```
