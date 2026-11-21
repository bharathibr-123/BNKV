# Methodology

Every number in the dashboard, `RESULTS.md` and the report is produced by
`python -m tokentax.experiments` from the data listed below. Nothing is typed in by hand.
Random sampling uses fixed seeds (`research.SEED = 20261120`), and `results/ledger.json`
records dataset hashes, tokenizer hashes, library versions and hardware.

## 1. Data

| Set | Use | Size |
|---|---|---|
| FLORES-200 **dev** | calibration only: verifier threshold, best-single-format baseline, spelling-ambiguity vocabulary, English lexicon | 997 sentences per language |
| FLORES-200 **devtest** | all reported results | 1,012 sentences per language |
| DravidianCodeMix (Kannada, Tamil, Malayalam) | conversational domain, real code-mixed comments | sampled, see ledger |
| Romanised FLORES | machine-made casual romanisation of FLORES (our `romanize.casual`) | same sentences |

FLORES sentences are parallel (same content in every language), so a difference in token count
comes from the language and tokenizer, not the content. Limitation: FLORES is news-style and
professionally translated (translationese); the code-mixed comments cover the informal side.

## 2. Measurement (Module A)

* **Tokens** are counted with each model's own `tokenizer.json` through the `tokenizers` library,
  without special tokens. Exact for those files; hosted APIs may add chat-formatting tokens.
* **Premium** = tokens(language) / tokens(English) on the same sentences.
* **Fertility** = tokens per whitespace word; **characters per token**; **byte-fragment rate** =
  share of tokens whose decoded text contains U+FFFD (a piece of a UTF-8 character).
* **Context effect** = window / tokens per message, compared with English.

## 3. Language Fairness Index (LFI)

For tokenizer *t* and language *l*, five parts, each a ratio to English and clipped to [0, 1]:

| Part | Definition |
|---|---|
| cost_parity | tokens(English) / tokens(l), same sentences |
| typing_parity | the same for the casually romanised version (how people often type) |
| word_integrity | share of the 500 most frequent words of *l* that are a single token, relative to English |
| char_integrity | 1 − byte-fragment rate, relative to English |
| pretoken_integrity | share of words the pre-tokenizer keeps whole (vowel signs not split off) |

**LFI = exp( Σᵢ wᵢ · ln(max(partᵢ, 0.01)) / Σᵢ wᵢ )**, with equal weights wᵢ = 1.

*Why a geometric mean:* a tokenizer cannot hide a very bad part behind good ones (a part at
0.01 pulls the index down hard), and the index is unit-free. *Why equal weights:* we have no data
that one part matters more to users. *Sensitivity:* `results/lfi_sensitivity.json` redraws the
weights 500 times from a Dirichlet(1,…,1) distribution and reports Kendall τ between the default
and re-weighted rankings, plus a leave-one-part-out test. The index excludes answer quality,
which needs model runs.

## 4. Candidates (Module D)

`original`, `normalized` (NFC, single spaces, danda → "."), `iso_roman` (ISO 15919, reversible),
`casual_roman` (native runs in casual Latin; English and Latin runs kept), `word_best` (per word,
native or casual Latin, whichever is cheaper on that tokenizer), `compressed` (spacing + pronoun
drop where the verb already marks the subject; Kannada, Tamil, Telugu), and
`compressed+word_best`. For each tokenizer, the chosen output is the candidate with the fewest
tokens among those that passed verification (ties go to the original).

## 5. Meaning-safety verification (Module E)

Both texts are mapped to a **canonical form**: every word in one casual Latin spelling, digits as
ASCII, punctuation removed. A script change therefore leaves the canonical form unchanged, while
a real edit does not. Checks:

| Layer | Check | Fails when |
|---|---|---|
| critical | numbers | the multiset of numbers differs |
| critical | dates / times | dd/mm/yyyy, hh:mm, "12 March" differ |
| critical | negation | count of negation words or suffixes differs (e.g. ಇಲ್ಲ, ಬೇಡ, -ಬೇಡಿ, नहीं, मत, இல்லை, లేదు, ഇല്ല, না, not) |
| critical | modality | must/should/can markers differ (ಬೇಕು, ಬಹುದು, चाहिए, வேண்டும் …) |
| critical | question form | "?" or question words appear/disappear |
| critical | names and links | a URL, handle or capitalised Latin name is missing |
| critical | content words | a word of the original is missing, unless a grammar rule declared its removal |
| similarity | chrF (or LaBSE) | score below the calibrated threshold |
| structure | sentence count | number of sentences differs |
| warning | length ratio | fewer than 60% of words remain (shown, not blocking) |

Casual-Latin candidates must also have spelling ambiguity ≤ 25% (Section 4 of ARCHITECTURE).

**Threshold calibration.** On FLORES **dev** (150 sentences × 6 languages) we generate safe
rewrites and set the chrF threshold to the 1st percentile of their scores (capped at 0.95), so
about 99% of safe rewrites clear it. The value and its source are stored in
`results/thresholds.json` and shown in every verification result.

**Evaluation.** On FLORES **devtest** (200 sentences × 6 languages) we apply controlled changes
with known labels (`perturb.py`):

* safe: ISO romanisation, casual romanisation, extra spaces, normalisation, grammatical pronoun drop
* unsafe: negation added, negation added *and* romanised, a number changed, a content word deleted,
  a content word replaced by a random word, a statement turned into a question

We report the share of safe rewrites accepted, unsafe ones caught, the same for a similarity-only
check, and a layer ablation (remove one check at a time).

*Threat to validity:* the unsafe changes are of the kinds the checks were designed to detect, so
the catch rate is an upper bound for those error types, not a general guarantee. Errors such as
a wrong synonym or a changed relationship between words can pass. The verifier is deliberately
strict; its false rejections cost savings, never safety.

## 6. Strategy comparison, statistics and ablation (Module F)

On 100 devtest sentences per language for 14 languages with a romaniser, and 150 code-mixed
comments per Dravidian language, every candidate is built and verified once; strategies are then
compared on the same table:

* `original`; `always_iso_roman`; `always_casual_roman`; `always_compressed`
* `best_fixed_per_model_language`: the single format (native, ISO or casual) cheapest for that
  tokenizer and language **on dev data**: the strongest form of a one-format-per-task policy
* `truncate_25pct`: drop the last quarter of the words (a naive, unsafe baseline)
* `adaptive_unverified`: cheapest candidate with checks off
* `adaptive_verified (ours)`

**Token reduction** = 1 − Σ tokens(strategy) / Σ tokens(original). **Meaning-safe rate** = share of
outputs that pass the full verifier (the original always does).

**Paired test** per tokenizer: per-sentence saving of ours minus the best fixed format. We report
the mean difference, a 95% bootstrap CI (2,000 resamples) and a two-sided sign-flip permutation
p-value (5,000 permutations). **Ablation** removes one candidate family (word mix, compression,
romanisation, ISO) or the verifier. **Failure taxonomy** groups every rejected candidate by the
first failed check. **Domain shift** repeats the comparison on code-mixed comments.

## 7. Cost and context simulation (Module G)

Tokens per message = words × English tokens per word (measured per tokenizer) × premium.
Daily cost = messages × (input tokens × input price + output tokens × output price) × USD→INR.
Prices are configurable (default: GPT-4o list price in `config/prices.json`). The optimizer
rewrites **questions only** unless the user ticks "also optimize replies". Results are
simulations, not API bills.

## 8. Efficiency

`results/efficiency.json`: tokenizer load time, encode time per sentence and resident memory per
tokenizer; average time to build, verify and choose for one sentence on all 7 tokenizers; Python
peak memory (tracemalloc). Measured on the machine recorded in the ledger.

## 9. Human evaluation protocol

1. Sample 50 devtest sentences per language (Kannada first), stratified by chosen candidate.
2. Two native speakers see the original and the chosen rewrite (order randomised, method hidden).
3. They rate meaning preservation (1–5), fluency (1–5) and "would you send this?" (yes/no) in the
   Optimizer page form; ratings are stored in the audit ledger with the request ID.
4. Report mean ratings with bootstrap CIs, Cohen's κ for the yes/no item, and the verifier's
   agreement with the "meaning ≥ 4" judgement.
5. Export with `GET /api/reviews.csv`.

The interface and storage are built and tested; **no human ratings are reported until collected.**
