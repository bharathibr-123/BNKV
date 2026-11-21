# Literature review, research gap and novelty

Every number about our own system is produced by `python -m tokentax.experiments` and
listed in `RESULTS.md`. Claims about other papers come from their abstracts or full text as
cited; where we could not read a full paper we say so.

## 1. Closest prior work

| Work | Problem | Method | Data | Metrics | Main finding | Limitation for our problem |
|---|---|---|---|---|---|---|
| Petrov et al. 2023, *Language Model Tokenizers Introduce Unfairness Between Languages* (arXiv 2305.15425) | Unequal tokenization across languages | Count tokens on parallel text | FLORES-200, 17 tokenizers | tokenization premium / parity | Same text can need many times more tokens | Measurement only; no user-side mitigation |
| Ahia et al., EMNLP 2023, *Do All Languages Cost the Same?* (arXiv 2305.13707) | API cost disparity | Cost + quality on commercial APIs | 22 languages | cost, accuracy | Speakers of some languages pay more and get less | No rewriting, no per-message choice |
| **Srivastava 2026**, *Tokenizer tax* (arXiv 2607.24276) | Indic tax in widely used tokenizers | Fertility, bytes/token, unmerged single-byte rate, context shrinkage | FLORES-200, 10 Indian languages incl. Kannada; cl100k, o200k, GPT-2, Qwen2.5, mBERT, XLM-R; Belebele | tax multiplier, byte rate, context | Mean Indic tax 8.0× English on cl100k (Malayalam 13.0×; Kannada 12.06×); byte rate explains the tax (r = 0.89); o200k cuts it 73%; fertility–accuracy link confounded by resource level | Native script only; no romanised or code-mixed text; recommendations aimed at model builders and API providers, no end-user mitigation; no meaning verification |
| *Removable and irreducible token cost* ledger, 2026 (arXiv 2609.00378) | Decompose token cost by language | Cost ledger | Includes Kannada | removable vs irreducible share | Part of the tax could be removed by better tokenization | No rewriting of user text, no per-input selection, no verification, no audit system |
| ACL Findings 2026 (2026.findings-acl.1632) | Tokenizer design for Indian languages | BPE vs Unigram, normalisation, vocabulary strategies | 17 languages | fertility, downstream | Design choices change fertility | Requires training new tokenizers |
| Dwivedi & Gopalan, ACL 2026; grapheme-kit (arXiv 2607.22456) | Grapheme-aware tokenization | Grapheme clusters as units | Hindi, Marathi | fertility, quality | Respecting graphemes helps | Retraining needed; does not help users of existing hosted models |
| *Vowel Signs Are Not Letters*, 2026 (arXiv 2608.26449) | Pre-tokenizer splits abugida words at vowel signs | Regex analysis | Indic scripts | fertility floor | Fix `[\p{L}\p{M}]+` known; o200k already uses it | We **do not** claim this discovery; we measure and explain it per word |
| RomanSetu, ACL 2024 (arXiv 2401.14280) | Use romanised text with LLMs | Continued pretraining on romanised text | Indic languages | fertility, task scores | Romanisation cuts fertility 2–4× | One format for a whole task and model; needs training |
| Nag et al. 2024 (arXiv 2403.05434) | Cheaper Indic processing on commercial LLMs | Code-mixing, translation, transliteration | 15 Indian languages, GPT-4 | cost, quality | Some formats cut cost | One model, choice per task, offline |
| *Lost in Compression*, 2026 (arXiv 2608.26175) | Prompt compression outside English | Audit of compressors | many languages; Hindi only for India | safe budget | English compressors hurt other languages; rule-based methods did not show the gap | No Dravidian languages, no per-edit justification |
| TEA cost audit 2026 (arXiv 2608.09046); TokLens; STRR (arXiv 2510.09947) | Cost / intrinsic metrics | Audits | many | parity, retention | Disparities across models | Measurement only |
| token-calibrator (PyPI) | Estimate tokens without the tokenizer | Regression on character classes | — | error | Generic classes | Not script-aware for abugidas |

We searched for the exact SSRN 5201887 paper and an EMNLP grapheme paper named in the brief and
could not access them; they are not cited as evidence.

## 2. The gap

Across this literature the tax is **measured** well (including for Kannada) and **tokenizer-side**
fixes are proposed, but three things are missing for someone who must use an existing hosted
model today:

1. **Per-message, per-model choice of how to write the text.** Prior work fixes one format per
   task and model. We find the cheapest format flips between tokenizers (native script is cheaper
   than casual Latin in 12/14 languages on Gemma 3 and 11/14 on GPT-4o, but 0/14 on GPT-4, Llama 3
   and Qwen3), so a single format leaves savings behind.
2. **Proof that a cheaper rewrite still means the same thing.** Papers that rewrite or compress
   report task scores on average; none we found blocks an individual rewrite that drops a
   negation, changes a number or turns a question into a statement, and similarity scores alone
   miss most of these (29.4% caught in our test).
3. **Accountability.** No system records, per request, which tokenizer version was used, which
   rewrites were tried, and why any were rejected, so results cannot be audited or reviewed by
   people.

## 3. What BhashaToken contributes (defensible novelty)

We do **not** claim to have invented tokenization-fairness measurement, benchmarking,
transliteration-based optimisation or the vowel-sign finding. Our contribution is the
combination, evaluated end to end:

> **Meaning-safe, per-input, per-tokenizer text optimisation for South Asian languages, with a
> layered critical-information verifier calibrated on held-out data, an auditable decision trace
> for every request, and a failure taxonomy.**

Measured on FLORES-200 devtest (100 sentences × 14 languages × 7 tokenizers):

* 51.4% fewer tokens with 100% of outputs passing verification, versus 49.8% (98.6% passing) for
  the best single format per model and language chosen on dev data.
* Per tokenizer, paired gains of +9.0 points (GPT-4o), +8.0 (Mistral NeMo), +6.2 (Gemma 3),
  +1.4 (DeepSeek V4), +1.3 (Llama 3), all p < 0.001; on GPT-4 (−0.5) and Qwen3 (−0.3) the fixed
  format is slightly ahead because it also sends the 1–2% of rewrites our verifier rejects.
* The verifier catches 100% of six kinds of synthetic meaning changes (6,242 tests, 6 languages)
  while accepting 99.8% of safe rewrites; a similarity score alone catches 29.4%.
* Turning verification off saves 0.3 more points but makes 1.8% of outputs unsafe; truncating
  messages saves 25% with 0% safe.
* On real code-mixed comments: 51.7% fewer tokens, 100% verified (best fixed format: 49.0%).

Counterfactual (`results/counterfactual.json`): patching the pre-tokenizer so vowel signs stay
inside words, with the same vocabulary, changes token counts by at most 0.2% on every tokenizer
(14 languages × 200 sentences), and adds tokens for Hindi on Llama 3. The vowel-sign split is real
(prior art), but the vocabulary was learned with it, so removing it alone does not help; a fix needs
retraining by the model's maker. Text-side optimisation helps users today.

## 4. Hypotheses

* **H1 (the tax is real and unequal):** supported. 1.21× (Bengali, Gemma 3) to 13.7× (Santali,
  GPT-4o) English tokens on the same sentences; Kannada 8.9× on GPT-4 and Llama 3.
* **H2 (no format is always cheapest):** supported (counts above). Casual Latin sits at 1.9–2.5×
  English on every tokenizer: it removes the extreme tax but never reaches parity.
* **H3 (meaning-safe savings exist):** supported for format choice; grammar-rule compression is a
  small lever (≤1.1% on formal text, ≤2.8% on chat), which we report as a negative result.
* **H4 (verification matters):** supported on controlled changes; real-world validation needs the
  human study described in METHODOLOGY.md.

## 5. Limits

See the Limitations panel in the app and `docs/TECHNICAL_REPORT.md` §8.
