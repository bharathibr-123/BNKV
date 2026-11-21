# Likely judge questions, with honest answers

**1. Isn't the token tax already known?**
Yes, and we cite it: Petrov 2023, Ahia 2023, and Srivastava 2026, which covers 10 Indian
languages including Kannada. We do not claim to discover it. Our contribution is what a user of
an existing model can do about it today, safely: per-message, per-model rewriting with a verifier
and an audit trail.

**2. Why not just fix the tokenizer?**
That is the right long-term fix and others work on it (grapheme tokenizers, vocabulary expansion).
But it requires retraining the model, and users of GPT-4o, Llama or Gemma cannot do that. We
change only the text that is sent, so it works with any hosted model today.

**3. Does your optimizer change the model's tokenizer?**
No. It rewrites the input text. Token counts are exact for each model's tokenizer file; a hosted
API may add a few tokens for chat formatting.

**4. Are the savings real money?**
Token savings are measured exactly. Rupee figures are simulations using list prices you can
change in the Simulator. By default we only rewrite questions, so the saving is on input tokens.

**5. Why is the GPT-4o saving small in the simulator?**
Because GPT-4o's tokenizer already handles Kannada reasonably (1.97×) and answers dominate cost.
On GPT-4 or Llama 3 tokenizers (8.9×) rewriting removes about 75% of Kannada input tokens. We show
both rather than only the impressive one.

**6. How do you know the meaning is preserved?**
A layered verifier: numbers, dates and times, negation (including suffixes like -ಬೇಡಿ), modality,
question form, names and links, and every content word must survive; then a similarity score and
a structure check. We calibrated the threshold on FLORES dev and tested on devtest with 6,242
deliberately unsafe changes: 100% caught, 99.8% of safe rewrites accepted.

**7. Isn't 100% suspicious?**
Fair point. The unsafe changes are of the kinds our checks look for, so 100% is an upper bound for
those types, not a guarantee. A wrong synonym can pass. That is why we built the human review
interface and say plainly that no human ratings are reported yet.

**8. Why not use embeddings (LaBSE)?**
Supported and optional. We could not download the model in our build environment, so the default
is chrF on a common spelling. Our own data shows similarity alone catches only 29.4% of meaning
changes, so the critical checks matter more than the similarity model.

**9. What if the best format is always casual Latin? Why the complexity?**
It is not: native script is cheaper in 12 of 14 languages on Gemma 3 and 11 of 14 on GPT-4o, but in
0 of 14 on GPT-4, Llama 3 and Qwen3. Our per-model choice beats the best single format by 9.0
points on GPT-4o, 8.0 on Mistral NeMo and 6.2 on Gemma 3 (p < 0.001).

**10. Where does your method lose?**
On GPT-4 (−0.5 points) and Qwen3 (−0.3) the best single format is slightly ahead, because it also
sends the 1–2% of rewrites our verifier rejects. We accept that trade.

**11. Does the model still answer well in romanised text?**
We have not run a benchmark of answer quality; that is future work. We can show it live: with a
small Gemma 3 model running on the laptop (Ollama), the Optimizer page asks the AI both versions
and shows the two answers side by side, with the tokens the model itself counted. Users can keep
native script (it is always a candidate), and the router can be limited to formats a deployment
has validated.

**12. Isn't compression the obvious way to save tokens?**
We tried grammar-safe compression (dropping a pronoun only when the verb already marks the
subject). It saves at most 1.1% on formal text and 2.8% on chat. We report that as a negative
result: the money is in format and model choice.

**13. How is the Fairness Index computed, and is it arbitrary?**
Geometric mean of five parity ratios against English with equal weights (formula on the Research
page). Over 500 random weightings, the ranking keeps Kendall τ 0.953 on average, so conclusions do
not hinge on the weights.

**14. Can it run in production?**
It is a FastAPI service with an OpenAI-compatible endpoint (dry run unless an upstream is set),
35 ms per sentence for all 7 tokenizers on a CPU, about 0.6 GB memory, and every decision is
logged locally in SQLite.

**15. What about privacy?**
The audit ledger is a local file and can be turned off (`BHASHATOKEN_AUDIT=off`). Nothing is sent
anywhere unless an upstream model is configured.

**16. Which languages and tokenizers?**
17 South Asian languages plus English; GPT-4 (cl100k), GPT-4o (o200k), Llama 3, Gemma 3, Qwen3,
DeepSeek V4, Mistral NeMo. Sarvam-1 (an Indian model's tokenizer) can be added with one command
(`python scripts/get_sarvam.py`); it is optional because of its non-commercial licence. Sinhala is
not included yet.

**17. Can someone reproduce your numbers?**
Yes: one command regenerates every result; `results/ledger.json` lists dataset hashes, tokenizer
hashes, versions, hardware and the seed.

**18. What would you do with more time?**
Collect native-speaker ratings with the built protocol; add LaBSE and an NLI check; measure answer
quality per format; add Sarvam and Sinhala.

**19. How is this different from tokenizer playgrounds?**
Playgrounds such as Hugging Face's Tokenizer Playground show how one text is cut by several
tokenizers, some with a cost estimate. They do not rewrite text, check meaning, show memory lost over
a whole conversation, audit message logs, or plug into an app as an API. Our Live Demos page shows
each of these.

**20. Is the memory race exaggerated by the small memory?**
No. The memory is scaled down so it fills in seconds, but only the ratio matters: if Kannada needs
about 9× the tokens of English on Llama 3, it fits about 1/9 as much conversation in any window size.

**21. Why not just fix the tokenizer's vowel-sign problem?**
We tested exactly that: we patched the word-splitting rule so vowel signs stay inside words, keeping
the same vocabulary. Token counts changed by at most 0.2% on all 7 tokenizers, because the
vocabulary never learned whole Indian-language words. A real fix needs a new vocabulary and a
retrained model. Our text rewrite saves 11–64% on the same models today.

**22. Are the crowd ratings a proper human evaluation?**
No, and we say so on screen. They are quick ratings from people at the event, counted only if the
rater says they read the language. They show how a proper study would work; the controlled protocol
is in docs/METHODOLOGY.md.

**23. Does voice input work offline?**
No. It uses Chrome's built-in speech recognition, which needs internet. Everything else works offline.

**24. Why do you use a small local model for the answer check, not GPT-4o?**
It is free, private and runs offline on a laptop, which fits our "no paid API" rule. The same
button would work with a bigger model. Small models make mistakes in both versions, so we present
it as a demonstration, not a quality benchmark.
