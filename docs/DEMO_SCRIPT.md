# Demo script (about 4–5 minutes)

Before judges arrive: print the one-page handout (Handout button, top right), start the server, open the dashboard full-screen, press **▶ Judge demo**
once and close it (warms the caches), then go back to Overview. Keep a phone on the same Wi-Fi
with the app installed from the QR code as a backup screen.

```
PYTHONPATH=src:. python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Press **▶ Judge demo** and use **Next** (or the → key). Each step below is one click.

| Step | Screen | Say (about 25 s each) |
|---|---|---|
| 1 · The problem | Overview, KPI cards | "AI is billed per token. The same Kannada sentence needs 8.9 times the tokens of English on GPT-4 and Llama 3. That is more money, less memory, slower replies. We measured 17 languages on 7 real tokenizers with identical sentences." |
| 2 · The gap | Overview, before/after panel | "Earlier research measures this tax, or retrains tokenizers, or picks one cheaper format for a whole task. Nobody checks, per message, that a cheaper rewrite still means the same thing, and nobody keeps an audit trail. That is our project." |
| 3 · Example 1 | Optimizer → model table | "One Kannada sentence, 70 characters. The table shows tokens, characters per token, broken letters and rupees per lakh messages on all 7 models. GPT-4 needs 118 tokens; GPT-4o needs 25. After our safe rewrite every model is between 20 and 27." |
| 4 · Microscope | Optimizer → tokens | "Every box is a token. The red dashed ones are broken letters: one Kannada letter turned into several tokens. That is pure waste." |
| 5 · Example 2 | Optimizer → safe cheaper version (Kanglish) | "Real Kanglish. We generate 7 versions and check each one: the time 9:30, the English words and the 'don't' must survive. The cheapest safe version is chosen per model, and the decision record is saved for audit." |
| 6 · Compare | Compare page | "The same sentence, professionally translated into 18 languages. English needs 17 tokens; Santali needs 192 for the same meaning. The last columns show the saving after a safe rewrite for each language." |
| 7 · Memory race | Live Demos → Memory race (plays automatically) | "Same conversation, same AI memory. Watch the right side: after 40 turns on Llama 3, the English user's AI remembers all 40 messages, the Kannada user's only 9. With BhashaToken it remembers 29, more than three times as many, and every rewrite is meaning-checked." |
| 8 · Example 3, rejection | Safety Lab | "A medicine instruction: 'do not take on an empty stomach'. A rewrite drops the 'not'. It is cheaper and about 90% similar, but our negation and word checks reject it. In our test a similarity score alone caught only 29% of such changes; our verifier caught 100% while accepting 99.8% of safe rewrites." |
| 9 · Evidence | Research → Pareto | "On held-out data: 51.4% fewer tokens with 100% verified outputs. The best single format saves 49.8% but 1.4% of its outputs fail. Truncating saves 25% and breaks everything." |
| 10 · Impact | Simulator | "For a state chatbot this becomes rupees per day, month and year, and how much conversation memory a Kannada user loses. Everything is reproducible: one command, seeds and hashes on the Evidence page." |

Close: "BhashaToken works today with any hosted model, because it changes the text, not the
tokenizer, and it never sends a rewrite it cannot verify. Measure the tax. Preserve the meaning.
Make AI more inclusive."

## If Ollama is installed: answer-quality check (30 seconds, after step 5)

On the Optimizer page press **Ask the AI both versions** and say: *"Judges always ask: does the
AI still understand the cheaper version? Here a small AI running on this laptop answers both. The
model itself counted the input tokens: fewer for our version, and the two answers say the same
thing. It's a small 1-billion-parameter model, so it is a demonstration, not a benchmark."*
Read both answers yourself first; if they differ, say so honestly.

## If judges want to try things themselves (Live Demos page)

* **Fool the checker:** let a judge press any button. Safe changes should pass, meaning changes
  should be blocked; the score counts how often the checker is right.
* **Audit your messages:** paste a few lines in any language (or upload a .txt/.csv) and press
  *Audit messages*; download the CSV report.
* **Drop-in API:** press *Send test request* to show that an app only changes `base_url`.
* **Live crowd check:** at the start, put the Live Demos page on screen with the QR code and ask
  judges and nearby teams to scan it and rate a few rewrites; come back to it at the end to show
  the live results.
* **Token Tax receipt:** choose Kannada + GPT-4, show "You pay ₹699, an English user pays ₹79",
  then print or save as PDF.
* **Voice:** on the Optimizer page press 🎤 Speak and say a sentence in Kannada (needs internet).
* **Fix the tokenizer or fix the text?** Research page: patching the tokenizer saves about 0%
  without retraining; our rewrite saves 11–64% today.

## If something goes wrong

* No internet needed. If the laptop fails, use the phone (same server) or the screenshots in
  `docs/screenshots/`.
* If a judge types their own sentence: go to Optimizer, paste it, scroll to step 4.
* If a judge asks to see the raw decision: switch to **Research** mode (top right) to show JSON
  traces, per-tokenizer timings and statistics tables.
