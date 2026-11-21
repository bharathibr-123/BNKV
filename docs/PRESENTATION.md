# Presentation outline (10 slides)

Tagline on every slide footer: **BhashaToken · Measure the Tax. Preserve the Meaning. Make AI More
Inclusive.** Use real screenshots from the app; every number below comes from `RESULTS.md`.

1. **Hook.** "Same question. Same answer. Why should Kannada pay 8.9 times more?" Background: the
   Arena chart for one Kannada sentence. Team BNKV, ATME College of Engineering, Mysuru.
2. **Problem and solution (one slide).** Left: the tax (1.2× to 13.7× English across 17
   languages; less context, more cost). Right: BhashaToken rewrites each message into its cheapest
   form per model and sends it only if the meaning is verified.
3. **What exists vs what we add.** The before/after panel from the Overview page (Petrov 2023,
   Srivastava 2026, RomanSetu 2024, Lost in Compression 2026 vs per-message choice, verifier, audit).
4. **How it works (step by step).** Measure → Explain → Generate 7 rewrites → Verify → Choose per
   model → Audit.
5. **Flowchart with objectives.** The system diagram from `docs/ARCHITECTURE.md`, with three
   objectives: cut tokens, never change meaning, make every decision checkable.
6. **Fragmentation Microscope.** Screenshot of byte pieces and per-word causes.
7. **Meaning Safety Lab.** The medicine example rejected; chart: similarity alone 29.4% vs ours
   100% (99.8% safe accepted).
8. **Evidence.** Pareto chart: 51.4% fewer tokens with 100% verified vs 49.8% / 98.6% best single
   format; per-model gains up to +9.0 points (p < 0.001); honest losses on GPT-4 and Qwen3.
9. **Impact.** Simulator: ₹ per day/month/year, context window bars; runs on a CPU laptop in 35 ms.
10. **Limits and next steps; thank you.** Human ratings to collect, LaBSE/NLI, answer quality,
    Sarvam and Sinhala. QR code to the live demo.
