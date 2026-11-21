"use strict";
/* BhashaToken dashboard. Plain JavaScript + Chart.js (vendored), no build step. */

const EXAMPLES = [
  ["Kannada", "ನಾನು ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ. ಟಿಕೆಟ್ ಬೆಲೆ ಎಷ್ಟು?"],
  ["Kanglish (mixed)", "ನಾಳೆ meeting ಇದೆ, please 9:30 ಗೆ ಬನ್ನಿ. Report ಮರೆಯಬೇಡಿ."],
  ["Kannada (farmer)", "ನನ್ನ ಟೊಮೆಟೊ ಬೆಳೆಗೆ ಎಲೆ ಮುರುಟು ರೋಗ ಬಂದಿದೆ. ಯಾವ ಔಷಧಿ ಸಿಂಪಡಿಸಬೇಕು?"],
  ["Hindi", "मेरी गेहूं की फसल में पीले पत्ते आ रहे हैं। कौन सी दवा डालनी चाहिए?"],
  ["Tamil", "அரசு கல்வி உதவித்தொகைக்கு விண்ணப்பிக்க என்ன ஆவணங்கள் தேவை?"],
  ["Telugu", "నేను రేపు కాలేజీకి వెళ్తాను."],
  ["Bengali", "কাল কি কলকাতায় বৃষ্টি হবে?"],
  ["Malayalam", "നാളെ രാവിലെ 8 മണിക്ക് ബസ് ഉണ്ടോ?"],
  ["Kanglish (English letters)", "Sir nivu tumba chennagi explain madidri, nange ivattu full clear aaytu"],
  ["English", "My tomato crop has leaf curl disease. Which medicine should I spray?"],
];
const MODEL_COLORS = {
  sarvam1: "#e0457b",
  gpt4: "#8a94a8", gpt4o: "#2f6db5", llama3: "#c0412c", gemma3: "#2e8a5f",
  qwen3: "#7b4fb3", deepseek_v4: "#d08a17", mistral_nemo: "#1f8e9e",
};
const S = { meta: null, res: {}, map: null, charts: {}, tryModel: "gpt4o", style: "native", last: null, demo: null };
const CAND_NAMES = { original: "as typed", normalized: "normalized spacing / Unicode", casual_roman: "native words in English letters", iso_roman: "exact (reversible) romanisation", word_best: "word-by-word cheapest mix", compressed: "grammar-safe shorter", "compressed+word_best": "shorter + word mix" };
const cname = (n) => CAND_NAMES[n] || n;

// ---------- helpers ----------
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 0) => Number(n).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (x, d = 0) => `${fmt(x * 100, d)}%`;
const rupees = (x) => (x >= 1e7 ? `₹${fmt(x / 1e7, 2)} cr` : x >= 1e5 ? `₹${fmt(x / 1e5, 1)} lakh` : `₹${fmt(x, 0)}`);
const label = (k) => (S.meta.tokenizers.find((t) => t.key === k) || { label: k }).label;
const langName = (c) => (c === "eng_Latn" ? "English" : (S.meta.languages.find((l) => l.code === c) || { name: c }).name);
// tokenizers with saved experiment results (charts) vs every tokenizer on disk (live tools)
const keys = () => S.meta.tokenizers.filter((t) => t.measured).map((t) => t.key);
const liveKeys = () => S.meta.tokenizers.map((t) => t.key);
const color = (k) => MODEL_COLORS[k] || "#888";

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function chart(id, config) {
  if (S.charts[id]) S.charts[id].destroy();
  S.charts[id] = new Chart(document.getElementById(id), config);
  return S.charts[id];
}

function fill(sel, items, value) {
  const el = typeof sel === "string" ? $(sel) : sel;
  el.innerHTML = items.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  if (value !== undefined) el.value = value;
}

// tax colour: 1x pale -> 12x deep red (log scale)
function taxColor(p) {
  if (p === null || p === undefined) return "#e3e7ee";
  const t = Math.max(0, Math.min(1, Math.log(p) / Math.log(12)));
  return `hsl(${32 - 26 * t}, ${75 + 15 * t}%, ${88 - 50 * t}%)`;
}

function pieces(list, flags) {
  const out = [];
  list.forEach((p, i) => {
    const isByte = Boolean(flags && flags[i]);
    const last = out[out.length - 1];
    if (isByte && last && last.byte && last.text === p) { last.n += 1; return; }
    out.push({ text: p, byte: isByte, n: 1 });
  });
  return `<div class="pieces">${out.map((o, i) => (o.byte
    ? `<span class="pc byte" title="${o.n} byte tokens">${esc(o.text)}<sub>×${o.n}</sub></span>`
    : `<span class="pc c${i % 4}">${esc(o.text)}</span>`)).join("")}</div>`;
}

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
function themeCharts() {
  const cs = getComputedStyle(document.documentElement);
  Chart.defaults.color = cs.getPropertyValue("--muted").trim() || "#56607a";
  Chart.defaults.borderColor = cs.getPropertyValue("--line").trim() || "#dfe4ee";
}
themeCharts();
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;

// ---------- data access ----------
const premium = (model, lang, style = "native") => {
  const c = S.res.benchmark.flores.find((x) => x.tokenizer === model && x.language === lang && x.style === style);
  return c ? c.premium : null;
};

// ================= OVERVIEW =================
function renderKpis() {
  const native = S.res.benchmark.flores.filter((x) => x.style === "native" && x.language !== "eng_Latn");
  const worstKan = Math.max(...native.filter((x) => x.language === "kan_Knda").map((x) => x.premium));
  const opt = S.res.optimizer.rows.find((r) => r.tokenizer === "llama3" && r.language === "kan_Knda");
  const sim = S.res.simulator.rows.find((r) => r.model === "gpt4o" && r.language === "kan_Knda");
  $("#kpis").innerHTML = `
    <div class="kpi"><b>${fmt(worstKan, 1)}×</b><span>more tokens for the same Kannada sentence than English (Llama 3, GPT-4)</span></div>
    <div class="kpi safe"><b>${pct(S.res.verification.summary.unsafe_caught, 1)}</b><span>of deliberately unsafe rewrites caught by our verifier (similarity alone: ${pct(S.res.verification.summary.similarity_only.unsafe_caught, 0)})</span></div>
    <div class="kpi save"><b>${pct(opt.saving_vs_native)}</b><span>fewer Kannada tokens on Llama 3 after verified rewriting (FLORES, ${S.res.optimizer.sentences_per_language} sentences)</span></div>
    <div class="kpi"><b>${rupees(sim.extra_annual_inr)}</b><span>simulated extra cost per year for one Kannada government chatbot on GPT-4o (1 lakh users/day)</span></div>`;
}

function project(states) {
  let minX = 180, maxX = -180, minY = 90, maxY = -90;
  states.forEach((s) => s.rings.forEach((r) => r.forEach(([x, y]) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  })));
  const k = Math.cos(((minY + maxY) / 2) * Math.PI / 180);
  const W = 1000, scale = W / ((maxX - minX) * k), H = (maxY - minY) * scale;
  const pt = ([x, y]) => `${((x - minX) * k * scale).toFixed(1)},${((maxY - y) * scale).toFixed(1)}`;
  return { W, H, path: (rings) => rings.map((r) => `M${r.map(pt).join("L")}Z`).join("") };
}

function renderMap() {
  const model = $("#map-model").value;
  const { W, H, path } = project(S.map.states);
  $("#map").innerHTML = `<svg viewBox="0 0 ${W} ${H.toFixed(0)}" role="img" aria-label="Map of India coloured by token tax">
    ${S.map.states.map((s, i) => {
      const p = s.language ? premium(model, s.language) : null;
      return `<path data-i="${i}" d="${path(s.rings)}" fill="${taxColor(p)}"></path>`;
    }).join("")}</svg>`;
  const tip = $("#tip");
  const show = (ev, s) => {
    const p = s.language ? premium(model, s.language) : null;
    tip.innerHTML = `<b>${esc(s.name)}</b><br>${s.language ? `${esc(langName(s.language))}: <b>${fmt(p, 2)}×</b> English on ${esc(label(model))}` : "Main language not in our study"}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(ev.clientX + 12, window.innerWidth - 250)}px`;
    tip.style.top = `${ev.clientY + 12}px`;
  };
  $("#map").querySelectorAll("path").forEach((el) => {
    const s = S.map.states[Number(el.dataset.i)];
    el.addEventListener("mousemove", (ev) => show(ev, s));
    el.addEventListener("mouseleave", () => { tip.hidden = true; });
    el.addEventListener("click", (ev) => {
      $("#map").querySelectorAll("path.sel").forEach((x) => x.classList.remove("sel"));
      el.classList.add("sel"); show(ev, s);
      clearTimeout(S.tipTimer); S.tipTimer = setTimeout(() => { tip.hidden = true; }, 3000);
      if (s.language) { $("#bar-lang").value = s.language; renderModelBars(); renderCauses(); }
    });
  });
  $("#map-legend").innerHTML = [1, 2, 4, 8, 12].map((v) => `<i style="background:${taxColor(v)}"></i>${v}×`).join(" ") +
    ` <i style="background:#e3e7ee"></i>not studied`;
}

function renderModelBars() {
  const lang = $("#bar-lang").value;
  const rows = keys().map((k) => [k, premium(k, lang)]).sort((a, b) => b[1] - a[1]);
  chart("chart-models", {
    type: "bar",
    data: {
      labels: ["English (reference)", ...rows.map(([k]) => label(k))],
      datasets: [{ data: [1, ...rows.map(([, v]) => v)], backgroundColor: ["#f39a2b", ...rows.map(([k]) => color(k))], borderRadius: 4 }],
    },
    options: {
      indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 2)}× English` } } },
      scales: { x: { beginAtZero: true, title: { display: true, text: `${langName(lang)}: tokens compared with English` } } },
    },
  });
  $("#donut-lang-name").textContent = langName(lang);
}

function renderCauses() {
  const model = $("#donut-model").value;
  const lang = $("#bar-lang").value;
  const r = S.res.explainer.rows.find((x) => x.tokenizer === model && x.language === lang);
  const box = $("#chart-causes").parentElement;
  if (!r) { if (S.charts["chart-causes"]) S.charts["chart-causes"].destroy(); box.dataset.empty = "1"; return; }
  chart("chart-causes", {
    type: "doughnut",
    data: {
      labels: ["1 token per word (unavoidable)", "Vowel signs split off", "Missing vocabulary"],
      datasets: [{ data: [r.base, r.vowel_split, r.vocab_gap], backgroundColor: ["#9fb3d1", "#c0412c", "#f39a2b"], borderWidth: 2 }],
    },
    options: {
      cutout: "58%", plugins: {
        legend: { position: "right" },
        tooltip: { callbacks: { label: (c) => `${c.label}: ${pct(c.raw / (r.base + r.vowel_split + r.vocab_gap))}` } },
        title: { display: true, text: `${label(model)} · ${pct(r.byte_share)} of tokens are raw byte pieces` },
      },
    },
  });
}

function renderFairBars() {
  const by = {};
  S.res.fairness.scores.forEach((s) => { (by[s.tokenizer] = by[s.tokenizer] || []).push(s.index); });
  const rows = Object.entries(by).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length]).sort((a, b) => b[1] - a[1]);
  chart("chart-fair", {
    type: "bar",
    data: { labels: rows.map(([k]) => label(k)), datasets: [{ data: rows.map(([, v]) => v), backgroundColor: rows.map(([k]) => color(k)), borderRadius: 4 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 1, title: { display: true, text: "Fairness Index (1.00 = like English)" } } } },
  });
}

function initOverview() {
  const toks = keys().map((k) => [k, label(k)]);
  fill("#map-model", toks, "llama3");
  fill("#donut-model", toks, "llama3");
  fill("#bar-lang", S.meta.languages.map((l) => [l.code, l.name]), "kan_Knda");
  $("#map-model").onchange = renderMap;
  $("#bar-lang").onchange = () => { renderModelBars(); renderCauses(); };
  $("#donut-model").onchange = renderCauses;
  renderKpis(); renderMap(); initOverviewCharts();
}
function initOverviewCharts() { renderModelBars(); renderCauses(); renderFairBars(); }

// ================= TRY IT =================
let tryTimer = null;
function scheduleTry() { clearTimeout(tryTimer); tryTimer = setTimeout(runTry, 400); }

async function runTry() {
  const text = $("#txt").value.trim();
  if (!text) return;
  const t0 = Date.now();
  const tick = async () => {
    if (S.last && S.last.text === text) return;
    const secs = Math.round((Date.now() - t0) / 1000);
    let ready = true;
    try { ready = (await api("/api/health")).ready; } catch (e) { ready = true; }
    $("#try-summary").innerHTML = `<div style="grid-column: span 3"><b>Measuring… ${secs} s</b><span>${ready
      ? "counting tokens on 7 models and checking 7 versions"
      : "first start: loading 7 AI tokenizers and spelling data (usually 20–60 seconds on a laptop, only once)"}</span></div>`;
  };
  if (!S.last) { tick(); }
  const timer = setInterval(() => { if (!S.last || S.last.text !== text) tick(); }, 2000);
  try {
    const [an, opt, pred, comp] = await Promise.all([
      api("/api/analyze", { text }), api("/api/optimize", { text }),
      api("/api/predict", { text }), api("/api/compress", { text, aggressive: $("#aggr").checked }),
    ]);
    clearInterval(timer);
    if ($("#txt").value.trim() !== text) return;  // a newer request is on its way
    S.last = { text, an, opt, pred, comp };
    renderTryCost(); renderTokens(); renderOpt(); renderPred(); renderComp();
    await renderWhy();
  } catch (e) {
    clearInterval(timer);
    $("#try-summary").innerHTML = `<div class="err" style="grid-column: span 5">Could not measure this sentence: ${esc(e.message)}. Check that the terminal running the server shows no error, then press Ctrl + Shift + R.</div>`;
  }
}

function renderTryCost() {
  const { an, opt, text } = S.last;
  const price = S.meta.input_price_usd, inr = S.meta.usd_to_inr;
  const lakh = (k, t) => (t * price[k] * inr * 100000) / 1e6;
  const rows = an.rows.map((r) => {
    const ch = opt.choices.find((c) => c.tokenizer === r.tokenizer);
    return { ...r, ch, before: lakh(r.tokenizer, r.tokens), after: lakh(r.tokenizer, ch.tokens),
      bytes: r.byte_flags.filter(Boolean).length };
  }).sort((a, b) => a.tokens - b.tokens);
  const maxTok = Math.max(...rows.map((r) => r.tokens), 1);
  const minTok = Math.min(...rows.map((r) => r.tokens));
  const bestAfter = rows.reduce((m, r) => (r.ch.tokens < m.ch.tokens ? r : m), rows[0]);
  const bestSave = rows.reduce((m, r) => (r.ch.saving > m.ch.saving ? r : m), rows[0]);
  const chars = [...text].length;
  $("#try-summary").innerHTML = `
    <div><b>${esc(langName(an.language))}</b><span>language detected (${esc(an.style)} style)</span></div>
    <div><b>${fmt(chars)} · ${fmt(an.words)}</b><span>characters · words</span></div>
    <div><b>${minTok} → ${maxTok}</b><span>tokens, cheapest → costliest model</span></div>
    <div><b>${esc(label(bestAfter.tokenizer))}: ${bestAfter.ch.tokens}</b><span>fewest tokens after safe rewrite</span></div>
    <div class="hl"><b>${pct(bestSave.ch.saving)}</b><span>biggest safe saving (${esc(label(bestSave.tokenizer))})</span></div>`;
  $("#model-table").innerHTML = `<table class="big"><tr><th>Model</th><th>Tokens (as typed)</th><th>Characters</th><th>Chars / token</th><th>Broken-letter pieces</th><th>₹ per 1 lakh msgs</th><th>After safe rewrite</th><th>₹ after</th><th>Saved</th><th title="Same vocabulary, pre-tokenizer fixed to keep vowel signs inside words">If tokenizer fixed*</th></tr>
    ${rows.map((r) => `<tr class="pick ${r.tokenizer === S.tryModel ? "sel" : ""}" data-k="${r.tokenizer}">
      <td class="model"><span class="dot" style="background:${color(r.tokenizer)}"></span>${esc(r.label)}${r.tokens === minTok ? ' <span class="best-tag">cheapest</span>' : ""}</td>
      <td class="bar-cell"><i style="width:${(r.tokens / maxTok) * 100}%"></i><span>${r.tokens}</span></td>
      <td class="num">${fmt(chars)}</td><td class="num">${fmt(chars / Math.max(r.tokens, 1), 1)}</td>
      <td class="num">${r.bytes ? `<span class="pill bad">${r.bytes}</span>` : '<span class="pill good">0</span>'}</td>
      <td class="num">₹${fmt(r.before, 0)}</td>
      <td><b>${r.ch.tokens} tokens</b><div class="note">${esc(cname(r.ch.best))}</div></td><td class="num">₹${fmt(r.after, 0)}</td>
      <td class="down">${r.ch.saving > 0 ? `▼ ${pct(r.ch.saving)}` : "–"}</td>
      <td class="num note">${r.tokens_fixed_pretokenizer} tokens</td></tr>`).join("")}</table>
    <p class="note">* What if the tokenizer stopped cutting words at vowel signs, without retraining its vocabulary? Almost no change: the vocabulary never learned whole words, so fixing the tokenizer needs a new model. Rewriting the text works today.</p>`;
  $("#model-table").querySelectorAll("tr[data-k]").forEach((tr) => {
    tr.onclick = () => { selectModel(tr.dataset.k); document.querySelector("#s-tokens").scrollIntoView({ behavior: "smooth" }); };
  });
}

function selectModel(k) {
  S.tryModel = k;
  document.querySelectorAll("#model-table tr[data-k]").forEach((tr) => tr.classList.toggle("sel", tr.dataset.k === k));
  renderTokens(); renderWhy(); renderOptBest(); renderComp();
}

function renderTokens() {
  const box = $("#tok-chips");
  box.innerHTML = liveKeys().map((k) => `<button data-k="${k}" class="${k === S.tryModel ? "on" : ""}">${esc(label(k))}</button>`).join("");
  box.querySelectorAll("button").forEach((b) => {
    b.onclick = () => selectModel(b.dataset.k);
  });
  const row = S.last.an.rows.find((r) => r.tokenizer === S.tryModel);
  $("#tok-model-name").textContent = row.label;
  const broken = row.byte_flags.filter(Boolean).length;
  $("#tok-view").innerHTML = `<p class="note"><b>${row.tokens} tokens</b> for ${fmt([...S.last.text].length)} characters on ${esc(row.label)} · ${broken} broken-letter piece(s)</p>${pieces(row.pieces, row.byte_flags)}`;
}

async function renderWhy() {
  if (!S.last) return;
  const d = await api("/api/explain", { text: S.last.text, tokenizer: S.tryModel });
  const t = d.totals;
  chart("chart-why", {
    type: "doughnut",
    data: { labels: ["1 per word", "Vowel signs split off", "Missing vocabulary"],
      datasets: [{ data: [t.base, t.vowel_split, t.vocab_gap], backgroundColor: ["#9fb3d1", "#c0412c", "#f39a2b"] }] },
    options: { cutout: "55%", plugins: { legend: { position: "bottom" }, title: { display: true, text: `${t.tokens} tokens on ${label(S.tryModel)}` } } },
  });
  const words = [...d.words].sort((a, b) => b.tokens - a.tokens).slice(0, 8);
  const causeText = { efficient: "efficient", vowel_split: "vowel signs split", missing_script: "no letters for this script", vocab_gap: "missing vocabulary" };
  $("#why-words").innerHTML = words.map((w) => `<div class="word"><b>${esc(w.word.trim())}</b>
    <span class="tag ${w.main_cause === "efficient" ? "good" : "bad"}">${esc(causeText[w.main_cause] || w.main_cause)}</span>
    <div class="note">${w.tokens} tokens · meaningful parts: ${w.morphemes.map((m) => esc(m.trim())).join(" + ")}</div>
    ${pieces(w.pieces, w.byte_flags)}</div>`).join("");
}

function renderOpt() {
  const { opt } = S.last;
  const orig = opt.candidates.find((c) => c.name === "original");
  chart("chart-opt", {
    type: "bar",
    data: {
      labels: opt.choices.map((c) => label(c.tokenizer)),
      datasets: [
        { label: "As typed", data: opt.choices.map((c) => orig.tokens[c.tokenizer]), backgroundColor: "#9fb3d1", borderRadius: 4 },
        { label: "After safe rewrite", data: opt.choices.map((c) => c.tokens), backgroundColor: "#ef8f1f", borderRadius: 4 },
      ],
    },
    options: { plugins: { tooltip: { callbacks: { afterBody: (items) => { const c = opt.choices[items[0].dataIndex]; return `${cname(c.best)} · saving ${pct(c.saving)}`; } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "tokens" } } } },
  });
  renderOptBest();
  renderTrace();
}

function renderOptBest() {
  if (!S.last) return;
  const { opt } = S.last;
  const choice = opt.choices.find((c) => c.tokenizer === S.tryModel) || opt.choices[0];
  const cand = opt.candidates.find((c) => c.name === choice.best);
  $("#opt-best").innerHTML = `<div class="best-card"><span class="note">Chosen for <b>${esc(label(choice.tokenizer))}</b>: <b>${esc(cname(choice.best))}</b> · ${choice.original_tokens} → ${choice.tokens} tokens (${pct(choice.saving)} fewer) · <span class="pill good">all checks passed</span></span>
    <div class="txt">${esc(choice.text || cand.text)}</div><div class="note">${esc(choice.why)}</div></div>`;
  $("#cand-model").textContent = `(token counts for ${label(choice.tokenizer)})`;
  const rows = opt.candidates.map((c) => {
    const text = (c.per_model_text && c.per_model_text[choice.tokenizer]) || c.text;
    const verdict = c.safe ? '<span class="pill good">safe</span>' : '<span class="pill bad">rejected</span>';
    const reasons = c.reasons.length ? `<ul class="reasons">${c.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";
    return `<tr class="${c.name === choice.best ? "chosen" : ""}"><td><b>${esc(cname(c.name))}</b><div class="note">${esc(c.transforms.join(" + ") || "none")}</div></td>
      <td class="txt">${esc(text)}</td><td class="num">${c.tokens[choice.tokenizer]}</td>
      <td>${verdict}<div class="note">${esc(c.meaning_check)} ${c.meaning_check.startsWith("chrF") || c.meaning_check.startsWith("embedding") ? fmt(c.meaning_score, 3) : ""}</div>${reasons}</td>
      <td class="num rmode">${fmt(c.build_ms, 1)} ms</td></tr>`;
  }).join("");
  $("#cand-table").innerHTML = `<table><tr><th>Candidate</th><th>Text sent</th><th>Tokens</th><th>Meaning check</th><th class="rmode">Build time</th></tr>${rows}</table>`;
}

function renderTrace() {
  const t = S.last.opt.trace;
  const rejected = Object.entries(t.rejected || {});
  $("#trace").innerHTML = `<dl class="trace-grid">
    <dt>Request ID</dt><dd><code>${esc(t.request_id)}</code> ${t.audited ? '<span class="pill good">saved to audit ledger</span>' : '<span class="pill">not stored</span>'}</dd>
    <dt>Time</dt><dd>${esc(t.timestamp)}</dd>
    <dt>System</dt><dd>${esc(t.system_version)}</dd>
    <dt>Language</dt><dd>${esc(langName(t.language_detected))} ${t.language_given ? "(given)" : "(detected)"}</dd>
    <dt>Similarity</dt><dd>${esc(t.similarity_backend)}</dd>
    <dt>Spelling ambiguity</dt><dd>${pct(t.spelling_ambiguity, 0)} of words (limit 25% for casual Latin)</dd>
    <dt>Candidates</dt><dd>${t.candidates_tried.map(cname).map(esc).join(" · ")}</dd>
    <dt>Rejected</dt><dd>${rejected.length ? rejected.map(([n, r]) => `<b>${esc(cname(n))}</b>: ${esc(r[0])}`).join("<br>") : "none"}</dd>
    <dt>Latency</dt><dd>${fmt(t.latency_ms, 1)} ms for all ${Object.keys(t.tokenizers).length} tokenizers, CPU</dd>
  </dl><pre class="json rmode">${esc(JSON.stringify(t, null, 1))}</pre>`;
}

async function sendReview() {
  if (!S.last) return;
  const choice = S.last.opt.choices.find((c) => c.tokenizer === S.tryModel) || S.last.opt.choices[0];
  try {
    const d = await api("/api/review", {
      request_id: S.last.opt.trace.request_id, candidate: choice.best, reviewer: $("#rv-name").value || "anonymous",
      meaning: Number($("#rv-meaning").value), fluency: Number($("#rv-fluency").value), acceptable: $("#rv-ok").value === "1",
    });
    $("#rv-status").textContent = `Saved (review #${d.id}). ${d.summary.total} review(s) so far.`;
    renderReviewSummary(d.summary);
  } catch (e) { $("#rv-status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

function renderPred() {
  const rows = S.last.pred.rows;
  chart("chart-pred", {
    type: "bar",
    data: {
      labels: rows.map((r) => label(r.tokenizer)),
      datasets: [
        { label: "Real count", data: rows.map((r) => r.actual), backgroundColor: "#1e2a44", borderRadius: 3 },
        { label: "Our prediction", data: rows.map((r) => r.predicted), backgroundColor: "#f39a2b", borderRadius: 3 },
        { label: "characters ÷ 4", data: rows.map((r) => r.chars_over_4), backgroundColor: "#c9d2e3", borderRadius: 3 },
      ],
    },
    options: { scales: { y: { beginAtZero: true, title: { display: true, text: "tokens" } } } },
  });
}

function renderComp() {
  if (!S.last) return;
  const d = S.last.comp;
  const k = S.tryModel in d.saving ? S.tryModel : Object.keys(d.saving)[0];
  if (!d.edits.length) {
    const why = d.within_budget
      ? "Every word in this sentence carries meaning: there is no pronoun like ನಾನು / ನೀವು that the verb already shows, and no extra spaces. Removing anything would change the meaning, so the safe answer is to change nothing."
      : "Removing words would have cut more than the safe limit, so the original was kept.";
    $("#comp-out").innerHTML = `<div class="best-card"><b>✓ No change needed, and that is the correct result.</b>
      <div class="note" style="margin-top:4px">${why} The real saving for this sentence comes from step 3 (writing it in the cheapest safe format), not from deleting words.</div>
      <p style="margin:8px 0 0"><button class="solid" type="button" id="comp-try">Try a sentence with a removable word</button></p></div>`;
    $("#comp-try").onclick = () => {
      $("#txt").value = "ನಾನು ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ  ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ.";
      document.querySelectorAll("#examples button").forEach((x) => x.classList.remove("on"));
      runTry().then(() => document.querySelector("#s-more").scrollIntoView({ behavior: "smooth" }));
    };
    return;
  }
  const edits = `<ul class="edits">${d.edits.map((e) => `<li><b>${esc(e.rule.replace("_", " "))}</b>: ${e.removed ? `removed “${esc(e.removed)}”` : "cleaned extra spaces"}${e.added ? ` → “${esc(e.added)}”` : ""} <span class="note">(${esc(e.reason)})</span></li>`).join("")}</ul>`;
  $("#comp-out").innerHTML = `<div class="compare"><div><small>Before</small>${esc(d.original)}</div><div><small>After</small><b>${esc(d.text)}</b></div></div>
    ${edits}<p class="note">${esc(label(k))}: ${d.tokens_before[k]} → ${d.tokens_after[k]} tokens (${pct(d.saving[k], 1)} saved). Every removal is justified by a grammar rule and passes the meaning check.</p>`;
}

function initTry() {
  const box = $("#examples");
  EXAMPLES.forEach(([name, text], i) => {
    const b = document.createElement("button");
    b.textContent = name;
    if (i === 0) b.classList.add("on");
    b.onclick = () => {
      $("#txt").value = text;
      box.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on"); runTry();
    };
    box.appendChild(b);
  });
  $("#txt").value = EXAMPLES[0][1];
  $("#txt").addEventListener("input", scheduleTry);
  $("#aggr").onchange = runTry;
  $("#rv-send").onclick = sendReview;
}

// ================= LANGUAGES =================
function renderHeatmap() {
  const ks = keys();
  const langs = S.meta.languages.filter((l) => ks.some((k) => premium(k, l.code, S.style) !== null));
  $("#heatmap").innerHTML = `<table><tr><th>Language</th>${ks.map((k) => `<th>${esc(label(k))}</th>`).join("")}</tr>
    ${langs.map((l) => `<tr><td>${esc(l.name)} <span class="note">${esc(l.script)}</span></td>${ks.map((k) => {
      const p = premium(k, l.code, S.style);
      const dark = p && Math.log(p) / Math.log(12) > 0.55;
      return `<td class="cell" style="background:${taxColor(p)};color:${dark ? "#fff" : "#1e2a44"}">${p ? fmt(p, 2) + "×" : "–"}</td>`;
    }).join("")}</tr>`).join("")}</table>`;
}

function renderStyleChart() {
  const k = $("#style-model").value;
  const langs = S.meta.languages.filter((l) => premium(k, l.code, "roman") !== null);
  chart("chart-style", {
    type: "bar",
    data: {
      labels: langs.map((l) => l.name),
      datasets: [
        { label: "Native script", data: langs.map((l) => premium(k, l.code, "native")), backgroundColor: "#1e2a44", borderRadius: 3 },
        { label: "English letters", data: langs.map((l) => premium(k, l.code, "roman")), backgroundColor: "#f39a2b", borderRadius: 3 },
      ],
    },
    options: { indexAxis: "y", scales: { x: { beginAtZero: true, title: { display: true, text: "tokens compared with English" } } } },
  });
}

function renderRadar() {
  const lang = $("#radar-lang").value;
  const parts = S.res.fairness.parts;
  const names = { cost_parity: "Cost", typing_parity: "Typing in English letters", word_integrity: "Whole words kept", char_integrity: "Whole letters kept", pretoken_integrity: "Vowel signs kept" };
  const pick = ["gpt4o", "llama3", "gemma3", "deepseek_v4"].filter((k) => keys().includes(k));
  chart("chart-radar", {
    type: "radar",
    data: {
      labels: parts.map((p) => names[p] || p),
      datasets: pick.map((k) => {
        const s = S.res.fairness.scores.find((x) => x.tokenizer === k && x.language === lang);
        return { label: `${label(k)} (index ${fmt(s.index, 2)})`, data: parts.map((p) => s[p] ?? 0),
          borderColor: color(k), backgroundColor: `${color(k)}22`, pointRadius: 3 };
      }),
    },
    options: { scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.25, backdropColor: "transparent" } } }, plugins: { legend: { position: "bottom" } } },
  });
}

function renderCodemix() {
  const rows = S.res.benchmark.codemix.filter((r) => r.style !== "roman");
  const groups = [...new Set(rows.map((r) => `${r.language}|${r.style}`))];
  const pick = ["gpt4o", "llama3", "gemma3", "mistral_nemo"].filter((k) => keys().includes(k));
  chart("chart-codemix", {
    type: "bar",
    data: {
      labels: groups.map((g) => { const [l, s] = g.split("|"); return `${langName(l)} ${s === "native" ? "script" : "mixed"}`; }),
      datasets: pick.map((k) => ({ label: label(k), backgroundColor: color(k), borderRadius: 3,
        data: groups.map((g) => { const [l, s] = g.split("|"); const r = rows.find((x) => x.tokenizer === k && x.language === l && x.style === s); return r ? r.latin_saving * 100 : 0; }) })),
    },
    options: { scales: { y: { title: { display: true, text: "% tokens saved by English letters" } } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw, 0)}%` } } } },
  });
}

function initLanguages() {
  fill("#style-model", keys().map((k) => [k, label(k)]), "gpt4o");
  fill("#radar-lang", S.meta.languages.map((l) => [l.code, l.name]), "kan_Knda");
  $("#style-model").onchange = renderStyleChart;
  $("#radar-lang").onchange = renderRadar;
  document.querySelectorAll(".seg button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".seg button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on"); S.style = b.dataset.style; renderHeatmap();
    };
  });
  languageCharts();
}
function languageCharts() { renderHeatmap(); renderStyleChart(); renderRadar(); renderCodemix(); }

// ================= COMPARE LANGUAGES =================
S.cmpIndex = 123;
let cmpTimer = null;

async function runCompare(index) {
  if (index !== undefined) S.cmpIndex = index;
  const model = $("#cmp-model").value;
  $("#cmp-head").innerHTML = '<p class="note">Counting tokens in 18 languages…</p>';
  try {
    const d = await api("/api/compare", { index: S.cmpIndex, tokenizer: model });
    renderCompare(d);
  } catch (e) { $("#cmp-head").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

function renderCompare(d) {
  document.querySelectorAll("#cmp-featured button").forEach((b) => b.classList.toggle("on", Number(b.dataset.i) === d.index));
  const rows = [...d.rows].sort((a, b) => a.tokens - b.tokens);
  const max = Math.max(...rows.map((r) => r.tokens), 1);
  const worst = rows[rows.length - 1];
  $("#cmp-head").innerHTML = `<div class="cmp-sentence"><span class="note">Sentence #${d.index + 1} of 1,012 · ${esc(d.label)} · input price $${fmt(d.price_input_usd_per_million, 2)} per 1M tokens</span><br><b>${esc(d.rows[0].text)}</b>
    <div class="note">English needs <b>${d.english_tokens}</b> tokens. ${esc(worst.language)} needs <b>${worst.tokens}</b> (${fmt(worst.x_english, 1)}× English) for the same meaning.</div></div>`;
  $("#cmp-table").innerHTML = `<table class="big"><tr><th>Language</th><th>Same sentence</th><th>Chars</th><th>Tokens</th><th>× English</th><th>Broken pieces</th><th>₹ / 1 lakh msgs</th><th>After safe rewrite</th><th>Saved</th></tr>
    ${rows.map((r) => `<tr class="${r.code === "eng_Latn" ? "sel" : ""}">
      <td class="model">${esc(r.language)}<div class="note">${esc(r.script)}</div></td>
      <td class="native">${esc(r.text)}${r.best !== "original" ? `<small>sent as (${esc(cname(r.best))}): ${esc(r.best_text)}</small>` : ""}</td>
      <td class="num">${fmt(r.characters)}</td>
      <td class="bar-cell"><i style="width:${(r.tokens / max) * 100}%;background:${taxColor(r.x_english)}"></i><span>${r.tokens}</span></td>
      <td class="num"><b>${fmt(r.x_english, 2)}×</b></td>
      <td class="num">${r.byte_pieces ? `<span class="pill bad">${r.byte_pieces}</span>` : '<span class="pill good">0</span>'}</td>
      <td class="num">₹${fmt(r.cost_inr_per_lakh, 0)}</td>
      <td class="num"><b>${r.best_tokens} tokens</b><div class="note">₹${fmt(r.best_cost_inr_per_lakh, 0)}</div></td>
      <td class="down">${r.saving > 0 ? `▼ ${pct(r.saving)}` : "–"}</td></tr>`).join("")}</table>`;
  chart("chart-cmp", {
    type: "bar",
    data: { labels: rows.map((r) => r.language), datasets: [
      { label: "As written", data: rows.map((r) => r.tokens), backgroundColor: rows.map((r) => (r.code === "eng_Latn" ? "#ef8f1f" : "#9fb3d1")), borderRadius: 3 },
      { label: "After safe rewrite", data: rows.map((r) => r.best_tokens), backgroundColor: "#23845a", borderRadius: 3 },
    ] },
    options: { indexAxis: "y", plugins: { tooltip: { callbacks: { afterLabel: (c) => `${fmt(rows[c.dataIndex].x_english, 2)}× English · ${rows[c.dataIndex].characters} characters` } } },
      scales: { x: { beginAtZero: true, title: { display: true, text: `tokens on ${d.label}` } } } },
  });
}

async function searchCompare() {
  const q = $("#cmp-q").value.trim();
  if (q.length < 2) { $("#cmp-results").innerHTML = ""; return; }
  const d = await api(`/api/compare/search?q=${encodeURIComponent(q)}`);
  $("#cmp-results").innerHTML = d.results.length
    ? d.results.map((r) => `<button data-i="${r.index}">#${r.index + 1} · ${esc(r.english)} <span class="note">(matched in ${esc(langName(r.matched_in))})</span></button>`).join("")
    : '<p class="note">No sentence found. Try another word.</p>';
  $("#cmp-results").querySelectorAll("button").forEach((b) => { b.onclick = () => { $("#cmp-results").innerHTML = ""; runCompare(Number(b.dataset.i)); }; });
}

async function compareFromOptimizer() {
  const text = $("#txt").value.trim();
  showPage("languages");
  try {
    const d = await api("/api/compare", { text, tokenizer: $("#cmp-model").value });
    S.cmpIndex = d.index; renderCompare(d);
  } catch (e) {
    $("#cmp-q").value = text.slice(0, 80);
    await searchCompare();
    $("#cmp-head").innerHTML = `<p class="note">Your exact sentence is not in the 1,012 professionally translated sentences, so here are the closest ones. Pick one to see it in all 18 languages.</p>`;
  }
}

async function initCompare() {
  fill("#cmp-model", S.meta.tokenizers.map((t) => [t.key, t.label]), "gpt4o");
  $("#cmp-model").onchange = () => runCompare();
  const f = await api("/api/compare/featured");
  $("#cmp-featured").innerHTML = f.featured.map((x) => `<button data-i="${x.index}" title="${esc(x.english)}">${esc(x.title)}</button>`).join("");
  $("#cmp-featured").querySelectorAll("button").forEach((b) => { b.onclick = () => runCompare(Number(b.dataset.i)); });
  $("#cmp-q").addEventListener("input", () => { clearTimeout(cmpTimer); cmpTimer = setTimeout(searchCompare, 350); });
  $("#to-compare").onclick = compareFromOptimizer;
}

// ================= LIVE DEMOS =================
const MEM = { data: null, step: 0, timer: null };

function memLaneRender(id, msgs, windowSize, t) {
  // keep the newest messages that fit in the window
  let used = 0, kept = 0;
  const keep = new Set();
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (used + msgs[i].tokens > windowSize) break;
    used += msgs[i].tokens; kept++; keep.add(i);
  }
  const lane = $(`#${id}`);
  const total = msgs.reduce((a, m) => a + m.tokens, 0);
  lane.querySelector(".meter i").style.width = `${Math.min(100, (total / windowSize) * 100)}%`;
  lane.querySelector(".meter").classList.toggle("full", total > windowSize);
  lane.querySelector(".lane-stat").textContent = `remembers ${kept} of ${t} · ${fmt(Math.min(total, windowSize))}/${fmt(windowSize)} tokens`;
  const start = Math.max(0, msgs.length - 9);
  lane.querySelector(".bubbles").innerHTML = msgs.slice(start).map((m, j) => {
    const i = start + j;
    const txt = m.text.length > 95 ? `${m.text.slice(0, 95)}…` : m.text;
    return `<div class="bubble ${keep.has(i) ? "" : "gone"}"><small>turn ${i + 1} · ${m.tokens} tokens${keep.has(i) ? "" : " · forgotten"}</small>${esc(txt)}</div>`;
  }).join("");
  return kept;
}

function memStep() {
  const d = MEM.data; const w = Number($("#mem-window").value);
  MEM.step++;
  const t = MEM.step, turns = d.turns.slice(0, t);
  const ke = memLaneRender("lane-en", turns.map((x) => ({ text: x.english, tokens: x.tokens_english })), w, t);
  const kn = memLaneRender("lane-nat", turns.map((x) => ({ text: x.native, tokens: x.tokens_native })), w, t);
  const ko = memLaneRender("lane-opt", turns.map((x) => ({ text: x.optimized, tokens: x.tokens_optimized })), w, t);
  if (t >= d.turns.length) {
    clearInterval(MEM.timer); MEM.timer = null; $("#mem-play").textContent = "↻ Replay";
    const name = langName(d.language);
    $("#mem-result").innerHTML = `
      <div><b>${ke}</b>messages an English user's AI still remembers</div>
      <div><b>${kn}</b>messages a ${esc(name)} user's AI remembers, as typed</div>
      <div style="background:var(--good-bg)"><b>${ko}</b>with BhashaToken: ${kn ? `${fmt(ko / kn, 1)}× more memory` : "much more memory"}, every rewrite meaning-checked</div>`;
  }
}

async function memPlay() {
  if (MEM.timer) { clearInterval(MEM.timer); MEM.timer = null; $("#mem-play").textContent = "▶ Play"; return; }
  $("#mem-result").innerHTML = "";
  if (!MEM.data || MEM.step >= MEM.data.turns.length || MEM.key !== `${$("#mem-lang").value}|${$("#mem-model").value}`) {
    $("#mem-play").textContent = "Loading…";
    MEM.data = await api("/api/demo/conversation", { language: $("#mem-lang").value, tokenizer: $("#mem-model").value, turns: 40 });
    MEM.key = `${$("#mem-lang").value}|${$("#mem-model").value}`;
    MEM.step = 0;
  }
  $("#lane-nat-name").textContent = `${langName(MEM.data.language)} user`;
  $("#lane-opt-name").textContent = `${langName(MEM.data.language)} + BhashaToken`;
  $("#mem-play").textContent = "❚❚ Pause";
  memStep();
  MEM.timer = setInterval(memStep, 550);
}

const FOOL_PRESETS = [
  ["Medicine", "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ."],
  ["Appointment", "ನಾನು ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಆಸ್ಪತ್ರೆಗೆ ಬರುವುದಿಲ್ಲ."],
  ["Hindi notice", "कल स्कूल 8 बजे खुलेगा, देर से मत आना।"],
  ["Tamil request", "நாளை காலை 9 மணிக்கு அலுவலகம் வர வேண்டும்."],
];
const FOOL = { tries: 0, right: 0, caught: 0, unsafe: 0, accepted: 0, safe: 0, seed: 1 };

function foolScore() {
  $("#fool-score").innerHTML = FOOL.tries ? `<span>Attempts: ${FOOL.tries}</span><span style="color:var(--good)">Checker right: ${FOOL.right}/${FOOL.tries}</span><span>Meaning changes caught: ${FOOL.caught}/${FOOL.unsafe}</span><span>Safe changes accepted: ${FOOL.accepted}/${FOOL.safe}</span>` : "";
}

async function foolMutate(kind) {
  const text = $("#fool-text").value.trim();
  if (!text) return;
  FOOL.seed++;
  const d = await api("/api/demo/mutate", { text, kind, seed: FOOL.seed });
  if (!d.applied) {
    $("#fool-out").innerHTML = `<div class="fool-card"><b>${esc(d.label)}</b>: this change doesn't apply to this sentence (for example, there is no number or no pronoun to drop). Try another button or sentence.</div>`;
    return;
  }
  FOOL.tries++;
  if (d.expected_safe) { FOOL.safe++; if (!d.caught) FOOL.accepted++; } else { FOOL.unsafe++; if (d.caught) FOOL.caught++; }
  if (d.correct) FOOL.right++;
  const fails = d.verification.reasons;
  $("#fool-out").innerHTML = `<div class="fool-card ${d.correct ? "right" : "wrong"}">
    <span class="pill ${d.expected_safe ? "good" : "bad"}">${d.expected_safe ? "safe change" : "meaning change"}</span> <b>${esc(d.label)}</b>
    <div class="txt">${esc(d.text)}</div>
    <div class="verdict ${d.caught ? "bad" : "good"}">${d.caught ? "✗ Checker blocked it" : "✓ Checker let it through"} ${d.correct ? "(correct)" : "(wrong)"}</div>
    ${fails.length ? `<ul class="reasons">${fails.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}</div>`;
  foolScore();
}

const BATCH_SAMPLE = `ನಮಸ್ಕಾರ, ನನ್ನ ರೇಷನ್ ಕಾರ್ಡ್ ಸ್ಥಿತಿ ಏನು?
ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಕಚೇರಿ ತೆರೆಯುತ್ತದೆಯೇ?
ನನ್ನ ಅರ್ಜಿ ಸಂಖ್ಯೆ 45821, ಇನ್ನೂ ಹಣ ಬಂದಿಲ್ಲ.
Sir please ನನ್ನ pension update ಮಾಡಿ
मेरे गेहूं की फसल के लिए बीमा कैसे मिलेगा?
அரசு கல்வி உதவித்தொகைக்கு கடைசி தேதி என்ன?
మా ఊరికి బస్ ఎప్పుడు వస్తుంది?
nanna aadhar link aagilla, enu maadbeku?
What documents do I need for a caste certificate?`;

async function runBatch() {
  const text = $("#batch-text").value;
  if (!text.trim()) return;
  $("#batch-out").innerHTML = '<p class="note">Auditing…</p>';
  try {
    const d = await api("/api/demo/batch", { text, tokenizer: $("#batch-model").value });
    S.batch = d;
    const t = d.totals;
    $("#batch-out").innerHTML = `<div class="kpis" style="margin-top:12px">
      <div class="kpi"><b>${t.messages}</b><span>messages in ${t.languages.length} language(s)</span></div>
      <div class="kpi"><b>${fmt(t.tokens)} → ${fmt(t.after)}</b><span>tokens on ${esc(d.label)}, as typed → after</span></div>
      <div class="kpi save"><b>${pct(t.saving, 1)}</b><span>fewer tokens, every rewrite meaning-checked</span></div>
      <div class="kpi"><b>₹${fmt(t.inr_per_lakh_before, 0)} → ₹${fmt(t.inr_per_lakh_after, 0)}</b><span>per 1 lakh messages like these (input) · ${t.blocked_rewrites} unsafe rewrites blocked</span></div></div>
      ${d.truncated ? '<p class="note">Only the first 200 lines were audited.</p>' : ""}
      <div class="scroll-x"><table class="big"><tr><th>#</th><th>Message</th><th>Language</th><th>Chars</th><th>Tokens</th><th>Sent as</th><th>After</th><th>Saved</th><th>Blocked</th></tr>
      ${d.rows.map((r, i) => `<tr><td class="num">${i + 1}</td><td class="native">${esc(r.text)}${r.chosen !== "original" ? `<small>${esc(r.sent)}</small>` : ""}</td><td>${esc(langName(r.language))}</td><td class="num">${r.characters}</td><td class="num">${r.tokens}</td><td>${esc(cname(r.chosen))}</td><td class="num"><b>${r.after}</b></td><td class="down">${r.saving > 0 ? `▼ ${pct(r.saving)}` : "–"}</td><td class="num">${r.blocked.length ? `<span class="pill bad" title="${esc(r.blocked.map(cname).join(", "))}">${r.blocked.length}</span>` : "0"}</td></tr>`).join("")}</table></div>
      <p><button class="solid" id="batch-csv" type="button">Download report (CSV)</button></p>`;
    $("#batch-csv").onclick = downloadBatch;
  } catch (e) { $("#batch-out").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

function downloadBatch() {
  const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const head = ["message", "language", "characters", "tokens", "sent_as", "text_sent", "tokens_after", "saving", "blocked_rewrites"];
  const lines = [head.join(","), ...S.batch.rows.map((r) => [r.text, r.language, r.characters, r.tokens, r.chosen, r.sent, r.after, r.saving.toFixed(4), r.blocked.join(" ")].map(q).join(","))];
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `bhashatoken_audit_${S.batch.tokenizer}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function runApiDemo() {
  const text = $("#api-text").value.trim();
  if (!text) return;
  const model = $("#api-model").value;
  $("#api-snippet-text").textContent = text.length > 40 ? `${text.slice(0, 40)}…` : text;
  const sent = { model, messages: [{ role: "system", content: "You are a helpful government assistant." }, { role: "user", content: text }] };
  try {
    const [res, an] = await Promise.all([api("/v1/chat/completions", sent), api("/api/analyze", { text })]);
    const before = an.rows.find((r) => r.tokenizer === model).tokens;
    const best = res.routing[0].best;
    $("#api-out").innerHTML = `<h4>Your app sends (${before} tokens)</h4><pre class="json api-json">${esc(JSON.stringify(sent, null, 1))}</pre>
      <h4>BhashaToken forwards (${best.input_tokens} tokens · ${esc(cname(best.format))})</h4><pre class="json api-json">${esc(JSON.stringify({ model: res.model, messages: res.messages }, null, 1))}</pre>
      <div class="verdict good">▼ ${pct(1 - best.input_tokens / Math.max(before, 1))} fewer input tokens on ${esc(label(model))}, meaning checked · ${res.dry_run ? "dry run (not sent to any AI)" : "sent upstream"}</div>`;
  } catch (e) { $("#api-out").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

async function initDemos() {
  fill("#mem-lang", S.meta.languages.map((l) => [l.code, l.name]), "kan_Knda");
  const models = S.meta.tokenizers.map((t) => [t.key, t.label]);
  fill("#mem-model", models, "llama3"); fill("#batch-model", models, "llama3"); fill("#api-model", models, "llama3");
  $("#mem-play").onclick = memPlay;
  ["#mem-lang", "#mem-model"].forEach((id) => $(id).addEventListener("change", () => { if (MEM.timer) { clearInterval(MEM.timer); MEM.timer = null; } MEM.data = null; $("#mem-play").textContent = "▶ Play"; }));
  $("#mem-window").onchange = () => { if (MEM.data && !MEM.timer) { const s0 = MEM.step; MEM.step = s0 - 1; memStep(); } };
  const pbox = $("#fool-presets");
  FOOL_PRESETS.forEach(([name, txt], i) => {
    const b = document.createElement("button"); b.textContent = name; if (i === 0) b.classList.add("on");
    b.onclick = () => { $("#fool-text").value = txt; pbox.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); $("#fool-out").innerHTML = ""; };
    pbox.appendChild(b);
  });
  $("#fool-text").value = FOOL_PRESETS[0][1];
  const muts = (await api("/api/demo/mutations")).mutations;
  const mk = (m) => `<button data-k="${m.kind}" type="button">${esc(m.label)}</button>`;
  $("#mut-safe").innerHTML = muts.filter((m) => m.safe).map(mk).join("");
  $("#mut-bad").innerHTML = muts.filter((m) => !m.safe).map(mk).join("");
  document.querySelectorAll("#mut-safe button, #mut-bad button").forEach((b) => { b.onclick = () => foolMutate(b.dataset.k); });
  $("#batch-text").value = BATCH_SAMPLE;
  $("#batch-run").onclick = runBatch;
  $("#batch-file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const txt = await f.text();
    $("#batch-text").value = txt.split(/\r?\n/).map((l) => l.replace(/^"|"$/g, "")).join("\n").slice(0, 200000);
    runBatch();
  };
  $("#api-send").onclick = runApiDemo;
}

// ================= VOICE INPUT =================
let REC = null;
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $("#mic-btn"), status = $("#mic-status");
  if (!SR) {
    btn.onclick = () => { status.hidden = false; status.textContent = "Speech input needs Google Chrome or Microsoft Edge (and internet). You can still type or paste."; };
    return;
  }
  btn.onclick = () => {
    if (REC) { REC.stop(); return; }
    REC = new SR();
    REC.lang = $("#mic-lang").value; REC.interimResults = true; REC.continuous = false;
    btn.classList.add("on"); btn.textContent = "■ Stop";
    status.hidden = false; status.textContent = "Listening… speak now.";
    REC.onresult = (e) => {
      const text = Array.from(e.results).map((r) => r[0].transcript).join(" ");
      $("#txt").value = text;
      status.textContent = e.results[e.results.length - 1].isFinal ? "Got it. Measuring…" : `Hearing: ${text}`;
    };
    REC.onerror = (e) => { status.textContent = e.error === "not-allowed" ? "Microphone permission was blocked. Allow it in the address bar and try again." : `Speech error: ${e.error}. Speech input needs internet.`; };
    REC.onend = () => {
      btn.classList.remove("on"); btn.textContent = "🎤 Speak"; REC = null;
      document.querySelectorAll("#examples button").forEach((x) => x.classList.remove("on"));
      if ($("#txt").value.trim()) runTry();
    };
    REC.start();
  };
}

// ================= CROWD CHECK =================
let crowdTimer = null;
async function crowdRefresh() {
  try {
    const d = await api("/api/crowd/summary");
    const t = d.total;
    $("#crowd-kpis").innerHTML = `
      <div class="kpi"><b>${d.ratings}</b><span>ratings from people who read the language${d.excluded_not_speakers ? ` (${d.excluded_not_speakers} more not counted)` : ""}</span></div>
      <div class="kpi safe"><b>${d.same_share === null ? "–" : pct(d.same_share, 0)}</b><span>said "same meaning"</span></div>
      <div class="kpi"><b>${t.partly} · ${t.different}</b><span>"partly" · "different"</span></div>`;
    const langs = d.by_language;
    chart("chart-crowd", {
      type: "bar",
      data: { labels: langs.length ? langs.map((x) => langName(x.language)) : ["waiting for ratings"], datasets: [
        { label: "Same", data: langs.map((x) => x.same), backgroundColor: "#23845a", borderRadius: 3 },
        { label: "Partly", data: langs.map((x) => x.partly), backgroundColor: "#d08a17", borderRadius: 3 },
        { label: "Different", data: langs.map((x) => x.different), backgroundColor: "#c13b27", borderRadius: 3 },
      ] },
      options: { animation: false, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: "ratings" } } } },
    });
  } catch (e) { /* server busy: try again next tick */ }
}
async function initCrowd() {
  try {
    const p = await api("/api/phone?path=rate");
    $("#crowd-qr").innerHTML = p.svg || "";
    $("#crowd-url").innerHTML = p.url ? `Same Wi-Fi: <b>${esc(p.url)}</b>` : `Open <b>http://127.0.0.1:8000/rate</b> (no Wi-Fi address found)`;
  } catch (e) { $("#crowd-url").textContent = "Open /rate on this server"; }
  crowdRefresh();
}
function crowdLive(on) {
  if (on && !crowdTimer) crowdTimer = setInterval(crowdRefresh, 3000);
  if (!on && crowdTimer) { clearInterval(crowdTimer); crowdTimer = null; }
}

// ================= RECEIPT =================
async function renderReceipt() {
  const lang = $("#rc-lang").value, model = $("#rc-model").value, msgs = Number($("#rc-msgs").value);
  try {
    const d = await api("/api/simulate", { daily_users: 1, messages_per_user: msgs / 30, regional_share: 1, input_words: 40, output_words: 120, model, language: lang });
    const month = d.annual_inr / 12, eng = d.english_only_annual_inr / 12, saved = d.saved_by_optimizer_annual_inr / 12;
    const r2 = (x) => `₹${x < 10 ? x.toFixed(2) : fmt(x, 0)}`;
    const now = new Date();
    $("#receipt").innerHTML = `<h4>TOKEN TAX RECEIPT</h4><div class="c">BhashaToken · ${now.toLocaleDateString("en-IN")} · simulated</div><hr>
      <div class="ln"><span>Language</span><span>${esc(langName(lang))}</span></div>
      <div class="ln"><span>AI model</span><span>${esc(label(model))}</span></div>
      <div class="ln"><span>Messages this month</span><span>${fmt(msgs)}</span></div>
      <div class="ln"><span>Tokens per message</span><span>${fmt(d.tokens_per_message_native, 0)} (English ${fmt(d.tokens_per_message_english, 0)})</span></div><hr>
      <div class="ln"><span>English user pays</span><span>${r2(eng)}</span></div>
      <div class="ln big"><span>You pay</span><span>${r2(month)}</span></div>
      <div class="ln red"><span>Language tax</span><span>+${r2(month - eng)} (${pct(month / eng - 1, 0)} extra)</span></div>
      <div class="ln"><span>AI memory vs English</span><span>${pct(d.effective_context_vs_english, 0)}</span></div><hr>
      <div class="ln grn"><span>With BhashaToken</span><span>${r2(month - saved)}</span></div>
      <div class="ln grn"><span>You save</span><span>${r2(saved)} / month · ${r2(saved * 12)} / year</span></div>
      <div class="ln grn"><span>AI memory vs English</span><span>${pct(d.effective_context_vs_english_optimized, 0)}</span></div><hr>
      <div class="c">Questions rewritten only when every meaning check passes.<br>Measure the Tax. Preserve the Meaning.</div>`;
  } catch (e) { $("#receipt").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
function initReceipt() {
  fill("#rc-lang", S.meta.languages.map((l) => [l.code, l.name]), "kan_Knda");
  fill("#rc-model", keys().map((k) => [k, label(k)]), "gpt4");
  ["#rc-lang", "#rc-model", "#rc-msgs"].forEach((id) => { $(id).onchange = renderReceipt; });
  $("#rc-print").onclick = () => window.print();
  renderReceipt();
}

// ================= LOCAL AI ANSWER CHECK =================
async function localAiStatus() {
  try {
    const st = await api("/api/localai/status");
    S.localai = st;
    $("#ai-status").innerHTML = st.running && st.model_ready
      ? `<span class="pill good">connected</span> ${esc(st.model)} on ${esc(st.where)}${st.backend === "ollama" ? " through Ollama" : ""}${st.where === "the internet" ? " (your text is sent to that service)" : ""}`
      : st.backend === "openai"
        ? `<span class="pill bad">not connected</span> ${esc(st.help)}`
      : st.running
        ? `<span class="pill warn">model missing</span> Ollama is running, but ${esc(st.model)} is not downloaded. In a terminal run: <code>ollama pull ${esc(st.model)}</code>`
        : `<span class="pill">not connected</span> Optional: install Ollama from ollama.com, then run <code>ollama pull ${esc(st.model)}</code>. Everything else works without it.`;
    $("#ai-ask").disabled = !(st.running && st.model_ready);
  } catch (e) { $("#ai-status").textContent = "Could not check the local AI."; }
}

async function localAiAsk() {
  if (!S.last) return;
  const choice = S.last.opt.choices.find((c) => c.tokenizer === "gemma3") || S.last.opt.choices[0];
  const rewrite = choice.text || S.last.text;
  const t0 = Date.now();
  $("#ai-ask").disabled = true;
  const timer = setInterval(() => { $("#ai-out").innerHTML = `<p class="note">Asking the local AI twice… ${Math.round((Date.now() - t0) / 1000)} s (a laptop CPU can take 10–60 seconds)</p>`; }, 1000);
  try {
    const d = await api("/api/localai/answer", { original: S.last.text, rewrite, language: S.last.an.language });
    clearInterval(timer);
    const col = (title, q, a, cls) => `<div class="ai-col ${cls}"><small>${title}</small><div class="ai-q">${esc(q)}</div>
      <div class="note">${a.input_tokens} input tokens counted by the model · ${a.seconds} s</div><div class="ai-a">${esc(a.answer || "(empty answer)")}</div></div>`;
    $("#ai-out").innerHTML = `<div class="compare">${col("Question as typed", S.last.text, d.original, "")}${col(`BhashaToken version (${cname(choice.best)})`, rewrite, d.rewrite, "opt")}</div>
      <div class="best-card"><b>${d.original.input_tokens} → ${d.rewrite.input_tokens} input tokens</b>, measured by ${esc(d.model)} itself.
      Answer overlap (rough text similarity): <b>${pct(d.answer_similarity, 0)}</b>. Read both answers: do they say the same thing? <span class="note">Small or free models make mistakes in both versions; this shows the method, not a benchmark.</span></div>`;
  } catch (e) {
    clearInterval(timer);
    $("#ai-out").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  } finally { $("#ai-ask").disabled = false; }
}

// ================= SAFETY LAB =================
const LAB_PRESETS = [
  ["Negation dropped (medicine)", "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.", "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ."],
  ["Number changed", "ಅರ್ಜಿ ಸಲ್ಲಿಸಲು ಕೊನೆಯ ದಿನಾಂಕ 15/12/2026, ಶುಲ್ಕ 250 ರೂಪಾಯಿ.", "ಅರ್ಜಿ ಸಲ್ಲಿಸಲು ಕೊನೆಯ ದಿನಾಂಕ 16/12/2026, ಶುಲ್ಕ 250 ರೂಪಾಯಿ."],
  ["Question became statement", "क्या कल स्कूल बंद रहेगा?", "कल स्कूल बंद रहेगा।"],
  ["Safe: English letters", "ನಾಳೆ ಬೆಳಿಗ್ಗೆ 9:30 ಕ್ಕೆ ಸಭೆ ಇದೆ.", "naale beligge 9:30 kke sabhe ide."],
  ["Safe: pronoun drop", "ನಾನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ.", "ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ."],
  ["Tamil: word deleted", "நாளை காலை மருத்துவமனைக்கு வர வேண்டும்.", "நாளை மருத்துவமனைக்கு வர வேண்டும்."],
];
const LAYER = { critical: "critical fact", similarity: "similarity", structure: "structure", warning: "warning" };
let labTimer = null;

async function runLab() {
  const original = $("#lab-a").value.trim(), candidate = $("#lab-b").value.trim();
  if (!original || !candidate) return;
  try {
    const d = await api("/api/verify", { original, candidate });
    $("#lab-out").innerHTML = `<div class="verdict ${d.passed ? "good" : "bad"}">${d.passed ? "✓ Safe: the meaning is preserved" : "✗ Rejected: this rewrite changes the meaning"}</div>
      <div class="checks">${d.checks.map((c) => `<div class="chk ${c.passed || c.layer === "warning" ? "" : "fail"}">
        <span class="${c.passed ? "ok" : "no"}">${c.passed ? "✓" : "✗"}</span><b>${esc(c.name.replace(/_/g, " "))}</b>
        <span class="detail">${esc(c.detail)}</span><span class="layer pill">${esc(LAYER[c.layer] || c.layer)}</span></div>`).join("")}</div>
      <p class="note">Language: ${esc(langName(d.language))} · similarity by ${esc(d.similarity_method)}, threshold ${fmt(d.threshold, 3)} (${esc(d.threshold_source)})</p>`;
  } catch (e) { $("#lab-out").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

function renderReviewSummary(sum) {
  if (!sum || !sum.total) {
    $("#review-summary").innerHTML = `<p class="note">No reviews collected yet. Rate rewrites on the Optimizer page (step 4); they appear here.</p>`;
    return;
  }
  $("#review-summary").innerHTML = `<p><b>${sum.total}</b> review(s) from <b>${sum.reviewers}</b> reviewer(s)</p>
    <table><tr><th>Candidate</th><th>n</th><th>Meaning</th><th>Fluency</th><th>Acceptable</th></tr>
    ${sum.by_candidate.map((r) => `<tr><td>${esc(cname(r.candidate))}</td><td class="num">${r.n}</td><td class="num">${fmt(r.meaning, 2)}</td><td class="num">${fmt(r.fluency, 2)}</td><td class="num">${pct(r.acceptable)}</td></tr>`).join("")}</table>`;
}

function initSafety() {
  const box = $("#lab-presets");
  LAB_PRESETS.forEach(([name, a, b], i) => {
    const btn = document.createElement("button");
    btn.textContent = name; if (i === 0) btn.classList.add("on");
    btn.onclick = () => { $("#lab-a").value = a; $("#lab-b").value = b; box.querySelectorAll("button").forEach((x) => x.classList.remove("on")); btn.classList.add("on"); runLab(); };
    box.appendChild(btn);
  });
  $("#lab-a").value = LAB_PRESETS[0][1]; $("#lab-b").value = LAB_PRESETS[0][2];
  ["#lab-a", "#lab-b"].forEach((id) => $(id).addEventListener("input", () => { clearTimeout(labTimer); labTimer = setTimeout(runLab, 350); }));
  runLab();
  safetyCharts();
  api("/api/reviews").then(renderReviewSummary).catch(() => renderReviewSummary(null));
}

function safetyCharts() {
  const v = S.res.verification;
  $("#safety-kpis").innerHTML = `
    <div class="kpi safe"><b>${pct(v.summary.unsafe_caught, 1)}</b><span>unsafe changes caught (${fmt(v.by_kind.filter((k) => !k.safe).reduce((a, k) => a + k.n, 0))} tests, 6 languages)</span></div>
    <div class="kpi"><b>${pct(v.summary.safe_accepted, 1)}</b><span>safe rewrites correctly accepted (the rest are false alarms)</span></div>
    <div class="kpi"><b>${pct(v.summary.similarity_only.unsafe_caught, 0)}</b><span>caught by a similarity score alone, the usual approach</span></div>`;
  const kinds = v.by_kind;
  const nice = (k) => k.replace("unsafe:", "✗ ").replace("safe:", "✓ ").replace(/_/g, " ");
  chart("chart-catch", {
    type: "bar",
    data: { labels: kinds.map((k) => nice(k.kind)), datasets: [
      { label: "Similarity score only", data: kinds.map((k) => (k.safe ? k.accepted_similarity_only : 1 - k.accepted_similarity_only) * 100), backgroundColor: "#9fb3d1", borderRadius: 3 },
      { label: "BhashaToken verifier", data: kinds.map((k) => (k.safe ? k.accepted : 1 - k.accepted) * 100), backgroundColor: "#ef8f1f", borderRadius: 3 },
    ] },
    options: { indexAxis: "y", scales: { x: { min: 0, max: 100, title: { display: true, text: "% handled correctly (✓ accepted / ✗ caught)" } } },
      plugins: { tooltip: { callbacks: { afterLabel: (c) => `n = ${kinds[c.dataIndex].n}` } } } },
  });
  const ab = v.ablation;
  chart("chart-layers", {
    type: "bar",
    data: { labels: ab.map((r) => r.removed_check.replace(/_/g, " ")), datasets: [
      { label: "Unsafe caught", data: ab.map((r) => r.unsafe_caught * 100), backgroundColor: "#c13b27", borderRadius: 3 },
      { label: "Safe accepted", data: ab.map((r) => r.safe_accepted * 100), backgroundColor: "#23845a", borderRadius: 3 },
    ] },
    options: { scales: { y: { min: 0, max: 100, title: { display: true, text: "%" } }, x: { ticks: { autoSkip: false, maxRotation: 50 } } } },
  });
}

// ================= RESEARCH =================
function initResearch() {
  const st = S.res.strategies;
  const pts = st.pareto;
  const ours = (n) => n.includes("ours");
  const PCOL = { original: "#8a94a8", always_iso_roman: "#7b4fb3", always_casual_roman: "#3f6fbf", always_compressed: "#1f8e9e",
    best_fixed_per_model_language: "#23845a", truncate_25pct: "#c13b27", adaptive_unverified: "#d08a17" };
  chart("chart-pareto", {
    type: "scatter",
    data: { datasets: pts.map((p) => ({
      label: p.strategy.replace(/_/g, " "),
      data: [{ x: p.token_reduction * 100, y: p.meaning_safe_rate * 100 }],
      pointRadius: ours(p.strategy) ? 13 : 8, pointHoverRadius: ours(p.strategy) ? 15 : 10,
      pointStyle: ours(p.strategy) ? "rectRot" : "circle",
      backgroundColor: ours(p.strategy) ? "#ef8f1f" : PCOL[p.strategy] || "#3f6fbf",
      borderColor: ours(p.strategy) ? "#1b2440" : "#ffffff", borderWidth: 2,
    })) },
    options: { plugins: { legend: { position: "bottom", labels: { usePointStyle: true } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw.x, 1)}% fewer tokens, ${fmt(c.raw.y, 1)}% meaning-safe` } } },
      scales: { x: { min: 0, max: 60, title: { display: true, text: "token reduction vs as typed (%)" } }, y: { min: 0, max: 105, title: { display: true, text: "outputs passing meaning verification (%)" } } } },
  });
  $("#pareto-table").innerHTML = `<table><tr><th>Strategy</th><th>Token reduction</th><th>Meaning-safe outputs</th></tr>${[...pts].sort((a, b) => b.token_reduction - a.token_reduction).map((p) => `<tr class="${ours(p.strategy) ? "chosen" : ""}"><td>${esc(p.strategy.replace(/_/g, " "))}</td><td class="num">${pct(p.token_reduction, 1)}</td><td class="num">${p.meaning_safe_rate < 1 ? `<span class="pill ${p.meaning_safe_rate < 0.9 ? "bad" : "warn"}">${pct(p.meaning_safe_rate, 1)}</span>` : '<span class="pill good">100%</span>'}</td></tr>`).join("")}</table>`;
  const best = pts.find((p) => ours(p.strategy)), trunc = pts.find((p) => p.strategy.startsWith("truncate")), unv = pts.find((p) => p.strategy === "adaptive_unverified");
  $("#pareto-note").innerHTML = `FLORES devtest, ${st.sentences_per_language} sentences × ${st.languages.length} languages × 7 tokenizers. Ours (diamond): <b>${pct(best.token_reduction, 1)}</b> fewer tokens with <b>${pct(best.meaning_safe_rate, 0)}</b> verified outputs. Cutting the last quarter of each message saves ${pct(trunc.token_reduction, 0)} but only ${pct(trunc.meaning_safe_rate, 0)} stay safe; picking the cheapest rewrite without checking makes ${pct(1 - unv.meaning_safe_rate, 1)} unsafe. Top-right is best.`;

  const pt = st.per_tokenizer;
  chart("chart-paired", {
    type: "bar",
    data: { labels: pt.map((r) => label(r.tokenizer)), datasets: [
      { label: "Best single format", data: pt.map((r) => r.best_fixed_saving * 100), backgroundColor: "#9fb3d1", borderRadius: 3 },
      { label: "BhashaToken adaptive", data: pt.map((r) => r.adaptive_saving * 100), backgroundColor: "#ef8f1f", borderRadius: 3 },
    ] },
    options: { plugins: { tooltip: { callbacks: { afterBody: (it) => { const t = pt[it[0].dataIndex].test; return `difference ${fmt(t.mean_diff * 100, 1)} pts, 95% CI [${fmt(t.ci95[0] * 100, 1)}, ${fmt(t.ci95[1] * 100, 1)}], p = ${t.p_value < 0.001 ? "< 0.001" : fmt(t.p_value, 3)}`; } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "mean token saving per sentence (%)" } } } },
  });
  $("#paired-table").innerHTML = `<table><tr><th>Tokenizer</th><th>Mean gain (pts)</th><th>95% CI</th><th>p (paired permutation)</th><th>n</th></tr>
    ${pt.map((r) => `<tr><td>${esc(label(r.tokenizer))}</td><td class="num">${fmt(r.test.mean_diff * 100, 2)}</td><td class="num">[${fmt(r.test.ci95[0] * 100, 2)}, ${fmt(r.test.ci95[1] * 100, 2)}]</td><td class="num">${r.test.p_value < 0.001 ? "&lt; 0.001" : fmt(r.test.p_value, 3)}</td><td class="num">${r.test.n}</td></tr>`).join("")}</table>`;

  const ab = st.ablation;
  chart("chart-ablation", {
    type: "bar",
    data: { labels: ab.map((r) => r.variant), datasets: [{ data: ab.map((r) => r.token_reduction * 100), borderRadius: 3,
      backgroundColor: ab.map((r) => (r.variant === "full system" ? "#ef8f1f" : r.unsafe_outputs ? "#c13b27" : "#9fb3d1")) }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, title: { display: true, text: "token reduction (%)" } } } },
  });
  const nv = ab.find((r) => r.unsafe_outputs !== undefined);
  $("#ablation-note").innerHTML = `Removing a part lowers the saving by the gap shown. The red bar saves more only because ${pct(nv.unsafe_outputs, 1)} of its outputs fail verification; that is the risk the verifier removes.`;

  const fails = {};
  st.failure_by_candidate.forEach((f) => { fails[f.reason] = (fails[f.reason] || 0) + f.count; });
  const fk = Object.entries(fails).sort((a, b) => b[1] - a[1]);
  chart("chart-fail", {
    type: "doughnut",
    data: { labels: fk.map(([k]) => k.replace(/_/g, " ")), datasets: [{ data: fk.map(([, v]) => v), backgroundColor: ["#c13b27", "#ef8f1f", "#3f6fbf", "#23845a", "#7b4fb3", "#1f8e9e", "#8a94a8", "#d08a17"] }] },
    options: { cutout: "55%", plugins: { legend: { position: "right" } } },
  });
  $("#fail-examples").innerHTML = `<h4>One real example per failure type</h4>${st.failure_examples.map((e) => `<div class="word"><b>${esc(e.reason.split(":")[0].replace(/_/g, " "))}</b> <span class="note">${esc(langName(e.language))} · ${esc(cname(e.candidate))}</span><div class="note">${esc(e.reason)}</div><div>${esc(e.original)}</div><div class="note">→ ${esc(e.rewrite)}</div></div>`).join("")}`;

  const doms = [...new Set(st.domains.map((d) => d.domain))];
  const strats = [...new Set(st.domains.map((d) => d.strategy))];
  chart("chart-domain", {
    type: "bar",
    data: { labels: doms.map((d) => d.split(" (")[0]), datasets: strats.map((sname, i) => ({
      label: sname.replace(/_/g, " "), backgroundColor: ["#ef8f1f", "#9fb3d1", "#3f6fbf", "#23845a"][i % 4], borderRadius: 3,
      data: doms.map((d) => { const r = st.domains.find((x) => x.domain === d && x.strategy === sname); return r ? r.token_reduction * 100 : 0; }) })) },
    options: { plugins: { tooltip: { callbacks: { afterLabel: (c) => { const r = st.domains.find((x) => x.domain === doms[c.dataIndex] && x.strategy === strats[c.datasetIndex]); return `meaning-safe ${pct(r.meaning_safe_rate, 1)} · ${r.sentences} texts`; } } } },
      scales: { y: { title: { display: true, text: "token reduction (%)" } } } },
  });

  const cf = S.res.counterfactual;
  chart("chart-cf", {
    type: "bar",
    data: { labels: cf.per_tokenizer.map((r) => label(r.tokenizer)), datasets: [
      { label: "Fix the tokenizer's word splitting (same vocabulary)", data: cf.per_tokenizer.map((r) => r.saving_fixed_pretokenizer * 100), backgroundColor: "#9fb3d1", borderRadius: 3 },
      { label: "BhashaToken: rewrite the text (verified)", data: cf.per_tokenizer.map((r) => r.saving_bhashatoken * 100), backgroundColor: "#ef8f1f", borderRadius: 3 },
    ] },
    options: { scales: { y: { title: { display: true, text: "token saving (%)" } } } },
  });
  const split = cf.per_tokenizer.filter((r) => r.has_split).map((r) => label(r.tokenizer));
  const maxFix = Math.max(...cf.per_tokenizer.map((r) => Math.abs(r.saving_fixed_pretokenizer)));
  const cfOurs = cf.per_tokenizer.map((r) => r.saving_bhashatoken);
  const range = `${fmt(Math.min(...cfOurs) * 100, 0)}–${fmt(Math.max(...cfOurs) * 100, 0)}%`;
  $("#cf-note").innerHTML = `<p>Researchers point out that some tokenizers cut words at vowel signs (${esc(split.join(", "))}). We tested the obvious fix: keep vowel signs inside words, with the same vocabulary.</p>
    <p><b>Result: it changes the token count by at most ${pct(maxFix, 1)}</b> across ${cf.languages.length} languages × 7 tokenizers (${cf.sentences_per_language} sentences each), and sometimes adds tokens. The words split differently, but the vocabulary never learned whole Indian-language words, so the same pieces come out.</p>
    <p>So fixing the tokenizer means <b>retraining the vocabulary and the model</b>, which only the model's maker can do. Rewriting the text, as BhashaToken does, saves <b>${range}</b> on the same models <b>today</b>.</p>`;

  const l = S.res.lfi_sensitivity;
  $("#lfi-box").innerHTML = `<p class="note"><code>${esc(l.formula)}</code>. Parts are each a ratio to English, floored at 0.01.</p>
    <p>Across <b>${l.draws}</b> random weightings of the 5 parts, the ranking of ${l.pairs} model–language pairs stays similar: Kendall τ mean <b>${fmt(l.kendall_tau.mean, 3)}</b> (5th percentile ${fmt(l.kendall_tau.p05, 3)}). The worst pair (${esc(l.worst_pair.split("|").map((x, i) => (i ? langName(x) : label(x))).join(", "))}) stays worst in ${pct(l.worst_pair_stable, 0)} of weightings.</p>`;
  chart("chart-lfi", {
    type: "bar",
    data: { labels: l.leave_one_out.map((r) => `without ${r.dropped_part.replace(/_/g, " ")}`), datasets: [{ data: l.leave_one_out.map((r) => r.kendall_tau), backgroundColor: "#3f6fbf", borderRadius: 3 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 1, title: { display: true, text: "Kendall τ vs full index (1 = same ranking)" } } } },
  });

  const w = Object.entries(st.winning_candidate);
  chart("chart-wins", {
    type: "pie",
    data: { labels: w.map(([k]) => cname(k)), datasets: [{ data: w.map(([, v]) => v), backgroundColor: ["#ef8f1f", "#3f6fbf", "#23845a", "#9fb3d1", "#7b4fb3", "#c13b27", "#1f8e9e"] }] },
    options: { plugins: { legend: { position: "right" } } },
  });
}

// ================= SIMULATOR =================
let simTimer = null;
function simOutputs() {
  const f = $("#sim-form");
  f.querySelectorAll("output").forEach((o) => {
    const v = Number(f.elements[o.dataset.for].value);
    o.textContent = o.dataset.for === "regional_share" ? pct(v) : fmt(v);
  });
}

async function runSim() {
  simOutputs();
  const f = $("#sim-form");
  const body = Object.fromEntries(new FormData(f).entries());
  ["daily_users"].forEach((k) => { body[k] = parseInt(body[k], 10); });
  ["messages_per_user", "regional_share", "input_words", "output_words", "input_price_usd", "output_price_usd"].forEach((k) => { body[k] = parseFloat(body[k]); });
  body.context_window = parseInt(body.context_window, 10) || null;
  body.reply_optimized = f.elements.reply_optimized.checked;
  ["input_price_usd", "output_price_usd"].forEach((k) => { if (Number.isNaN(body[k])) body[k] = null; });
  try {
    const d = await api("/api/simulate", body);
    $("#sim-kpis").innerHTML = `
      <div class="kpi"><b>${rupees(d.annual_inr)}</b><span>per year · ${rupees(d.monthly_inr)} a month · ${rupees(d.daily_inr)} a day</span></div>
      <div class="kpi"><b>${rupees(d.extra_annual_inr)}</b><span>of that is the token tax</span></div>
      <div class="kpi save"><b>${rupees(d.saved_by_optimizer_annual_inr)}</b><span>saved per year by verified rewriting (simulated)</span></div>
      <div class="kpi"><b>${fmt(d.rates.premium, 2)}× → ${fmt(d.rates.premium_compressed, 2)}×</b><span>tokens vs English, before → after our fixes</span></div>`;
    chart("chart-sim-pie", {
      type: "pie",
      data: { labels: ["Same chats in English", "Token tax"], datasets: [{ data: [d.english_only_annual_inr, Math.max(d.extra_annual_inr, 0)], backgroundColor: ["#9fb3d1", "#c0412c"] }] },
      options: { plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => `${c.label}: ${rupees(c.raw)}` } } } },
    });
    const after1 = d.annual_inr - d.saved_by_optimizer_annual_inr;
    const after2 = after1 - d.saved_by_compression_annual_inr;
    chart("chart-sim-bar", {
      type: "bar",
      data: { labels: ["As typed", "+ Rewriting", "+ Compression", "English only"],
        datasets: [{ data: [d.annual_inr, after1, after2, d.english_only_annual_inr], backgroundColor: ["#c0412c", "#f39a2b", "#f6b867", "#9fb3d1"], borderRadius: 4 }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => rupees(c.raw) } } },
        scales: { y: { beginAtZero: true, ticks: { callback: (v) => rupees(v) } } } },
    });
    chart("chart-ctx", {
      type: "bar",
      data: { labels: ["English", `${langName(body.language)} as typed`, `${langName(body.language)} optimized`],
        datasets: [{ data: [d.messages_in_context_english, d.messages_in_context_native, d.messages_in_context_optimized], backgroundColor: ["#9fb3d1", "#c13b27", "#ef8f1f"], borderRadius: 4 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 0)} question+answer pairs` } } },
        scales: { x: { beginAtZero: true, title: { display: true, text: "question + answer pairs that fit in the context window" } } } },
    });
    $("#ctx-note").innerHTML = `A ${langName(body.language)} user gets <b>${pct(d.effective_context_vs_english, 0)}</b> of the conversation memory an English user gets for the same content; <b>${pct(d.effective_context_vs_english_optimized, 0)}</b> after optimization. Prices used: $${fmt(d.price_input_usd, 2)} in / $${fmt(d.price_output_usd, 2)} out per 1M tokens.`;
  } catch (e) {
    $("#sim-kpis").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

function initSimulator() {
  fill("#sim-model", keys().map((k) => [k, label(k)]), "gpt4o");
  fill("#sim-lang", [["kan_Knda", "Kannada"], ["tam_Taml", "Tamil"], ["tel_Telu", "Telugu"], ["hin_Deva", "Hindi"], ["ben_Beng", "Bengali"], ["mal_Mlym", "Malayalam"]], "kan_Knda");
  $("#sim-form").addEventListener("input", () => { simOutputs(); clearTimeout(simTimer); simTimer = setTimeout(runSim, 200); });
  $("#sim-form").addEventListener("submit", (e) => e.preventDefault());
  runSim();
}

// ================= EVIDENCE =================
function initHow() {
  const b = S.res.benchmark;
  const native = b.flores.filter((x) => x.style === "native" && x.language !== "eng_Latn");
  const worst = native.reduce((a, x) => (x.premium > a.premium ? x : a));
  const best = native.reduce((a, x) => (x.premium < a.premium ? x : a));
  const nativeWins = (k) => Object.values(b.cheaper_style[k] || {}).filter((s) => s === "native").length;
  const al = S.res.explainer.rows.filter((r) => r.grammar_alignment !== null);
  const meanAl = al.reduce((s, r) => s + r.grammar_alignment, 0) / al.length;
  const comp = Math.max(...S.res.compression.rows.filter((r) => r.set.startsWith("flores")).flatMap((r) => Object.values(r.saving)));
  $("#findings").innerHTML = [
    `The same sentence needs from <b>${fmt(best.premium, 2)}×</b> (${esc(langName(best.language))}, ${esc(label(best.tokenizer))}) to <b>${fmt(worst.premium, 1)}×</b> (${esc(langName(worst.language))}, ${esc(label(worst.tokenizer))}) the tokens of English.`,
    `Native script is cheaper than English letters in ${nativeWins("gemma3")} of 14 languages on Gemma 3, but ${nativeWins("llama3")} of 14 on Llama 3. Never assume: measure.`,
    `Typing in English letters never gets below about 1.9× English on any model.`,
    `Only ${pct(meanAl, 0)} of token cuts land on real word parts (roots, suffixes): tokenizers ignore grammar.`,
    `Our predictor errs by 3–8% vs 29–60% for "characters ÷ 4".`,
    `Grammar-safe compression saves at most ${pct(comp, 1)} on formal text: the money is in format and model choice.`,
    `A similarity score alone catches only ${pct(S.res.verification.summary.similarity_only.unsafe_caught, 0)} of meaning-changing edits; our layered verifier catches ${pct(S.res.verification.summary.unsafe_caught, 1)} while accepting ${pct(S.res.verification.summary.safe_accepted, 1)} of safe rewrites.`,
  ].map((t) => `<li>${t}</li>`).join("");
}

function renderLedger() {
  const l = S.res.ledger, e = S.res.efficiency;
  const ds = Object.keys(l.datasets).length;
  $("#ledger").innerHTML = `<table>
    <tr><th>System</th><td>${esc(l.system_version)} · seed <code>${l.seed}</code> · ${esc(l.created)}</td></tr>
    <tr><th>Machine</th><td>${esc(l.platform)} · ${l.cpu_count} CPU threads · ${esc(l.gpu)}</td></tr>
    <tr><th>Python / libs</th><td>${esc(l.python)} · ${Object.entries(l.packages).map(([k, v]) => `${esc(k)} ${esc(v)}`).join(" · ")}</td></tr>
    <tr><th>Tokenizers</th><td>${Object.entries(l.tokenizers).map(([k, h]) => `${esc(label(k))} <code>${esc(h)}</code>`).join("<br>")}</td></tr>
    <tr><th>Data</th><td>${ds} files hashed (SHA-256) in <code>results/ledger.json</code></td></tr>
    <tr><th>Speed</th><td>${fmt(e.plan_ms_per_sentence_all_tokenizers, 0)} ms to optimize + verify one sentence for all 7 tokenizers · process memory ${fmt(e.process_rss_mb, 0)} MB · CPU only</td></tr>
    <tr><th>Re-run</th><td><code>python -m tokentax.experiments</code></td></tr></table>
    <div class="rmode"><h4>Per tokenizer</h4><table><tr><th>Tokenizer</th><th>Load</th><th>Encode / sentence</th></tr>
    ${e.tokenizers.map((r) => `<tr><td>${esc(label(r.tokenizer))}</td><td class="num">${fmt(r.load_ms, 0)} ms</td><td class="num">${fmt(r.encode_ms_per_sentence, 3)} ms</td></tr>`).join("")}</table></div>`;
}

async function renderAudit() {
  try {
    const d = await api("/api/audit?limit=12");
    if (!d.items.length) { $("#audit-table").innerHTML = `<p class="note">No decisions logged yet. Try a sentence on the Optimizer page.</p>`; return; }
    $("#audit-table").innerHTML = `<table><tr><th>ID</th><th>Time (UTC)</th><th>Input</th><th>Rejected</th></tr>${d.items.map((i) => `<tr data-id="${esc(i.request_id)}" style="cursor:pointer"><td><code>${esc(i.request_id)}</code></td><td class="note">${esc(i.created.replace("T", " ").replace("+00:00", ""))}</td><td>${esc(i.input.slice(0, 60))}${i.input.length > 60 ? "…" : ""}</td><td class="num">${Object.keys(i.rejected).length}</td></tr>`).join("")}</table>`;
    $("#audit-table").querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.onclick = async () => { const one = await api(`/api/audit/${tr.dataset.id}`); const box = $("#audit-detail"); box.hidden = false; box.textContent = JSON.stringify(one, null, 1); };
    });
  } catch (e) { $("#audit-table").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

async function phoneBox(target) {
  const d = await api("/api/phone").catch(() => ({ url: null, svg: null }));
  if (d.url && d.svg) {
    target.innerHTML = `${d.svg}<div><p>Scan with your phone camera, or open:</p><code>${esc(d.url)}</code>
      <p class="note">The laptop must run the server with <code>--host 0.0.0.0</code> and both devices must be on the same Wi-Fi.</p></div>`;
  } else {
    target.innerHTML = `<p class="note">Start the server with <code>--host 0.0.0.0</code> and install <code>qrcode</code> to show a QR code here. Then open http://&lt;laptop-IP&gt;:8000 on the phone.</p>`;
  }
}

// ================= JUDGE DEMO =================
const DEMO_STEPS = [
  { page: "overview", focus: "#kpis", title: "1 · The problem", text: "AI is billed per token. The same sentence costs far more tokens in Indian languages: more money, less memory, slower replies. Measured on 1,012 identical sentences per language, 7 models." },
  { page: "overview", focus: "#gap-panel", title: "2 · The research gap", text: "Earlier work measures the tax, retrains tokenizers, or picks one format for a whole task. None checks, per message, that a cheaper rewrite still means the same thing, and none keeps an audit trail. That is what BhashaToken adds." },
  { page: "try", example: 0, focus: "#s-cost", title: "3 · Example 1: one Kannada sentence, 7 models", text: "Tokens, characters, cost per 1 lakh messages and the safe saving for every model, in one table. The same sentence can cost several times more on one model than another." },
  { page: "try", focus: "#s-tokens", title: "4 · Fragmentation Microscope", text: "Every box is a token. Red dashed boxes are broken letters: one Kannada letter split into several tokens. That is where the money goes." },
  { page: "try", example: 1, focus: "#s-cheaper", title: "5 · Example 2: real Kanglish, optimized safely", text: "Seven versions are generated and each is checked: the time 9:30, the English words and the negation must survive. The cheapest safe version is chosen per model, and the decision is recorded." },
  { page: "languages", compare: 123, focus: "#cmp-panel", title: "6 · Same sentence, 18 languages", text: "A professionally translated sentence in every language, side by side: characters, tokens, × English, rupees, and the saving after a safe rewrite. English is highlighted." },
  { page: "demos", memory: true, focus: "#d-memory", title: "7 · Memory race", text: "The same conversation typed by an English user and a Kannada user. Watch the Kannada user's AI forget earlier messages much sooner, and how BhashaToken gives back most of that memory." },
  { page: "safety", preset: 0, focus: "#lab-panel", title: "8 · Example 3: an unsafe rewrite is refused", text: "A medicine instruction loses its 'do not'. It is cheaper and about 90% similar, but the negation and word checks reject it. Similarity alone would have let it through." },
  { page: "research", focus: "#pareto-panel", title: "9 · Evidence: saving vs safety", text: "Every strategy on held-out data. Ours (diamond) saves 51% with 100% verified outputs; cutting text saves less and breaks meaning. Statistical tests are on this page." },
  { page: "simulator", focus: "#sim-kpis", title: "10 · Impact in rupees and memory", text: "For a state chatbot, measured rates become daily, monthly and yearly cost, and how much conversation memory each language loses. Everything is reproducible from the Evidence page." },
];

function demoGo(i) {
  const steps = DEMO_STEPS;
  S.demo = Math.max(0, Math.min(steps.length - 1, i));
  const st = steps[S.demo];
  showPage(st.page, true);
  if (st.example !== undefined) { const b = $("#examples").querySelectorAll("button")[DEMO_EXAMPLE_INDEX[st.example]]; if (b) b.click(); }
  if (st.memory) { MEM.data = null; if (MEM.timer) { clearInterval(MEM.timer); MEM.timer = null; } memPlay(); }
  if (st.compare !== undefined) { S.cmpLoaded = true; runCompare(st.compare); }
  if (st.preset !== undefined) { const b = $("#lab-presets").querySelectorAll("button")[st.preset]; if (b) b.click(); }
  document.querySelectorAll(".focus").forEach((x) => x.classList.remove("focus"));
  const el = $(st.focus);
  if (el) { el.classList.add("focus"); setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "start" }), 120); }
  $("#demo-title").textContent = st.title;
  $("#demo-text").textContent = st.text;
  $("#demo-count").textContent = `Step ${S.demo + 1} of ${steps.length}`;
  $("#demo-bar").style.width = `${((S.demo + 1) / steps.length) * 100}%`;
  $("#demo-prev").disabled = S.demo === 0;
  $("#demo-next").textContent = S.demo === steps.length - 1 ? "Finish" : "Next";
  $("#demo").hidden = false;
}
function demoClose() { $("#demo").hidden = true; S.demo = null; document.querySelectorAll(".focus").forEach((x) => x.classList.remove("focus")); }
const DEMO_EXAMPLE_INDEX = [0, 1];

// ================= wiring =================
const PAGES = ["overview", "try", "languages", "demos", "safety", "research", "simulator", "evidence"];
function showPage(id, fromDemo = false) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.page === id));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("on", p.id === id));
  if (!fromDemo) window.scrollTo({ top: 0 });
  if (id === "try" && !S.last) runTry();
  if (id === "evidence") renderAudit();
  if (id === "languages" && !S.cmpLoaded) { S.cmpLoaded = true; runCompare(); }
  crowdLive(id === "demos");
  Object.values(S.charts).forEach((c) => c.resize());
  history.replaceState(null, "", `#${id}`);
}

function renderAllCharts() {
  themeCharts();
  initOverviewCharts(); languageCharts(); safetyCharts(); initResearch(); runSim();
  if (S.last) { renderTryCost(); renderOpt(); renderPred(); renderWhy(); }
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("bt-theme", t); } catch (e) { /* storage unavailable */ }
  $("#theme-btn").textContent = t === "dark" ? "☀" : "☾";
}

function setMode(m) {
  document.body.classList.toggle("research", m === "research");
  document.querySelectorAll(".mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  try { localStorage.setItem("bt-mode", m); } catch (e) { /* storage unavailable */ }
  Object.values(S.charts).forEach((c) => c.resize());
}

async function init() {
  const names = ["benchmark", "fairness", "explainer", "predictor", "optimizer", "compression", "simulator",
    "verification", "strategies", "lfi_sensitivity", "efficiency", "ledger", "counterfactual"];
  const [meta, map, ...res] = await Promise.all([
    api("/api/meta"), fetch("/static/data/india_states.json").then((r) => r.json()),
    ...names.map((n) => api(`/api/results/${n}`)),
  ]);
  S.meta = meta; S.map = map;
  names.forEach((n, i) => { S.res[n] = res[i]; });
  document.querySelectorAll(".tabs button").forEach((b) => { b.onclick = () => showPage(b.dataset.page); });
  setTheme(document.documentElement.dataset.theme || "light");
  let mode = "simple"; try { mode = localStorage.getItem("bt-mode") || "simple"; } catch (e) { /* ignore */ }
  setMode(mode);
  initOverview(); initTry(); initLanguages(); initCompare(); initDemos(); initVoice(); initCrowd(); initReceipt(); localAiStatus(); $("#ai-ask").onclick = localAiAsk; initSafety(); initResearch(); initSimulator(); initHow();
  renderLedger();
  phoneBox($("#phone-box"));
  $("#phone-btn").onclick = () => { $("#sheet").hidden = false; phoneBox($("#sheet-body")); };
  $("#sheet-close").onclick = () => { $("#sheet").hidden = true; };
  $("#sheet").onclick = (e) => { if (e.target.id === "sheet") $("#sheet").hidden = true; };
  $("#theme-btn").onclick = () => { setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); renderAllCharts(); renderMap(); };
  document.querySelectorAll(".mode button").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
  $("#demo-btn").onclick = () => demoGo(0);
  $("#demo-next").onclick = () => (S.demo === DEMO_STEPS.length - 1 ? demoClose() : demoGo(S.demo + 1));
  $("#demo-prev").onclick = () => demoGo(S.demo - 1);
  $("#demo-close").onclick = demoClose;
  document.addEventListener("keydown", (e) => {
    if (S.demo === null) return;
    if (e.key === "ArrowRight") $("#demo-next").click();
    if (e.key === "ArrowLeft" && S.demo > 0) demoGo(S.demo - 1);
    if (e.key === "Escape") demoClose();
  });
  window.addEventListener("hashchange", () => {
    const h = location.hash.replace("#", "");
    if (PAGES.includes(h) && !document.getElementById(h).classList.contains("on")) showPage(h);
  });
  const start = location.hash.replace("#", "");
  if (start === "demo") demoGo(0);
  else if (PAGES.includes(start)) showPage(start);
}

init().catch((e) => { document.querySelector("main").innerHTML = `<div class="panel"><p class="err">Could not load results: ${esc(e.message)}. Run <code>python -m tokentax.experiments</code> first.</p></div>`; });
