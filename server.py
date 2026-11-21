"""BhashaToken: API and dashboard.

Run:  PYTHONPATH=src uvicorn app.server:app --port 8000
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import threading
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tokentax import (
    audit,
    compare,
    compression,
    demos,
    experiments,
    explainer,
    localai,
    metrics,
    optimizer,
    predictor,
    pricing,
    registry,
    router,
    simulator,
    verify,
)
from tokentax.languages import SOUTH_ASIAN
from tokentax.languages import get as get_language
from tokentax.normalize import canonical_words
from tokentax.paths import RESULTS
from tokentax.scripts import profile

STATIC = Path(__file__).parent / "static"
MAX_CHARS = 5000
RESULT_NAMES = {"benchmark", "fairness", "explainer", "predictor", "optimizer",
                "compression", "simulator", "verification", "strategies", "lfi_sensitivity",
                "efficiency", "ledger", "thresholds", "counterfactual"}

READY = threading.Event()


def _warm() -> None:
    """Load the tokenizers and the Kannada/Hindi spelling indexes first (what the
    Optimizer page needs), then the rest, so the first demo request is fast."""
    with contextlib.suppress(OSError, KeyError, ValueError):
        for spec in registry.available():
            registry.load(spec.key)
        for text, lang in (("ನಾನು ನಾಳೆ ಬರುತ್ತೇನೆ.", "kan_Knda"), ("मैं कल आऊँगा।", "hin_Deva")):
            optimizer.plan(text, language=lang)
    READY.set()
    print("BhashaToken is ready: open http://127.0.0.1:8000", flush=True)
    with contextlib.suppress(OSError, KeyError, ValueError):
        compare.compare(compare.FEATURED[0][0], "gpt4o")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="BhashaToken", lifespan=lifespan, version="3.0.0",
              description="Measure the Tax. Preserve the Meaning. Make AI More Inclusive.")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    language: str | None = None


class ExplainIn(TextIn):
    tokenizer: str = "llama3"


class CompressIn(TextIn):
    aggressive: bool = False


class SimulateIn(BaseModel):
    daily_users: int = Field(100_000, ge=1, le=100_000_000)
    messages_per_user: float = Field(4.0, gt=0, le=1000)
    regional_share: float = Field(0.7, ge=0, le=1)
    input_words: float = Field(40, gt=0, le=10_000)
    output_words: float = Field(120, gt=0, le=10_000)
    model: str = "gpt4o"
    language: str = "kan_Knda"
    input_price_usd: float | None = Field(None, ge=0, le=1000)
    output_price_usd: float | None = Field(None, ge=0, le=1000)
    context_window: int | None = Field(None, ge=1000, le=10_000_000)
    reply_optimized: bool = False


class VerifyIn(BaseModel):
    original: str = Field(min_length=1, max_length=MAX_CHARS)
    candidate: str = Field(min_length=1, max_length=MAX_CHARS)
    language: str | None = None


class ReviewIn(BaseModel):
    request_id: str = Field(min_length=6, max_length=40)
    candidate: str = Field(min_length=1, max_length=40)
    reviewer: str = Field(min_length=1, max_length=60)
    meaning: int = Field(ge=1, le=5)
    fluency: int = Field(ge=1, le=5)
    acceptable: bool
    note: str = Field("", max_length=500)


def _keys() -> list[str]:
    return [s.key for s in registry.available()]


def _check_key(key: str) -> None:
    if key not in registry.BY_KEY:
        raise HTTPException(400, f"unknown tokenizer: {key}")


@lru_cache(maxsize=1)
def _predictors() -> dict[str, predictor.TokenPredictor]:
    path = RESULTS / "predictors.json"
    return predictor.load(path) if path.exists() else {}


def _measured() -> set[str]:
    """Tokenizers that appear in the saved experiment results."""
    path = RESULTS / "benchmark.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["tokenizer"] for row in data.get("flores", [])}


def _result(name: str) -> Any:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"run `python -m tokentax.experiments` first ({name})")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/manifest.webmanifest")
def manifest() -> JSONResponse:
    """Lets phones install the dashboard from the browser ("Add to Home Screen")."""
    return JSONResponse({
        "name": "BhashaToken", "short_name": "BhashaToken",
        "start_url": "/", "display": "standalone",
        "background_color": "#fcfcfa", "theme_color": "#1e2a44",
        "icons": [{"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
    }, media_type="application/manifest+json")


def _lan_ip() -> str | None:
    """This computer's address on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            ip = str(sock.getsockname()[0])
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


@app.get("/api/phone")
def phone(request: Request, path: str = "") -> dict[str, Any]:
    """A QR code that opens this dashboard (or ``/rate``) on a phone on the same Wi-Fi."""
    port = request.url.port or 8000
    ip = _lan_ip()
    suffix = "/rate" if path == "rate" else ""
    url = f"http://{ip}:{port}{suffix}" if ip else None
    svg = None
    if url:
        try:
            import qrcode
            import qrcode.image.svg
        except ImportError:
            svg = None
        else:
            buf = io.BytesIO()
            qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage).save(buf)
            svg = buf.getvalue().decode("utf-8")
    return {"url": url, "svg": svg}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    book = pricing.load()
    return {
        "tokenizers": [{"key": s.key, "label": s.label, "vendor": s.vendor,
                        "open_weights": s.open_weights, "measured": s.key in _measured()}
                       for s in registry.available()],
        "languages": [{"code": lang.code, "name": lang.name, "script": lang.script.name,
                       "family": lang.family} for lang in SOUTH_ASIAN],
        "usd_to_inr": book.usd_to_inr,
        "context_window": book.context_window,
        "input_price_usd": {k: p.input for k, p in book.models.items()},
    }


@app.post("/api/analyze")
def analyze(body: TextIn) -> dict[str, Any]:
    book = pricing.load()
    prof = profile(body.text)
    rows = []
    for key in _keys():
        enc = registry.load(key).encode(body.text, add_special_tokens=False)
        pieces = [body.text[a:b] for a, b in enc.offsets]
        rows.append({
            "tokenizer": key,
            "label": registry.BY_KEY[key].label,
            "tokens": len(enc.ids),
            "pieces": pieces,
            "byte_flags": [metrics.is_fragment(key, i) for i in enc.ids],
            "tokens_fixed_pretokenizer": registry.count_fixed(key, body.text),
            "cost_inr_per_1000": book.cost_inr(key, len(enc.ids) * 1000),
            "context_share": len(enc.ids) / book.context_window,
        })
    return {
        "language": body.language or explainer.guess_language(body.text),
        "style": prof.style.value,
        "chars": len(body.text),
        "words": len(body.text.split()),
        "rows": rows,
    }


@app.post("/api/explain")
def explain(body: ExplainIn) -> dict[str, Any]:
    _check_key(body.tokenizer)
    return explainer.explain(body.tokenizer, body.text, body.language).to_dict()


@app.post("/api/optimize")
def optimize(body: TextIn) -> dict[str, Any]:
    plan = optimizer.plan(body.text, keys=_keys(), language=body.language).to_dict()
    try:
        audit.record(body.text, plan)
        plan["trace"]["audited"] = audit.enabled()  # type: ignore[index]
    except OSError:
        plan["trace"]["audited"] = False  # type: ignore[index]
    return plan


@app.post("/api/verify")
def verify_pair(body: VerifyIn) -> dict[str, Any]:
    lang = body.language or explainer.guess_language(body.original)
    return {"language": lang, **verify.verify(body.original, body.candidate, lang).to_dict()}


@app.get("/api/audit")
def audit_recent(limit: int = 20) -> dict[str, Any]:
    return {"enabled": audit.enabled(), "items": audit.recent(max(1, min(limit, 200)))}


@app.get("/api/audit/{request_id}")
def audit_one(request_id: str) -> dict[str, Any]:
    item = audit.get(request_id)
    if item is None:
        raise HTTPException(404, "unknown request id")
    return item


@app.post("/api/review")
def review(body: ReviewIn) -> dict[str, Any]:
    try:
        rid = audit.add_review(body.request_id, body.candidate, body.reviewer, body.meaning,
                               body.fluency, body.acceptable, body.note)
    except KeyError as exc:
        raise HTTPException(404, "unknown request id") from exc
    return {"id": rid, "summary": audit.review_summary()}


@app.get("/api/reviews")
def reviews() -> dict[str, Any]:
    return audit.review_summary()


@app.get("/api/reviews.csv")
def reviews_csv() -> PlainTextResponse:
    return PlainTextResponse(audit.export_csv(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=bhashatoken_reviews.csv"})


class CompareIn(BaseModel):
    index: int | None = Field(None, ge=0, le=1011)
    text: str | None = Field(None, max_length=MAX_CHARS)
    tokenizer: str = "gpt4o"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ready": READY.is_set()}


@app.get("/api/compare/featured")
def compare_featured() -> dict[str, Any]:
    return {"featured": compare.featured()}


@app.get("/api/compare/search")
def compare_search(q: str = "") -> dict[str, Any]:
    return {"results": compare.search(q[:200])}


@app.post("/api/compare")
def compare_languages(body: CompareIn) -> dict[str, Any]:
    _check_key(body.tokenizer)
    index = body.index
    if index is None and body.text:
        hit = compare.find(body.text)
        if hit is None:
            raise HTTPException(404, "This exact sentence is not in the parallel set. "
                                     "Search by a few words instead.")
        index = hit[0]
    if index is None:
        raise HTTPException(422, "give an index or a text")
    return compare.compare(index, body.tokenizer)


class ConversationIn(BaseModel):
    language: str = "kan_Knda"
    tokenizer: str = "llama3"
    start: int = Field(0, ge=0, le=1011)
    turns: int = Field(40, ge=1, le=demos.MAX_TURNS)


class BatchIn(BaseModel):
    text: str = Field(min_length=1, max_length=demos.MAX_LINES * (demos.MAX_LINE_CHARS + 2))
    tokenizer: str = "gpt4o"


class MutateIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    kind: str
    language: str | None = None
    seed: int = 0


@app.post("/api/demo/conversation")
def demo_conversation(body: ConversationIn) -> dict[str, Any]:
    _check_key(body.tokenizer)
    if body.language not in {lang.code for lang in SOUTH_ASIAN}:
        raise HTTPException(400, f"unknown language: {body.language}")
    return demos.conversation(body.language, body.tokenizer, body.start, body.turns)


@app.post("/api/demo/batch")
def demo_batch(body: BatchIn) -> dict[str, Any]:
    _check_key(body.tokenizer)
    return demos.batch(body.text, body.tokenizer)


@app.get("/api/demo/mutations")
def demo_mutations() -> dict[str, Any]:
    return {"mutations": [{"kind": k, "label": v[0], "safe": v[1]}
                          for k, v in demos.MUTATIONS.items()]}


@app.post("/api/demo/mutate")
def demo_mutate(body: MutateIn) -> dict[str, Any]:
    try:
        out = demos.mutate(body.text, body.kind, body.language, body.seed)
    except KeyError as exc:
        raise HTTPException(400, f"unknown mutation: {body.kind}") from exc
    v = verify.verify(body.text, out["text"], out["language"])
    out["verification"] = v.to_dict()
    out["caught"] = not v.passed
    out["correct"] = (not v.passed) != out["expected_safe"] if out["applied"] else None
    return out


class CrowdIn(BaseModel):
    pair_id: str = Field(min_length=3, max_length=40)
    language: str = Field(min_length=3, max_length=20)
    candidate: str = Field(min_length=1, max_length=40)
    answer: str = Field(pattern="^(same|partly|different)$")
    knows_language: bool


class AnswerIn(BaseModel):
    original: str = Field(min_length=1, max_length=2000)
    rewrite: str = Field(min_length=1, max_length=2000)
    language: str | None = None


@app.get("/api/localai/status")
def localai_status() -> dict[str, Any]:
    return localai.status()


@app.post("/api/localai/answer")
def localai_answer(body: AnswerIn) -> dict[str, Any]:
    st = localai.status()
    if not st["running"] or not st["model_ready"]:
        raise HTTPException(503, st["help"] or "local AI not available")
    code = body.language or explainer.guess_language(body.original)
    name = get_language(code).name
    try:
        a = localai.ask(body.original, name)
        b = localai.ask(body.rewrite, name)
    except (OSError, ValueError) as exc:
        raise HTTPException(502, f"local AI error: {exc}") from exc
    lang = get_language(code)
    similarity = verify.chrf(" ".join(canonical_words(a["answer"], lang)),
                             " ".join(canonical_words(b["answer"], lang)))
    return {"model": st["model"], "language": code, "original": a, "rewrite": b,
            "answer_similarity": similarity}


@app.get("/handout")
def handout_page() -> FileResponse:
    """One-page printable summary for judges."""
    return FileResponse(STATIC / "handout.html")


@app.get("/rate")
def rate_page() -> FileResponse:
    """Phone page for the live crowd check."""
    return FileResponse(STATIC / "rate.html")


@app.get("/api/crowd/pair")
def crowd_pair(language: str = "kan_Knda", seed: int = 0) -> dict[str, Any]:
    if language not in demos.CROWD_LANGS:
        raise HTTPException(400, f"crowd check languages: {', '.join(demos.CROWD_LANGS)}")
    try:
        return demos.crowd_pair(language, seed)
    except LookupError as exc:
        raise HTTPException(404, "no rewritten sentence found") from exc


@app.post("/api/crowd/rate")
def crowd_rate(body: CrowdIn) -> dict[str, Any]:
    if body.language not in demos.CROWD_LANGS:
        raise HTTPException(400, "unknown language")
    audit.add_crowd(body.pair_id, body.language, body.candidate, body.answer,
                    body.knows_language)
    return audit.crowd_summary()


@app.get("/api/crowd/summary")
def crowd_summary() -> dict[str, Any]:
    return audit.crowd_summary()


@app.get("/api/crowd.csv")
def crowd_csv() -> PlainTextResponse:
    return PlainTextResponse(audit.crowd_csv(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=bhashatoken_crowd.csv"})


DEMO = [
    {"id": "save", "title": "A Kannada request that gets cheaper, safely",
     "text": "ನಾನು ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ. ಟಿಕೆಟ್ ಬೆಲೆ ಎಷ್ಟು?",
     "language": "kan_Knda"},
    {"id": "codemix", "title": "Real code-mixed chat (Kannada + English)",
     "text": "ನಾಳೆ meeting ಇದೆ, please 9:30 ಗೆ ಬನ್ನಿ. Report ಮರೆಯಬೇಡಿ.",
     "language": "kan_Knda"},
    {"id": "reject", "title": "An unsafe rewrite the system refuses",
     "text": "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.",
     "language": "kan_Knda",
     "unsafe": "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ."},
]


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    """Three fixed examples for Judge Demo Mode (results are computed live)."""
    return {"examples": DEMO}


@app.post("/api/compress")
def compress(body: CompressIn) -> dict[str, Any]:
    return compression.compress(body.text, _keys(), body.language,
                                aggressive=body.aggressive).to_dict()


_WORD = re.compile(r"\S+")


@app.post("/api/predict")
def predict(body: TextIn) -> dict[str, Any]:
    models = _predictors()
    if not models:
        raise HTTPException(404, "run `python -m tokentax.experiments` first (predictor)")
    book = pricing.load()
    rows = []
    for key, model in models.items():
        guess = model.predict(body.text)
        words = _WORD.findall(body.text)
        costly = sorted(
            ({"word": w, "predicted_tokens": model.predict(" " + w)} for w in set(words)),
            key=lambda d: d["predicted_tokens"], reverse=True,
        )[:5]
        rows.append({
            "tokenizer": key,
            "predicted": guess,
            "actual": registry.count(key, body.text),
            "chars_over_4": predictor.chars_over_four(body.text),
            "cost_inr_per_1000": book.cost_inr(key, guess * 1000),
            "context_share": guess / book.context_window,
            "expensive_words": costly,
        })
    return {"rows": rows}


@app.post("/api/simulate")
def simulate(body: SimulateIn) -> dict[str, Any]:
    _check_key(body.model)
    bench, opt, comp = _result("benchmark"), _result("optimizer"), _result("compression")
    try:
        rates = experiments.rates_for(body.model, body.language, bench, opt, comp)
    except KeyError as exc:
        raise HTTPException(400, f"no measurements for {body.model} / {body.language}") from exc
    scen = simulator.Scenario(**body.model_dump())
    res = simulator.run(scen, rates, pricing.load())
    return {"rates": rates.__dict__, **res.to_dict()}


@app.post("/api/route")
def route(body: TextIn) -> dict[str, Any]:
    return router.route(body.text, language=body.language).to_dict()


@app.get("/api/results/{name}")
def results(name: str) -> JSONResponse:
    if name not in RESULT_NAMES:
        raise HTTPException(404, "unknown result")
    return JSONResponse(_result(name))


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=MAX_CHARS)


class ChatIn(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)


@app.post("/v1/chat/completions")
def chat_proxy(body: ChatIn) -> dict[str, Any]:
    """OpenAI-compatible endpoint that rewrites each user message into its cheapest
    safe format and picks the cheapest model.

    Without ``TOKENTAX_UPSTREAM`` set it runs as a dry run and returns the plan,
    so no message ever leaves the machine by accident.
    """
    rewritten: list[dict[str, str]] = []
    decisions: list[router.Decision] = []
    for m in body.messages:
        if m.role != "user":
            rewritten.append(m.model_dump())
            continue
        d = router.route(m.content, models=[body.model] if body.model else None)
        decisions.append(d)
        rewritten.append({"role": m.role, "content": d.best.text})
    model = decisions[-1].best.model if decisions else body.model
    upstream = os.environ.get("TOKENTAX_UPSTREAM")
    plan: dict[str, Any] = {"dry_run": upstream is None, "model": model,
                            "messages": rewritten, "routing": [d.to_dict() for d in decisions]}
    if upstream is None:
        return plan
    # Forward to an OpenAI-compatible server. The upstream model name is taken
    # from TOKENTAX_UPSTREAM_MODEL so a tokenizer key is never sent as a model id.
    payload = {"model": os.environ.get("TOKENTAX_UPSTREAM_MODEL", model), "messages": rewritten}
    req = urllib.request.Request(
        upstream.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('TOKENTAX_API_KEY', '')}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            answer: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except OSError as exc:
        raise HTTPException(502, f"upstream error: {exc}") from exc
    answer["tokentax"] = plan
    return answer
