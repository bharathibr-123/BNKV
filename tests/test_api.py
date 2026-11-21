"""API tests with FastAPI's test client."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import app
from tokentax import registry
from tokentax.paths import RESULTS

pytestmark = pytest.mark.skipif(not registry.available(), reason="tokenizer files missing")
client = TestClient(app)
KANNADA = "ನಾನು ನಾಳೆ ಕಾಲೇಜಿಗೆ ಹೋಗುತ್ತೇನೆ."
needs_results = pytest.mark.skipif(not (RESULTS / "benchmark.json").exists(),
                                   reason="run python -m tokentax.experiments first")


def test_index_and_meta() -> None:
    assert client.get("/").status_code == 200
    meta = client.get("/api/meta").json()
    assert {t["key"] for t in meta["tokenizers"]} >= {"llama3", "gpt4o"}


@pytest.mark.parametrize("path", ["analyze", "explain", "optimize", "compress", "route"])
def test_text_endpoints(path: str) -> None:
    r = client.post(f"/api/{path}", json={"text": KANNADA})
    assert r.status_code == 200, r.text


def test_analyze_flags_byte_pieces() -> None:
    rows = client.post("/api/analyze", json={"text": "ನಾನು"}).json()["rows"]
    llama = next(r for r in rows if r["tokenizer"] == "llama3")
    assert all(llama["byte_flags"])


def test_rejects_bad_input() -> None:
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422
    assert client.post("/api/analyze", json={"text": "x" * 6000}).status_code == 422
    assert client.post("/api/explain", json={"text": "a", "tokenizer": "nope"}).status_code == 400
    assert client.get("/api/results/../../etc").status_code == 404


def test_chat_proxy_is_dry_run_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENTAX_UPSTREAM", raising=False)
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "system", "content": "be brief"},
                                       {"role": "user", "content": KANNADA}]})
    body = r.json()
    assert body["dry_run"] is True
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert len(body["routing"]) == 1


@needs_results
def test_simulate_and_predict() -> None:
    sim = client.post("/api/simulate", json={"model": "gpt4o", "language": "kan_Knda"}).json()
    assert sim["extra_annual_inr"] > 0
    assert sim["saved_by_optimizer_annual_inr"] >= 0
    pred = client.post("/api/predict", json={"text": KANNADA}).json()
    for row in pred["rows"]:
        assert abs(row["predicted"] - row["actual"]) / row["actual"] < 0.5


@needs_results
def test_results_endpoint() -> None:
    assert client.get("/api/results/fairness").status_code == 200
    assert client.get("/api/results/secret").status_code == 404


def test_phone_and_manifest() -> None:
    m = client.get("/manifest.webmanifest")
    assert m.status_code == 200 and m.json()["display"] == "standalone"
    phone = client.get("/api/phone").json()
    assert set(phone) == {"url", "svg"}


def test_static_assets() -> None:
    for path in ("/static/vendor/chart.umd.min.js", "/static/data/india_states.json",
                 "/static/app.js", "/static/icon.svg"):
        assert client.get(path).status_code == 200, path


def test_verify_endpoint_rejects_dropped_negation() -> None:
    r = client.post("/api/verify", json={
        "original": "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.",
        "candidate": "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ."}).json()
    assert r["passed"] is False and r["language"] == "kan_Knda"
    assert any(x.startswith("negation") for x in r["reasons"])


def test_optimize_is_audited_and_reviewable() -> None:
    plan = client.post("/api/optimize", json={"text": KANNADA}).json()
    rid = plan["trace"]["request_id"]
    assert plan["trace"]["audited"] is True
    assert client.get(f"/api/audit/{rid}").json()["input"] == KANNADA
    assert any(i["request_id"] == rid for i in client.get("/api/audit").json()["items"])
    ok = client.post("/api/review", json={"request_id": rid, "candidate": "word_best",
                                          "reviewer": "judge", "meaning": 5, "fluency": 4,
                                          "acceptable": True})
    assert ok.status_code == 200 and ok.json()["summary"]["total"] == 1
    bad = client.post("/api/review", json={"request_id": rid, "candidate": "x", "reviewer": "j",
                                           "meaning": 9, "fluency": 4, "acceptable": True})
    assert bad.status_code == 422
    assert client.post("/api/review", json={"request_id": "nope-nope", "candidate": "x",
                                            "reviewer": "j", "meaning": 3, "fluency": 3,
                                            "acceptable": False}).status_code == 404
    csv = client.get("/api/reviews.csv")
    assert csv.status_code == 200 and "judge" in csv.text
    assert client.get("/api/audit/doesnotexist").status_code == 404


def test_demo_examples() -> None:
    ex = client.get("/api/demo").json()["examples"]
    assert len(ex) == 3 and any("unsafe" in e for e in ex)


@needs_results
def test_simulate_custom_price_and_context() -> None:
    sim = client.post("/api/simulate", json={"model": "gpt4o", "language": "kan_Knda",
                                             "input_price_usd": 0.5, "output_price_usd": 1.5,
                                             "context_window": 32000}).json()
    assert sim["price_input_usd"] == 0.5
    assert sim["messages_in_context_optimized"] >= sim["messages_in_context_native"]
    assert sim["annual_inr"] > 0
    assert client.post("/api/simulate", json={"input_price_usd": -1}).status_code == 422


def test_compare_same_sentence_all_languages() -> None:
    d = client.post("/api/compare", json={"index": 123, "tokenizer": "gpt4o"}).json()
    assert len(d["rows"]) == 18
    eng = next(r for r in d["rows"] if r["code"] == "eng_Latn")
    assert eng["x_english"] == 1.0 and eng["tokens"] == d["english_tokens"]
    for r in d["rows"]:
        assert r["characters"] == len(r["text"])
        assert r["best_tokens"] <= r["tokens"]
        assert r["cost_inr_per_lakh"] > 0


def test_compare_by_exact_text_and_search() -> None:
    eng = client.post("/api/compare", json={"index": 5}).json()["rows"][0]["text"]
    kan = next(r["text"] for r in client.post("/api/compare", json={"index": 5}).json()["rows"]
               if r["code"] == "kan_Knda")
    assert client.post("/api/compare", json={"text": kan}).json()["index"] == 5
    assert client.post("/api/compare", json={"text": eng}).json()["index"] == 5
    assert client.post("/api/compare", json={"text": "not a flores sentence"}).status_code == 404
    assert client.post("/api/compare", json={"index": 5000}).status_code == 422
    assert client.post("/api/compare", json={"index": 1, "tokenizer": "nope"}).status_code == 400
    hits = client.get("/api/compare/search", params={"q": "ಭೂಕಂಪ"}).json()["results"]
    assert any(h["index"] == 181 for h in hits)
    assert len(client.get("/api/compare/featured").json()["featured"]) == 6


def test_demo_memory_race() -> None:
    d = client.post("/api/demo/conversation",
                    json={"language": "kan_Knda", "tokenizer": "llama3", "turns": 10}).json()
    assert len(d["turns"]) == 10
    for t in d["turns"]:
        assert t["tokens_optimized"] <= t["tokens_native"]
    native = sum(t["tokens_native"] for t in d["turns"])
    assert native > sum(t["tokens_english"] for t in d["turns"])
    assert client.post("/api/demo/conversation", json={"language": "xx"}).status_code == 400
    assert client.post("/api/demo/conversation", json={"turns": 500}).status_code == 422


def test_demo_batch_audit() -> None:
    d = client.post("/api/demo/batch", json={
        "text": "ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಕಚೇರಿ ತೆರೆಯುತ್ತದೆಯೇ?\n\nhello there\nमेरी फसल खराब हो गई",
        "tokenizer": "llama3"}).json()
    t = d["totals"]
    assert t["messages"] == 3 and t["after"] <= t["tokens"]
    assert {"kan_Knda", "eng_Latn", "hin_Deva"} <= set(t["languages"])
    assert client.post("/api/demo/batch", json={"text": "x", "tokenizer": "bad"}).status_code == 400


def test_demo_fool_the_checker() -> None:
    med = "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ."
    neg = client.post("/api/demo/mutate", json={"text": med, "kind": "negation"}).json()
    assert neg["text"] == "ಈ ಔಷಧಿಯನ್ನು ಖಾಲಿ ಹೊಟ್ಟೆಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ."
    assert neg["caught"] and neg["correct"]
    rom = client.post("/api/demo/mutate", json={"text": med, "kind": "roman"}).json()
    assert not rom["caught"] and rom["correct"]
    num = client.post("/api/demo/mutate", json={"text": med, "kind": "number"}).json()
    assert num["applied"] is False and num["correct"] is None
    assert client.post("/api/demo/mutate", json={"text": med, "kind": "nope"}).status_code == 400
    assert len(client.get("/api/demo/mutations").json()["mutations"]) == 8


def test_crowd_check_flow() -> None:
    pair = client.get("/api/crowd/pair", params={"language": "kan_Knda", "seed": 1}).json()
    assert pair["rewrite"] != pair["original"] and pair["tokens_after"] <= pair["tokens_before"]
    body = {"pair_id": pair["pair_id"], "language": "kan_Knda", "candidate": pair["candidate"]}
    client.post("/api/crowd/rate", json={**body, "answer": "same", "knows_language": True})
    s = client.post("/api/crowd/rate",
                    json={**body, "answer": "different", "knows_language": False}).json()
    assert s["ratings"] == 1 and s["excluded_not_speakers"] == 1 and s["same_share"] == 1.0
    assert client.post("/api/crowd/rate",
                       json={**body, "answer": "maybe", "knows_language": True}).status_code == 422
    assert client.get("/api/crowd/pair", params={"language": "eng_Latn"}).status_code == 400
    assert "kan_Knda" in client.get("/api/crowd.csv").text
    assert client.get("/rate").status_code == 200


def test_fixed_pretokenizer_counterfactual() -> None:
    from tokentax import registry

    word = "ಬೆಂಗಳೂರಿಗೆ"
    assert len(registry.pretokens("llama3", word)) > len(registry.pretokens("llama3", word, True))
    # same vocabulary, so fixing the split alone does not change the count here
    assert registry.count_fixed("llama3", word) == registry.count("llama3", word)
    rows = client.post("/api/analyze", json={"text": word}).json()["rows"]
    assert all("tokens_fixed_pretokenizer" in r for r in rows)


def test_dashboard_javascript_parses() -> None:
    """Catch syntax errors in the dashboard script (skipped if Node.js is missing)."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    for name in ("app.js",):
        path = Path(__file__).parent.parent / "app" / "static" / name
        res = subprocess.run([node, "--check", str(path)], capture_output=True, text=True,
                             check=False)
        assert res.returncode == 0, res.stderr


def test_local_ai_unavailable_and_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokentax import localai

    monkeypatch.setattr(localai, "URL", "http://127.0.0.1:9")  # nothing listens here
    assert client.get("/api/localai/status").json()["running"] is False
    body = {"original": "ನಾನು ನಾಳೆ ಬರುತ್ತೇನೆ.", "rewrite": "naanu naale baruttene."}
    assert client.post("/api/localai/answer", json=body).status_code == 503

    monkeypatch.setattr(localai, "status", lambda: {"running": True, "model_ready": True,
                                                    "model": "fake", "help": ""})
    monkeypatch.setattr(localai, "ask", lambda q, lang: {"answer": "ಸರಿ, ನಾಳೆ ನೋಡೋಣ.",
                                                         "input_tokens": len(q) // 3,
                                                         "output_tokens": 5, "seconds": 0.1})
    d = client.post("/api/localai/answer", json=body).json()
    assert d["language"] == "kan_Knda" and d["answer_similarity"] == pytest.approx(1.0)


def test_optional_sarvam_tokenizer_is_picked_up() -> None:
    import shutil

    from tokentax import optimizer

    src = registry.BY_KEY["llama3"].path
    dst = registry.BY_KEY["sarvam1"].path
    assert not dst.exists(), "remove the real Sarvam file before running this test"
    shutil.copy(src, dst)  # stand-in with the right file name
    try:
        assert "sarvam1" in [s.key for s in registry.available()]
        meta = client.get("/api/meta").json()
        assert {"key": "sarvam1", "measured": False}.items() <= next(
            t for t in meta["tokenizers"] if t["key"] == "sarvam1").items()
        rows = client.post("/api/analyze", json={"text": "ನಾನು ನಾಳೆ ಬರುತ್ತೇನೆ"}).json()["rows"]
        assert any(r["tokenizer"] == "sarvam1" for r in rows)
        assert client.post("/api/predict", json={"text": "ನಾನು"}).status_code == 200
        assert client.post("/api/route", json={"text": "ನಾನು ನಾಳೆ ಬರುತ್ತೇನೆ"}).status_code == 200
    finally:
        dst.unlink()
        registry.load.cache_clear()
        optimizer.tokenizer_hash.cache_clear()


def test_handout_page() -> None:
    assert "BhashaToken" in client.get("/handout").text


def test_ai_check_openai_compatible_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokentax import localai

    monkeypatch.setattr(localai, "OPENAI_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(localai, "OPENAI_MODEL", "gemma-3-1b")
    monkeypatch.setattr(localai, "_get", lambda url, headers=None: {"data": []})
    monkeypatch.setattr(localai, "_post", lambda url, body, headers=None: {
        "choices": [{"message": {"content": "ಸರಿ."}}],
        "usage": {"prompt_tokens": len(body["messages"][-1]["content"]),
                  "completion_tokens": 2}})
    st = client.get("/api/localai/status").json()
    assert st["backend"] == "openai" and st["model_ready"] and st["where"] == "this laptop"
    d = client.post("/api/localai/answer", json={"original": "ನಾಳೆ ಬನ್ನಿ.",
                                                 "rewrite": "naale banni."}).json()
    assert d["model"] == "gemma-3-1b" and d["original"]["answer"] == "ಸರಿ."
