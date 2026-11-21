"""Ask a real AI the same question twice: once as typed and once as BhashaToken
would send it, and show both answers with the tokens the AI itself counted.

Two ways to connect (both optional):

1. Ollama on this laptop (default): install from ollama.com, then
   ``ollama pull gemma3:1b``. Offline and private.
2. Any OpenAI-compatible server, set by environment variables:
   * ``BHASHATOKEN_AI_URL``   for example ``http://127.0.0.1:8080/v1`` (llamafile, a
     single downloaded file, no installer) or
     ``https://generativelanguage.googleapis.com/v1beta/openai`` (Google Gemini API,
     free key from Google AI Studio, needs internet; text is sent to Google)
   * ``BHASHATOKEN_AI_KEY``   API key (any text for llamafile)
   * ``BHASHATOKEN_AI_MODEL`` model name

``BHASHATOKEN_OLLAMA_URL`` and ``BHASHATOKEN_OLLAMA_MODEL`` change the Ollama defaults.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

URL = os.environ.get("BHASHATOKEN_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("BHASHATOKEN_OLLAMA_MODEL", "gemma3:1b")
OPENAI_URL = os.environ.get("BHASHATOKEN_AI_URL", "").rstrip("/")
OPENAI_KEY = os.environ.get("BHASHATOKEN_AI_KEY", "no-key")
OPENAI_MODEL = os.environ.get("BHASHATOKEN_AI_MODEL", "local-model")
TIMEOUT = 180


def _system(language_name: str) -> str:
    return (f"You are a helpful assistant for people in India. Answer in {language_name}, "
            f"in its own script, in at most three short sentences.")


def _get(url: str, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _post(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def backend() -> str:
    return "openai" if OPENAI_URL else "ollama"


def status() -> dict[str, Any]:
    if backend() == "openai":
        where = "this laptop" if OPENAI_URL.startswith(("http://127.", "http://localhost")) \
            else "the internet"
        try:
            _get(OPENAI_URL + "/models", {"Authorization": f"Bearer {OPENAI_KEY}"})
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return {"running": True, "model_ready": False, "model": OPENAI_MODEL,
                        "backend": "openai", "where": where, "models": [],
                        "help": "The AI server refused the API key (BHASHATOKEN_AI_KEY)."}
            # some servers have no /models; the server answered, so treat it as ready
        except (urllib.error.URLError, OSError, ValueError):
            return {"running": False, "model_ready": False, "model": OPENAI_MODEL,
                    "backend": "openai", "where": where, "models": [],
                    "help": f"Cannot reach {OPENAI_URL}. Is the AI server running?"}
        return {"running": True, "model_ready": True, "model": OPENAI_MODEL,
                "backend": "openai", "where": where, "models": [OPENAI_MODEL], "help": ""}
    try:
        tags = _get(URL + "/api/tags")
    except (urllib.error.URLError, OSError, ValueError):
        return {"running": False, "model": MODEL, "model_ready": False, "models": [],
                "backend": "ollama", "where": "this laptop",
                "help": "Install Ollama from ollama.com and keep it running."}
    names = [m.get("name", "") for m in tags.get("models", [])]
    ready = MODEL in names or f"{MODEL}:latest" in names
    return {"running": True, "model": MODEL, "model_ready": ready, "models": names,
            "backend": "ollama", "where": "this laptop",
            "help": "" if ready else f"Run: ollama pull {MODEL}"}


def ask(question: str, language_name: str) -> dict[str, Any]:
    t = time.perf_counter()
    if backend() == "openai":
        out = _post(OPENAI_URL + "/chat/completions", {
            "model": OPENAI_MODEL, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": _system(language_name)},
                         {"role": "user", "content": question}],
        }, {"Authorization": f"Bearer {OPENAI_KEY}"})
        usage = out.get("usage") or {}
        answer = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {"answer": str(answer).strip(),
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
                "seconds": round(time.perf_counter() - t, 1)}
    out = _post(URL + "/api/generate", {
        "model": MODEL, "system": _system(language_name), "prompt": question, "stream": False,
        "options": {"temperature": 0, "num_predict": 160, "seed": 7},
    })
    return {"answer": str(out.get("response", "")).strip(),
            "input_tokens": int(out.get("prompt_eval_count", 0)),
            "output_tokens": int(out.get("eval_count", 0)),
            "seconds": round(time.perf_counter() - t, 1)}
