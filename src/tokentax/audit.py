"""Audit ledger and human review store (SQLite, local file, standard library only).

Every optimisation the app performs is written here with its full decision
trace, so any result shown to a judge can be looked up again later. Reviewers
can rate a rewrite (meaning kept? fluent? would you send it?) and the ratings
can be exported as CSV for the human-evaluation protocol in docs/METHODOLOGY.md.

The database stays on this computer. Set ``BHASHATOKEN_AUDIT=off`` to disable it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tokentax.paths import ROOT

DB_PATH = Path(os.environ.get("BHASHATOKEN_DB", ROOT / "var" / "audit.sqlite"))
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    request_id TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    language TEXT,
    input_text TEXT NOT NULL,
    chosen_json TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    candidates_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crowd (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id TEXT NOT NULL,
    language TEXT NOT NULL,
    candidate TEXT NOT NULL,
    answer TEXT NOT NULL CHECK (answer IN ('same', 'partly', 'different')),
    knows_language INTEGER NOT NULL CHECK (knows_language IN (0, 1)),
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    candidate TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    meaning INTEGER NOT NULL CHECK (meaning BETWEEN 1 AND 5),
    fluency INTEGER NOT NULL CHECK (fluency BETWEEN 1 AND 5),
    acceptable INTEGER NOT NULL CHECK (acceptable IN (0, 1)),
    note TEXT,
    created TEXT NOT NULL
);
"""


def enabled() -> bool:
    return os.environ.get("BHASHATOKEN_AUDIT", "on").lower() != "off"


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        con = sqlite3.connect(DB_PATH)
        try:
            con.executescript(SCHEMA)
            yield con
            con.commit()
        finally:
            con.close()


def record(text: str, plan: dict[str, Any]) -> None:
    if not enabled():
        return
    trace = plan["trace"]
    with _db() as con:
        con.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trace["request_id"], trace["timestamp"], plan["language"], text,
             json.dumps(plan["choices"], ensure_ascii=False),
             json.dumps(trace, ensure_ascii=False),
             json.dumps([{k: c[k] for k in ("name", "text", "safe", "tokens", "reasons")}
                         for c in plan["candidates"]], ensure_ascii=False)),
        )


def recent(limit: int = 20) -> list[dict[str, Any]]:
    with _db() as con:
        rows = con.execute(
            "SELECT request_id, created, language, input_text, chosen_json, trace_json "
            "FROM decisions ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for rid, created, lang, text, chosen, trace in rows:
        t = json.loads(trace)
        out.append({"request_id": rid, "created": created, "language": lang,
                    "input": text[:160], "choices": json.loads(chosen),
                    "rejected": t.get("rejected", {}), "latency_ms": t.get("latency_ms")})
    return out


def get(request_id: str) -> dict[str, Any] | None:
    with _db() as con:
        row = con.execute("SELECT * FROM decisions WHERE request_id = ?",
                          (request_id,)).fetchone()
    if row is None:
        return None
    rid, created, lang, text, chosen, trace, cands = row
    return {"request_id": rid, "created": created, "language": lang, "input": text,
            "choices": json.loads(chosen), "trace": json.loads(trace),
            "candidates": json.loads(cands)}


def add_review(request_id: str, candidate: str, reviewer: str, meaning: int, fluency: int,
               acceptable: bool, note: str = "") -> int:
    with _db() as con:
        if con.execute("SELECT 1 FROM decisions WHERE request_id = ?",
                       (request_id,)).fetchone() is None:
            raise KeyError(request_id)
        cur = con.execute(
            "INSERT INTO reviews (request_id, candidate, reviewer, meaning, fluency, acceptable,"
            " note, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, candidate, reviewer[:60], meaning, fluency, int(acceptable), note[:500],
             datetime.now(UTC).isoformat(timespec="seconds")))
        return int(cur.lastrowid or 0)


def review_summary() -> dict[str, Any]:
    with _db() as con:
        rows = con.execute(
            "SELECT candidate, COUNT(*), AVG(meaning), AVG(fluency), AVG(acceptable) "
            "FROM reviews GROUP BY candidate ORDER BY candidate").fetchall()
        total = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        reviewers = con.execute("SELECT COUNT(DISTINCT reviewer) FROM reviews").fetchone()[0]
    return {"total": int(total), "reviewers": int(reviewers),
            "by_candidate": [{"candidate": c, "n": int(n), "meaning": m, "fluency": f,
                              "acceptable": a} for c, n, m, f, a in rows]}


def export_csv() -> str:
    with _db() as con:
        rows = con.execute(
            "SELECT r.request_id, d.language, r.candidate, d.input_text, r.reviewer, r.meaning,"
            " r.fluency, r.acceptable, r.note, r.created FROM reviews r JOIN decisions d"
            " USING (request_id) ORDER BY r.id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["request_id", "language", "candidate", "input", "reviewer", "meaning_1to5",
                "fluency_1to5", "acceptable", "note", "created"])
    w.writerows(rows)
    return buf.getvalue()


def add_crowd(pair_id: str, language: str, candidate: str, answer: str,
              knows_language: bool) -> int:
    with _db() as con:
        cur = con.execute(
            "INSERT INTO crowd (pair_id, language, candidate, answer, knows_language, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (pair_id[:40], language[:20], candidate[:40], answer, int(knows_language),
             datetime.now(UTC).isoformat(timespec="seconds")))
        return int(cur.lastrowid or 0)


def crowd_summary() -> dict[str, Any]:
    """Counts of crowd answers, only from raters who said they know the language."""
    with _db() as con:
        rows = con.execute(
            "SELECT language, answer, COUNT(*) FROM crowd WHERE knows_language = 1"
            " GROUP BY language, answer").fetchall()
        skipped = con.execute("SELECT COUNT(*) FROM crowd WHERE knows_language = 0").fetchone()[0]
    by_lang: dict[str, dict[str, int]] = {}
    for lang, answer, n in rows:
        by_lang.setdefault(lang, {"same": 0, "partly": 0, "different": 0})[answer] = int(n)
    total = {a: sum(v[a] for v in by_lang.values()) for a in ("same", "partly", "different")}
    n_all = sum(total.values())
    return {"ratings": n_all, "excluded_not_speakers": int(skipped), "total": total,
            "same_share": total["same"] / n_all if n_all else None,
            "by_language": [{"language": k, **v} for k, v in sorted(by_lang.items())]}


def crowd_csv() -> str:
    with _db() as con:
        rows = con.execute("SELECT pair_id, language, candidate, answer, knows_language, created"
                           " FROM crowd ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["pair_id", "language", "candidate", "answer", "knows_language", "created"])
    w.writerows(rows)
    return buf.getvalue()
