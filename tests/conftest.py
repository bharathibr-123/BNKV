"""Keep tests from writing to the real audit ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokentax import audit


@pytest.fixture(autouse=True)
def _temp_audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "DB_PATH", tmp_path / "audit.sqlite")
