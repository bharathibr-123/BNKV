"""Project paths. Everything is relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TOKENTAX_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = ROOT / "data_raw"
RESULTS = ROOT / "results"
TOKENIZERS = ROOT / "vendor" / "tokenizers"
CONFIG = ROOT / "config"
