"""Download Sarvam-1's tokenizer so BhashaToken can include it.

    python scripts/get_sarvam.py

Sarvam-1 is an open model from Sarvam AI built for Indian languages, under the
Sarvam non-commercial licence (fine for research and this hackathon). Only the
tokenizer file (about 8.5 MB) is downloaded, not the model.

If Hugging Face asks for a login, accept the licence on
https://huggingface.co/sarvamai/sarvam-1, create a token, and run again with
HF_TOKEN set (PowerShell: $env:HF_TOKEN="hf_...").
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/sarvamai/sarvam-1/resolve/main/tokenizer.json"
OUT = Path(__file__).resolve().parent.parent / "vendor" / "tokenizers" / "sarvam1.json"


def main() -> int:
    req = urllib.request.Request(URL, headers={"User-Agent": "bhashatoken"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as exc:
        print(f"Download failed: HTTP {exc.code}. See the instructions at the top of this file.")
        return 1
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc.reason}. Check your internet connection.")
        return 1
    json.loads(data)  # make sure it is a real tokenizer.json
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(data.decode("utf-8"))
    sample = "ನಾನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಹೋಗುತ್ತೇನೆ."
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"Saved {OUT} ({len(data) / 1e6:.1f} MB).")
    print(f"Test: '{sample}' -> {len(tok.encode(sample, add_special_tokens=False).ids)} tokens.")
    print("Restart the server; Sarvam-1 now appears in the Optimizer, Compare and Live Demos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
