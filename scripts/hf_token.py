from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_hf_token() -> str | None:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    env_file = project_root() / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def ensure_hf_token_in_env() -> str | None:
    token = load_hf_token()
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    return token
