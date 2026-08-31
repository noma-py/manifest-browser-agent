"""Env-var + storage-state loading. Fails fast with a clear message."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    manifest_api_key: str
    deepseek_api_key: str
    storage_state: str | None  # path, or None if not present
    model: str = "deepseek-v4-flash"


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file. Real env vars always win."""
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load(storage_state: str | None = None) -> Config:
    _load_dotenv()
    missing = [k for k in ("MANIFEST_API_KEY", "DEEPSEEK_API_KEY") if not os.environ.get(k)]
    if missing:
        raise ConfigError(
            f"Missing required env var(s): {', '.join(missing)}. "
            "Set them (or use a .env sourced into your shell) before running."
        )

    path = storage_state or os.environ.get("STORAGE_STATE") or "storage_state.json"
    if not os.path.isfile(path):
        print(f"[config] no storage_state at {path!r} — running without a logged-in session")
        path = None

    return Config(
        manifest_api_key=os.environ["MANIFEST_API_KEY"],
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        storage_state=path,
        model=os.environ.get("AGENT_MODEL", "deepseek-v4-flash"),
    )
