"""Env-var + storage-state loading. Fails fast with a clear message."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    manifest_api_key: str
    anthropic_api_key: str
    storage_state: str | None  # path, or None if not present
    model: str = "claude-sonnet-4-6"


def load(storage_state: str | None = None) -> Config:
    missing = [k for k in ("MANIFEST_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
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
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        storage_state=path,
        model=os.environ.get("AGENT_MODEL", "claude-sonnet-4-6"),
    )
