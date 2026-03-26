"""Persistent web UI settings."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from kimi_cli.share import get_share_dir
from kimi_cli.utils.io import atomic_json_write
from kimi_cli.utils.logging import logger


class WebSettings(BaseModel):
    """Persisted settings for the web UI."""

    agent_ran_enabled: bool = False


def _settings_path() -> Path:
    return get_share_dir() / "web_settings.json"


def load_web_settings() -> WebSettings:
    path = _settings_path()
    if not path.exists():
        return WebSettings()
    try:
        with path.open(encoding="utf-8") as handle:
            return WebSettings.model_validate(json.load(handle))
    except (json.JSONDecodeError, ValidationError, UnicodeDecodeError):
        logger.warning("Corrupted web settings file, using defaults: {path}", path=path)
        return WebSettings()


def save_web_settings(settings: WebSettings) -> None:
    path = _settings_path()
    atomic_json_write(settings.model_dump(mode="json"), path)
