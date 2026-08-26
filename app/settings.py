from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "data" / "settings.json"


def get_gemini_api_key() -> str:
    stored = _read_settings().get("gemini_api_key", "")
    return stored or os.getenv("GEMINI_API_KEY", "")


def set_gemini_api_key(api_key: str) -> None:
    settings = _read_settings()
    settings["gemini_api_key"] = api_key.strip()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def has_gemini_api_key() -> bool:
    key = get_gemini_api_key()
    return bool(key and "put_your" not in key)


def get_gemini_model(default: str) -> str:
    return _read_settings().get("gemini_model", default)


def set_gemini_model(model: str) -> None:
    settings = _read_settings()
    settings["gemini_model"] = model.strip()
    _write_settings(settings)


def get_words_per_subtitle(default: int = 1) -> int:
    raw_value = _read_settings().get("words_per_subtitle", str(default))
    try:
        return min(3, max(1, int(raw_value)))
    except ValueError:
        return default


def set_words_per_subtitle(count: int) -> None:
    settings = _read_settings()
    settings["words_per_subtitle"] = str(min(3, max(1, count)))
    _write_settings(settings)


def _read_settings() -> dict[str, str]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_settings(settings: dict[str, str]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
