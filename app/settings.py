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


def _read_settings() -> dict[str, str]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
