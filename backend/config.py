"""Settings for the MedScribe backend.

Values come from the project's `.env` file, and real environment variables
override it. Settings are re-read on every request, so pasting a new
GROQ_API_KEY into .env takes effect without a restart.

Key values are never logged or returned by the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    transcription_provider: str  # "auto", "groq", "local" or "mock"
    groq_api_key: str
    groq_model: str
    port: int
    note_provider: str = "auto"  # "auto", "groq", "ollama" or "off"
    note_model: str = "llama-3.3-70b-versatile"  # Groq model for notes
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    local_whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key.strip())


def load_settings() -> Settings:
    env_file = Path(os.environ.get("MEDSCRIBE_ENV_FILE", str(DEFAULT_ENV_FILE)))
    file_values = _read_env_file(env_file)

    def get(name: str, default: str = "") -> str:
        if name in os.environ:
            return os.environ[name]
        return file_values.get(name, default)

    try:
        port = int(get("MEDSCRIBE_BACKEND_PORT", "8770") or "8770")
    except ValueError:
        port = 8770
    return Settings(
        # "auto": Groq if a key exists, else local Whisper, else mock (see transcription.resolve_provider).
        transcription_provider=(get("TRANSCRIPTION_PROVIDER", "auto") or "auto").strip().lower(),
        groq_api_key=get("GROQ_API_KEY", "").strip(),
        groq_model=(get("GROQ_STT_MODEL", "whisper-large-v3") or "whisper-large-v3").strip(),
        port=port,
        note_provider=(get("NOTE_PROVIDER", "auto") or "auto").strip().lower(),
        note_model=(get("GROQ_NOTE_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile").strip(),
        ollama_url=(get("OLLAMA_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").strip(),
        ollama_model=(get("OLLAMA_NOTE_MODEL", "llama3.1:8b") or "llama3.1:8b").strip(),
        local_whisper_model=(get("LOCAL_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
                             or "mlx-community/whisper-large-v3-turbo").strip(),
    )
