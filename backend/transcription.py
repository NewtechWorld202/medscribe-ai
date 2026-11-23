"""Speech-to-text adapter: Groq Whisper (cloud), local Whisper (MLX, on this Mac) or mock (demo) data.

Rules (deliberate, do not relax):
- Mock is used ONLY when TRANSCRIPTION_PROVIDER=mock, or when nothing real is
  available (no GROQ_API_KEY and no local Whisper), and every mock response is
  flagged `demo: true`.
- If a real provider is configured and fails, a TranscriptionError is raised.
  There is never a silent fallback to mock data, because a fake transcript can
  put medicines into a note that nobody said.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .config import Settings

SUPPORTED_LANGUAGES = {"hi", "mr", "en", "gu"}


class ConfigError(Exception):
    """The backend is misconfigured (e.g. an unknown TRANSCRIPTION_PROVIDER)."""


class TranscriptionError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    speaker_basis: str = ""

    def to_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "text": self.text,
                "speaker": self.speaker, "speaker_basis": self.speaker_basis}


@dataclass
class TranscriptionResult:
    provider: str
    model: str
    language: str
    duration: Optional[float]
    segments: List[Segment]
    demo: bool = False
    notice: str = ""
    raw_language: str = ""
    extra: dict = field(default_factory=dict)


def normalise_language(lang: Optional[str]) -> str:
    code = (lang or "hi").strip().lower().replace("_", "-").split("-")[0]
    if code not in SUPPORTED_LANGUAGES:
        raise TranscriptionError("unsupported_language",
                                 f"Language '{lang}' is not supported. Use one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}.",
                                 http_status=400)
    return code


# --------------------------------------------------------------------------- mock
_MOCK_DIALOGUE = {
    "hi": [
        "नमस्ते, बताइए क्या तकलीफ है?",
        "डॉक्टर साहब, मेरे दाएं घुटने में दो हफ्ते से दर्द है।",
        "सीढ़ी चढ़ने में दर्द बढ़ता है क्या?",
        "हाँ, सीढ़ी चढ़ने में बहुत दर्द होता है।",
        "मैं पैंटोप्राजोल और एसिक्लोफेनाक लिख रहा हूँ, खाने के बाद लीजिए।",
    ],
    "mr": [
        "नमस्कार, काय त्रास होतोय?",
        "डॉक्टर, माझ्या कमरेत एक आठवड्यापासून दुखतंय.",
        "जड वस्तू उचलल्यावर जास्त दुखतं का?",
        "हो, वाकल्यावर खूप दुखतं.",
        "मी पॅन्टोप्राझोल आणि एसिक्लोफेनाक देतो, जेवणानंतर घ्या.",
    ],
    "en": [
        "Good morning, what brings you in today?",
        "Doctor, I have had pain in my right knee for two weeks.",
        "Does it hurt more when you climb stairs?",
        "Yes, climbing stairs is very painful.",
        "I am prescribing pantoprazole and aceclofenac, take them after food.",
    ],
}


class MockProvider:
    name = "mock"
    model = "mock-demo"

    def __init__(self, reason: str):
        self.reason = reason

    def transcribe(self, audio: bytes, filename: str, language: str) -> TranscriptionResult:
        lines = _MOCK_DIALOGUE.get(language) or _MOCK_DIALOGUE["en"]
        segs, t = [], 0.0
        for line in lines:
            dur = 1.5 + len(line) / 18.0
            segs.append(Segment(start=t, end=t + dur, text=line))
            t += dur + 0.8  # pause between turns
        return TranscriptionResult(
            provider=self.name, model=self.model, language=language, duration=t, segments=segs, demo=True,
            notice=f"DEMO DATA, not a real transcription ({self.reason}). The audio was not transcribed.",
        )


# --------------------------------------------------------------------------- groq
def _scrub(text: str, key: str) -> str:
    """Remove anything that looks like an API key from an error message."""
    text = re.sub(r"gsk_[A-Za-z0-9*._-]+", "[redacted]", str(text))
    if key:
        text = text.replace(key, "[redacted]")
    return text[:300]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None) or {}
        val = extra.get(name, default)
    return default if val is None else val


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self.model = model

    def transcribe(self, audio: bytes, filename: str, language: str) -> TranscriptionResult:
        try:
            import groq as groq_sdk
        except ImportError as e:  # pragma: no cover - only when the venv is broken
            raise TranscriptionError("groq_sdk_missing", "The groq package is not installed in the backend virtualenv.") from e

        client = groq_sdk.Groq(api_key=self._api_key, timeout=90.0, max_retries=1)
        try:
            resp = client.audio.transcriptions.create(
                file=(filename or "audio.webm", audio),
                model=self.model,
                response_format="verbose_json",
                language=language,
                temperature=0.0,
            )
        except groq_sdk.AuthenticationError as e:
            raise TranscriptionError("groq_auth_failed",
                                     f"Groq rejected the API key (HTTP {getattr(e, 'status_code', 401)}). "
                                     "Check GROQ_API_KEY in .env. No transcript was produced.") from None
        except groq_sdk.PermissionDeniedError as e:
            raise TranscriptionError("groq_permission_denied",
                                     f"Groq denied access (HTTP {e.status_code}): {_scrub(e.message, self._api_key)}") from None
        except groq_sdk.RateLimitError:
            raise TranscriptionError("groq_rate_limited",
                                     "Groq rate limit reached. Wait a minute and try again. No transcript was produced.",
                                     http_status=503) from None
        except (groq_sdk.APITimeoutError, groq_sdk.APIConnectionError):
            raise TranscriptionError("groq_unreachable",
                                     "Could not reach Groq (network error or timeout). No transcript was produced.",
                                     http_status=504) from None
        except groq_sdk.APIStatusError as e:
            raise TranscriptionError("groq_error",
                                     f"Groq returned HTTP {e.status_code}: {_scrub(e.message, self._api_key)}") from None
        except Exception as e:  # noqa: BLE001 - anything else is still a hard failure, never mock
            raise TranscriptionError("groq_failed", f"Groq transcription failed: {_scrub(type(e).__name__ + ': ' + str(e), self._api_key)}") from None

        raw_segments = _get(resp, "segments", []) or []
        segs = []
        for s in raw_segments:
            text = str(_get(s, "text", "")).strip()
            if text:
                segs.append(Segment(start=float(_get(s, "start", 0.0)), end=float(_get(s, "end", 0.0)), text=text))
        if not segs:
            text = str(_get(resp, "text", "")).strip()
            if text:
                segs.append(Segment(start=0.0, end=float(_get(resp, "duration", 0.0) or 0.0), text=text))
        duration = _get(resp, "duration", None)
        return TranscriptionResult(
            provider=self.name, model=self.model, language=language,
            duration=float(duration) if duration is not None else None, segments=segs,
            raw_language=str(_get(resp, "language", "") or ""),
        )


# --------------------------------------------------------------------------- local (MLX Whisper)
_LOCAL_LOCK = threading.Lock()  # one transcription at a time; the model uses the whole GPU


def _ensure_ffmpeg_on_path() -> None:
    """mlx-whisper decodes audio with ffmpeg; apps started from Finder may lack Homebrew's PATH."""
    if shutil.which("ffmpeg"):
        return
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.exists(os.path.join(d, "ffmpeg")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


def local_whisper_status(model: str) -> tuple[bool, str]:
    """(ready, reason). Ready means mlx-whisper is installed and the model is fully downloaded."""
    try:
        import mlx_whisper  # noqa: F401
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "mlx-whisper is not installed"
    try:
        snapshot_download(model, local_files_only=True)
    except Exception:  # noqa: BLE001 - not in the cache, or only partly downloaded
        return False, f"the local Whisper model '{model}' is not downloaded yet"
    _ensure_ffmpeg_on_path()
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg is not installed (brew install ffmpeg)"
    return True, ""


class LocalWhisperProvider:
    """Whisper running on this Mac (Apple silicon, MLX). No key; audio never leaves the machine."""
    name = "local"

    def __init__(self, model: str):
        self.model = model

    def transcribe(self, audio: bytes, filename: str, language: str) -> TranscriptionResult:
        import mlx_whisper
        _ensure_ffmpeg_on_path()
        suffix = os.path.splitext(filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio)
            f.flush()
            try:
                with _LOCAL_LOCK:
                    resp = mlx_whisper.transcribe(
                        f.name, path_or_hf_repo=self.model, language=language, temperature=0.0,
                        condition_on_previous_text=False,  # avoids Whisper repeating itself on pauses
                        verbose=None,
                    )
            except Exception as e:  # noqa: BLE001 - a hard failure, never mock
                raise TranscriptionError("local_whisper_failed",
                                         f"Local Whisper could not transcribe the audio: {type(e).__name__}: {str(e)[:200]}") from None
        segs = [Segment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", "")).strip())
                for s in resp.get("segments", []) if str(s.get("text", "")).strip()]
        if not segs and str(resp.get("text", "")).strip():
            segs = [Segment(start=0.0, end=0.0, text=str(resp["text"]).strip())]
        duration = max((s.end for s in segs), default=None)
        return TranscriptionResult(provider=self.name, model=self.model, language=language, duration=duration,
                                   segments=segs, raw_language=str(resp.get("language", "") or ""))


# --------------------------------------------------------------------------- selection
def resolve_provider(settings: Settings):
    """TRANSCRIPTION_PROVIDER: "auto" (default) = Groq if a key is set, else local Whisper if it is
    installed and downloaded, else mock (flagged demo). "groq", "local" or "mock" force one."""
    choice = settings.transcription_provider
    if choice not in ("auto", "groq", "local", "mock"):
        raise ConfigError(f"TRANSCRIPTION_PROVIDER must be auto, groq, local or mock (got '{choice}').")
    if choice == "mock":
        return MockProvider(reason="TRANSCRIPTION_PROVIDER=mock")
    if choice == "groq" or (choice == "auto" and settings.has_groq_key):
        if not settings.has_groq_key:
            return MockProvider(reason="GROQ_API_KEY is not set")
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    ready, reason = local_whisper_status(settings.local_whisper_model)
    if ready:
        return LocalWhisperProvider(settings.local_whisper_model)
    return MockProvider(reason=reason if choice == "local" else f"no GROQ_API_KEY, and {reason}")
