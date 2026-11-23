"""MedScribe AI backend (FastAPI).

Run from the project folder:
    .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8770

Endpoints
    GET  /api/health             provider in use, and whether it is demo data
    POST /api/transcribe         multipart: audio=<file>, language=hi|mr|en|gu
    POST /api/extract-medicines  JSON: {"transcript": "..."}
    POST /api/note               JSON: {"transcript": "...", "language": "hi"} -> AI case-paper fields
    GET  /api/clinic-medicines   the clinic medicine list
    GET  /                       the MedScribe page itself
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, load_settings
from .diarization import label_speakers
from .medicines import load_clinic_list, match_medicines
from .notes import NoteError, resolve_note_writer
from .transcription import (ConfigError, MockProvider, TranscriptionError,
                            normalise_language, resolve_provider)

log = logging.getLogger("medscribe")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # Groq's limit for direct uploads
PAGE = PROJECT_ROOT / "medscribe-ai.html"

app = FastAPI(title="MedScribe AI backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{h}:{p}" for h in ("localhost", "127.0.0.1") for p in (8765, 8766, 8770)],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": code, "message": message})


@app.get("/api/health")
def health():
    settings = load_settings()
    try:
        provider = resolve_provider(settings)
    except ConfigError as e:
        return _error(500, "config_error", str(e))
    is_mock = isinstance(provider, MockProvider)
    return {
        "ok": True,
        "provider": provider.name,
        "model": provider.model,
        "demo": is_mock,
        "reason": getattr(provider, "reason", ""),
        "groq_key_present": settings.has_groq_key,  # a yes/no only, never the value
        "note_ai": _note_status(settings),
        "clinic_list_version": load_clinic_list().get("version", ""),
    }


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("hi")):
    settings = load_settings()
    try:
        provider = resolve_provider(settings)
        lang = normalise_language(language)
    except ConfigError as e:
        return _error(500, "config_error", str(e))
    except TranscriptionError as e:
        return _error(e.http_status, e.code, e.message)

    data = await audio.read()
    if not data:
        return _error(400, "empty_audio", "The uploaded audio file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(413, "audio_too_large", "Audio is larger than 25 MB. Record a shorter clip.")

    try:
        # in a worker thread: local Whisper can take a minute, and health checks must keep answering
        result = await run_in_threadpool(provider.transcribe, data, audio.filename or "audio.webm", lang)
    except TranscriptionError as e:
        log.warning("Transcription failed: %s", e.code)
        return _error(e.http_status, e.code, e.message)

    segments = label_speakers(result.segments)
    transcript = "\n".join(f"{s.speaker}: {s.text}" for s in segments)
    return {
        "ok": True,
        "provider": result.provider,
        "model": result.model,
        "demo": result.demo,
        "notice": result.notice,
        "language": result.language,
        "detected_language": result.raw_language,
        "duration": result.duration,
        "speaker_labels": "text-cue heuristic (not voice-based); doctor should check",
        "segments": [s.to_dict() for s in segments],
        "transcript": transcript,
        "medicines": match_medicines(transcript),
    }


class ExtractIn(BaseModel):
    transcript: str = ""


@app.post("/api/extract-medicines")
def extract_medicines(body: ExtractIn):
    return {"ok": True, "clinic_list_version": load_clinic_list().get("version", ""),
            "medicines": match_medicines(body.transcript)}


def _note_status(settings) -> dict:
    writer, reason = resolve_note_writer(settings)
    return {"enabled": writer is not None, "provider": writer.name if writer else "",
            "model": writer.model if writer else "", "reason": reason}


class NoteIn(BaseModel):
    transcript: str = ""
    language: str = ""


@app.post("/api/note")
def write_note(body: NoteIn):
    """AI case-paper draft from a transcript. Never returns a made-up note:
    if AI notes are off or fail, it returns ok=false and the page uses its own builder."""
    writer, reason = resolve_note_writer(load_settings())
    if writer is None:
        return _error(503, "note_ai_unavailable", f"AI note writing is off ({reason}).")
    try:
        result = writer.write(body.transcript, body.language)
    except NoteError as e:
        log.warning("AI note failed: %s", e.code)
        return _error(e.http_status, e.code, e.message)
    return {"ok": True, "provider": result.provider, "model": result.model,
            "note": result.note, "dropped": result.dropped,
            "medicines": match_medicines(body.transcript)}


@app.get("/api/clinic-medicines")
def clinic_medicines():
    return load_clinic_list()


@app.get("/")
@app.get("/medscribe-ai.html")
def page():
    return FileResponse(Path(PAGE), media_type="text/html")
