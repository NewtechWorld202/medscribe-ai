# MedScribe AI

Ambient clinical scribe for an orthopaedic / physiotherapy OPD (Parasnath Speciality Clinic, Badlapur).
It records a doctor–patient consultation in Hindi, Marathi or English, transcribes it, and drafts a
12-section case paper and SOAP note for the doctor to review, edit and approve.

Everything can run **on the Mac, with no API key**: speech-to-text with Whisper (MLX) and note writing
with a local model in Ollama. Groq (cloud) can be switched on later by adding a key.

## Run it

Double-click **`Start MedScribe.command`**. It starts the page on `http://localhost:8765/medscribe-ai.html`
and the backend on `http://127.0.0.1:8770`, and opens Chrome (Chrome needs localhost for the microphone).

## First-time setup (macOS, Apple silicon)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
brew install ffmpeg          # used by local Whisper to read audio
ollama pull llama3.1:8b      # local note-writing model (install Ollama from ollama.com)
```

The local Whisper model (`mlx-community/whisper-large-v3-turbo`, about 1.6 GB) downloads on first use.
Without the backend, the page still works with Chrome's own speech recognition and a rule-based note builder.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `TRANSCRIPTION_PROVIDER` | `auto` | `auto` = Groq if a key is set, else Whisper on this Mac, else demo data (always labelled). Or `groq`, `local`, `mock` |
| `NOTE_PROVIDER` | `auto` | `auto` = Groq if a key is set, else Ollama on this Mac, else the page's rule-based builder. Or `groq`, `ollama`, `off` |
| `GROQ_API_KEY` | empty | Optional. Never commit `.env` |
| `LOCAL_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Local speech-to-text model |
| `OLLAMA_NOTE_MODEL` | `llama3.1:8b` | Local note-writing model |

## Safety rules (deliberate)

- A real recording is never replaced by demo data. If transcription fails, the error is shown.
- The AI writes only from the transcript. Every medicine must come with a quote that is really in the
  transcript, or it is removed (and the review screen says so).
- Dose details (strength, frequency, timing, duration) that cannot be found near the medicine in the
  transcript, and clinic-list default doses, are shown as **"Suggested · not confirmed"**. Approval is
  blocked until the doctor confirms or clears each one.
- Doctor/Patient labels are guessed from the words, not the voices, and can be wrong.
- Every note is a draft that needs doctor review and approval.

## Tests

```bash
.venv/bin/python -m unittest backend.tests.test_backend backend.tests.test_notes -v
```

Tests that need a Groq key, Ollama or the local Whisper model are skipped (and say why) when those are missing.

## Layout

- `medscribe-ai.html` – the whole app UI (single page)
- `backend/main.py` – FastAPI endpoints: `/api/health`, `/api/transcribe`, `/api/note`, `/api/extract-medicines`
- `backend/transcription.py` – Groq / local Whisper / mock speech-to-text
- `backend/notes.py` – AI note writing (Groq / Ollama) and the transcript checks
- `backend/diarization.py` – Doctor / Patient labels
- `backend/medicines.py`, `backend/clinic_medicines.json` – clinic medicine list matching (sample defaults: review before use)
