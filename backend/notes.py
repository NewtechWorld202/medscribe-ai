"""AI note writing: transcript -> 12-section case paper fields.

Providers: Groq (cloud, needs GROQ_API_KEY) or Ollama (runs on this Mac, no key,
nothing leaves the machine). Both use the same prompt and the same checks.

Rules (deliberate, do not relax):
- The note is written ONLY from the transcript. Nothing that was not said is
  added: no examination findings, diagnoses, investigations or medicines.
- Every medicine must come with `quote`, a verbatim snippet of the transcript
  where it was prescribed. The backend checks that the quote really is in the
  transcript and drops the medicine if it is not. Dropped items are reported
  under `dropped` so the review screen can say so.
- There is no mock or demo note. If no key is set or the call fails, the API
  says so and the page falls back to its own rule-based builder, which reads
  the same real transcript.
- Each dose detail (strength, frequency, timing, duration) is checked against what
  was said around that medicine. A detail with no support there (small local
  models guess "40 mg, OD, 5 days") is moved to `unverified`: the page shows it as
  "not confirmed" and the doctor must confirm it before approving.
- Patient name, age, gender and fee are filled by the page, never by the model.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import Settings
from .medicines import FREQ, NUM, TIMING
from .transcription import _scrub

MAX_TRANSCRIPT_CHARS = 30000
TEXT_FIELDS = ("chief_complaint", "pain_details", "past_history", "examination",
               "investigation", "diagnosis", "advice")
MED_FIELDS = ("name", "strength", "frequency", "timing", "duration")
MIN_QUOTE_CHARS = 4

SYSTEM_PROMPT = """You are a clinical scribe for an orthopaedic / physiotherapy OPD clinic in Maharashtra, India.
You receive the transcript of ONE doctor-patient consultation. It may be in Hindi, Marathi, English or a mix,
in Devanagari or Roman script. Lines are labelled "Doctor:" or "Patient:"; these labels were guessed from the
text and can be wrong, so judge who is speaking from the content.

Write the case-paper fields in clear, concise clinical ENGLISH. Return ONLY one JSON object with exactly these keys:
{
  "chief_complaint": "e.g. Right knee pain since 2 weeks",
  "pain_details": "one item per line: Duration, Site, Character, Aggravating, Relieving, Radiation, Severity — only those that were said",
  "past_history": "relevant past illness, surgery, injury, diabetes/hypertension, current medicines — only if said",
  "examination": "only findings the doctor actually stated aloud during the consultation",
  "investigation": "reports reviewed, and tests advised (write 'X-ray right knee AP/lateral — advised')",
  "diagnosis": "only a diagnosis or impression the doctor stated; otherwise empty",
  "medicines": [
    {"name": "Tab Pantoprazole", "strength": "40 mg", "frequency": "OD (1-0-0)", "timing": "Before food",
     "duration": "5 days", "quote": "exact words from the transcript where this medicine was prescribed"}
  ],
  "advice": "non-drug advice the doctor gave, one item per line",
  "follow_up": {"after_days": 7, "note": "e.g. With X-ray report"}
}

STRICT RULES — this is a medical record:
1. Use ONLY what is in the transcript. Never add findings, diagnoses, tests, advice or medicines that were not said.
   If something was not said, use "" (or [] for medicines, null for after_days). Blank is always better than a guess.
2. Medicines: include a medicine only if the DOCTOR prescribed or continued it in this consultation. Skip medicines the
   doctor stopped, refused or only asked about, and medicines the patient merely mentions taking (put those in past_history).
3. "quote" must be copied character-for-character from the transcript (same script and spelling, no translation),
   8 to 120 characters, and must contain the medicine's mention. No quote, no medicine.
4. Medicine fields: write the standard English drug name with its form (Tab/Cap/Gel/Inj/Syp/Oint). Fill strength,
   frequency, timing and duration ONLY if they were spoken; otherwise "". Do not use typical or default doses.
   Frequency style: "OD (1-0-0)", "BD (1-0-1)", "TDS (1-1-1)", "HS (0-0-1)", "SOS", "Once weekly".
5. follow_up.after_days: convert what was said (1 week = 7, 1 month = 30). null if no follow-up was said.
6. Do not include patient name, age, gender or fee. Do not add any key that is not listed above."""


class NoteError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class NoteResult:
    provider: str
    model: str
    note: dict
    dropped: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- verification
def _norm(text: str) -> str:
    """Lower-case, NFC, punctuation and speaker labels removed, whitespace collapsed."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"(?m)^\s*(doctor|patient)\s*:\s*", " ", t)
    t = t.replace("‌", "").replace("‍", "")
    t = re.sub(r"[^\wऀ-ॿ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_transcript(quote: str, transcript: str) -> bool:
    q = _norm(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _norm(transcript)


def _evidence(quote: str, transcript: str) -> str:
    """The transcript line holding the quote, plus the next line (instructions often follow)."""
    lines = [l for l in transcript.splitlines() if l.strip()]
    head = _norm(quote)[:20]
    for i, line in enumerate(lines):
        if head and head in _norm(line):
            return "\n".join(lines[i:i + 2])
    return quote


def _numbers(text: str) -> set:
    nums = {int(d) for d in re.findall(r"\d+", text)}
    for tok in re.findall(r"[\w\u0900-\u097f]+", text.lower()):
        if tok in NUM:
            nums.add(NUM[tok])
    return nums


def _supported(field_name: str, value: str, evidence: str) -> bool:
    if field_name in ("strength", "duration"):
        said = _numbers(evidence)
        needed = {int(d) for d in re.findall(r"\d+", value)}
        return bool(needed) and needed <= said
    patterns = FREQ if field_name == "frequency" else TIMING
    return any(re.search(p, evidence, flags=re.IGNORECASE) for p, _ in patterns)


def _clean_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        v = "\n".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


_UNSAID = re.compile(r"^(not (mentioned|specified|stated|said|discussed|known|available)|none( mentioned)?|n/?a|nil|unknown|-+)\.?$",
                     re.IGNORECASE)
_FORM = re.compile(r"^(tab|cap|gel|inj|syp|oint|tablet|capsule|injection|syrup|ointment)\.?\s+", re.IGNORECASE)


def _drop_unsaid(text: str) -> str:
    """Remove filler like 'Not mentioned' (small models add it despite the prompt)."""
    keep = []
    for line in text.splitlines():
        value = line.split(":", 1)[1] if ":" in line else line
        if not _UNSAID.match(value.strip()):
            keep.append(line)
    return "\n".join(keep).strip()


def _drop_medicine_lines(advice: str, meds: List[dict]) -> str:
    """Advice is for non-drug advice; drop lines that just repeat a prescribed medicine."""
    words = {_FORM.sub("", m["name"]).split()[0].lower() for m in meds if _FORM.sub("", m["name"]).split()}
    if not words:
        return advice
    words |= {"tab", "tablet", "cap", "capsule", "inj", "injection", "syp", "syrup"}
    pat = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, words)) + r")(?!\w)", re.IGNORECASE)
    return "\n".join(l for l in advice.splitlines() if not pat.search(l)).strip()


def sanitise(raw: dict, transcript: str) -> tuple[dict, List[dict]]:
    """Keep only the expected keys, and only medicines whose quote is in the transcript."""
    if not isinstance(raw, dict):
        raise NoteError("bad_model_output", "The AI did not return a JSON object.")
    note = {k: _drop_unsaid(_clean_str(raw.get(k))) for k in TEXT_FIELDS}

    meds, dropped = [], []
    for m in raw.get("medicines") or []:
        if not isinstance(m, dict):
            continue
        med = {k: _drop_unsaid(_clean_str(m.get(k))) for k in MED_FIELDS}
        quote = _clean_str(m.get("quote"))
        if not med["name"]:
            continue
        if not quote_in_transcript(quote, transcript):
            dropped.append({"name": med["name"], "reason": "its quote was not found in the transcript"})
            continue
        med["quote"] = quote
        evidence = _evidence(quote, transcript)
        unverified = {}
        for f in ("strength", "frequency", "timing", "duration"):
            if med[f] and not _supported(f, med[f], evidence):
                unverified[f] = med[f]
                med[f] = ""
        med["unverified"] = unverified
        meds.append(med)
    note["medicines"] = meds
    note["advice"] = _drop_medicine_lines(note["advice"], meds)

    fu = raw.get("follow_up") if isinstance(raw.get("follow_up"), dict) else {}
    days = fu.get("after_days")
    try:
        days = int(days) if days not in (None, "") else None
    except (TypeError, ValueError):
        days = None
    if days is not None and not (0 < days <= 365):
        days = None
    note["follow_up"] = {"after_days": days, "note": _clean_str(fu.get("note"))}
    return note, dropped


# --------------------------------------------------------------------------- writers
Completer = Callable[[list], str]


class NoteWriter:
    """Shared prompt, JSON parsing and quote check. Subclasses only implement _call()."""
    name = ""
    model = ""
    _completer: Optional[Completer] = None

    def _complete(self, messages: list) -> str:
        if self._completer:  # tests inject a fake; real use calls the provider
            return self._completer(messages)
        return self._call(messages)

    def _call(self, messages: list) -> str:  # pragma: no cover
        raise NotImplementedError

    def write(self, transcript: str, language: str = "") -> NoteResult:
        transcript = str(transcript or "").strip()
        if not transcript:
            raise NoteError("empty_transcript", "The transcript is empty.", 400)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            raise NoteError("transcript_too_long", "The transcript is too long for one note.", 413)
        user = (f"Consultation language setting: {language or 'unknown'}\n\n"
                f"TRANSCRIPT (between the markers):\n<<<\n{transcript}\n>>>")
        content = self._complete([{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user}])
        try:
            raw = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise NoteError("bad_model_output", "The AI returned something that is not valid JSON.") from None
        note, dropped = sanitise(raw, transcript)
        return NoteResult(provider=self.name, model=self.model, note=note, dropped=dropped)


class GroqNoteWriter(NoteWriter):
    name = "groq"

    def __init__(self, api_key: str, model: str, completer: Optional[Completer] = None):
        self._api_key = api_key
        self.model = model
        self._completer = completer

    def _call(self, messages: list) -> str:
        try:
            import groq as groq_sdk
        except ImportError as e:  # pragma: no cover
            raise NoteError("groq_sdk_missing", "The groq package is not installed in the backend virtualenv.") from e
        client = groq_sdk.Groq(api_key=self._api_key, timeout=60.0, max_retries=1)
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0,
                response_format={"type": "json_object"}, max_tokens=2000,
            )
        except groq_sdk.AuthenticationError:
            raise NoteError("groq_auth_failed", "Groq rejected the API key. Check GROQ_API_KEY in .env.") from None
        except groq_sdk.RateLimitError:
            raise NoteError("groq_rate_limited", "Groq rate limit reached. Wait a minute and try again.", 503) from None
        except (groq_sdk.APITimeoutError, groq_sdk.APIConnectionError):
            raise NoteError("groq_unreachable", "Could not reach Groq (network error or timeout).", 504) from None
        except groq_sdk.APIStatusError as e:
            raise NoteError("groq_error", f"Groq returned HTTP {e.status_code}: {_scrub(e.message, self._api_key)}") from None
        except Exception as e:  # noqa: BLE001
            raise NoteError("groq_failed", f"AI note failed: {_scrub(type(e).__name__ + ': ' + str(e), self._api_key)}") from None
        return resp.choices[0].message.content or ""


class OllamaNoteWriter(NoteWriter):
    """Local model via Ollama (http://127.0.0.1:11434). No key; nothing leaves the Mac."""
    name = "ollama"

    def __init__(self, base_url: str, model: str, completer: Optional[Completer] = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._completer = completer

    def _call(self, messages: list) -> str:
        import httpx
        try:
            r = httpx.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "messages": messages, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_ctx": 8192},
            })
        except httpx.TimeoutException:
            raise NoteError("ollama_timeout", "The local AI (Ollama) took too long to write the note.", 504) from None
        except httpx.HTTPError:
            raise NoteError("ollama_unreachable", "Could not reach Ollama on this Mac. Is the Ollama app running?", 504) from None
        if r.status_code == 404:
            raise NoteError("ollama_model_missing",
                            f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}", 503)
        if r.status_code != 200:
            raise NoteError("ollama_error", f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["message"]["content"] or ""
        except (ValueError, KeyError, TypeError):
            raise NoteError("bad_model_output", "Ollama returned an unexpected response.") from None


def ollama_status(base_url: str, model: str, timeout: float = 1.0) -> tuple[bool, str]:
    """(ready, reason). Ready means Ollama answers and has the model downloaded."""
    import httpx
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001 - any failure just means "not available"
        return False, "Ollama is not running on this Mac"
    if model not in names and f"{model}:latest" not in names:
        return False, f"Ollama has no '{model}' model (run: ollama pull {model})"
    return True, ""


# --------------------------------------------------------------------------- selection
def resolve_note_writer(settings: Settings) -> tuple[Optional[NoteWriter], str]:
    """Return (writer, reason). writer is None when AI notes are off or unavailable.

    NOTE_PROVIDER: "auto" (default) = Groq if a key is set, otherwise Ollama on this Mac;
    "groq", "ollama" or "off" to force one."""
    choice = settings.note_provider
    if choice == "off":
        return None, "NOTE_PROVIDER=off"
    if choice not in ("auto", "groq", "ollama"):
        return None, f"NOTE_PROVIDER must be auto, groq, ollama or off (got '{choice}')"
    if choice == "groq" or (choice == "auto" and settings.has_groq_key):
        if not settings.has_groq_key:
            return None, "GROQ_API_KEY is not set"
        return GroqNoteWriter(api_key=settings.groq_api_key, model=settings.note_model), ""
    ready, reason = ollama_status(settings.ollama_url, settings.ollama_model)
    if not ready:
        return None, reason if choice == "ollama" else f"no GROQ_API_KEY, and {reason}"
    return OllamaNoteWriter(settings.ollama_url, settings.ollama_model), ""
