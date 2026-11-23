"""AI note-writing tests. Run from the project folder:

    .venv/bin/python -m unittest backend.tests.test_notes -v

Most tests use a fake model, so they need no key and no network. The live
test runs only when .env has a GROQ_API_KEY; otherwise it is skipped and says so.
"""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from backend import main
from backend.notes import (GroqNoteWriter, NoteError, OllamaNoteWriter, ollama_status,
                           quote_in_transcript, sanitise)
from backend.tests.test_backend import REAL_KEY, EnvCase

TRANSCRIPT = """Doctor: नमस्ते, बताइए क्या तकलीफ है?
Patient: डॉक्टर साहब, मेरे दाएं घुटने में दो हफ्ते से दर्द है।
Doctor: सीढ़ी चढ़ने में दर्द बढ़ता है क्या?
Patient: हाँ, सीढ़ी चढ़ने में बहुत दर्द होता है। मैं पहले से कैल्शियम ले रहा हूँ।
Doctor: Tab Pan 40 khali pet lijiye 10 din. Aceclofenac din mein do baar khane ke baad paanch din.
Doctor: एक हफ्ते बाद दिखाइए।"""

GOOD = {
    "chief_complaint": "Right knee pain since 2 weeks",
    "pain_details": "Duration: 2 weeks\nSite: Right knee\nAggravating: Climbing stairs",
    "past_history": "On calcium supplement",
    "examination": "", "investigation": "", "diagnosis": "",
    "medicines": [
        {"name": "Tab Pantoprazole", "strength": "40 mg", "frequency": "", "timing": "Before food",
         "duration": "10 days", "quote": "Tab Pan 40 khali pet lijiye 10 din"},
        {"name": "Tab Aceclofenac", "strength": "", "frequency": "BD (1-0-1)", "timing": "After food",
         "duration": "5 days", "quote": "aceclofenac din mein do baar khane ke baad"},
        # hallucinated: never said
        {"name": "Tab Etoricoxib", "strength": "90 mg", "frequency": "OD (1-0-0)", "timing": "", "duration": "",
         "quote": "Etoricoxib 90 roz ek"},
        # no quote at all
        {"name": "Cap Vitamin D3", "strength": "60000 IU", "frequency": "Once weekly", "timing": "", "duration": ""},
    ],
    "advice": "", "follow_up": {"after_days": 7, "note": ""},
    "patient_name": "should be ignored",
}


def fake(payload):
    return lambda messages: payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)


class QuoteCheckTests(unittest.TestCase):
    def test_quotes_match_ignoring_case_punctuation_and_labels(self):
        self.assertTrue(quote_in_transcript("tab pan 40, khali pet", TRANSCRIPT))
        self.assertTrue(quote_in_transcript("Doctor: एक हफ्ते बाद दिखाइए", TRANSCRIPT))
        self.assertFalse(quote_in_transcript("Etoricoxib 90", TRANSCRIPT))
        self.assertFalse(quote_in_transcript("din", TRANSCRIPT), "too short to count as evidence")
        self.assertFalse(quote_in_transcript("", TRANSCRIPT))


class SanitiseTests(unittest.TestCase):
    def test_unsupported_medicines_are_dropped_and_reported(self):
        note, dropped = sanitise(GOOD, TRANSCRIPT)
        self.assertEqual([m["name"] for m in note["medicines"]], ["Tab Pantoprazole", "Tab Aceclofenac"])
        self.assertEqual({d["name"] for d in dropped}, {"Tab Etoricoxib", "Cap Vitamin D3"})
        self.assertNotIn("patient_name", note)
        self.assertEqual(note["follow_up"], {"after_days": 7, "note": ""})

    def test_guessed_dose_details_are_unverified(self):
        t = "Doctor: मैं पैंतो क्राजोल लिख रहा हूँ, खाने से पहले लिजिए."
        raw = {"medicines": [{"name": "Tab Pantoprazole", "strength": "40 mg", "frequency": "OD (1-0-0)",
                              "timing": "Before food", "duration": "5 days", "quote": "मैं पैंतो क्राजोल लिख रहा हूँ"}]}
        [m] = sanitise(raw, t)[0]["medicines"]
        self.assertEqual(m["timing"], "Before food", "खाने से पहले was said")
        self.assertEqual((m["strength"], m["frequency"], m["duration"]), ("", "", ""))
        self.assertEqual(m["unverified"], {"strength": "40 mg", "frequency": "OD (1-0-0)", "duration": "5 days"})

    def test_spoken_dose_details_stay_verified(self):
        note, _ = sanitise(GOOD, TRANSCRIPT)
        pan, acec = note["medicines"]
        self.assertEqual((pan["strength"], pan["timing"], pan["duration"]), ("40 mg", "Before food", "10 days"))
        self.assertEqual((acec["frequency"], acec["timing"], acec["duration"]), ("BD (1-0-1)", "After food", "5 days"))
        self.assertEqual(pan["unverified"], {})

    def test_filler_and_repeated_medicines_are_removed(self):
        raw = dict(GOOD, pain_details="Duration: 2 weeks\nRelieving: Not mentioned\nSeverity: N/A",
                   examination="Not mentioned", advice="Take Tab Pan 40 before food\nAvoid stairs\nAceclofenac after food")
        note, _ = sanitise(raw, TRANSCRIPT)
        self.assertEqual(note["pain_details"], "Duration: 2 weeks")
        self.assertEqual(note["examination"], "")
        self.assertEqual(note["advice"], "Avoid stairs")

    def test_bad_follow_up_and_list_fields_are_normalised(self):
        note, _ = sanitise({"advice": ["Ice 10 min", "Avoid stairs"], "follow_up": {"after_days": "lots"}}, TRANSCRIPT)
        self.assertEqual(note["advice"], "Ice 10 min\nAvoid stairs")
        self.assertIsNone(note["follow_up"]["after_days"])
        self.assertEqual(note["medicines"], [])
        self.assertEqual(note["diagnosis"], "")


class WriterTests(unittest.TestCase):
    def test_ollama_writer_uses_the_same_checks(self):
        r = OllamaNoteWriter(OLLAMA_URL, "m", completer=fake(GOOD)).write(TRANSCRIPT, "hi")
        self.assertEqual((r.provider, len(r.note["medicines"]), len(r.dropped)), ("ollama", 2, 2))

    def test_ollama_unreachable_is_an_error(self):
        with self.assertRaises(NoteError) as cm:
            OllamaNoteWriter(DEAD_URL, "m", timeout=2).write(TRANSCRIPT)
        self.assertEqual(cm.exception.code, "ollama_unreachable")

    def test_writer_parses_and_verifies(self):
        r = GroqNoteWriter("k", "m", completer=fake(GOOD)).write(TRANSCRIPT, "hi")
        self.assertEqual(len(r.note["medicines"]), 2)
        self.assertEqual(len(r.dropped), 2)

    def test_invalid_json_is_an_error_not_a_note(self):
        with self.assertRaises(NoteError) as cm:
            GroqNoteWriter("k", "m", completer=fake("Sure! Here is the note: ...")).write(TRANSCRIPT)
        self.assertEqual(cm.exception.code, "bad_model_output")

    def test_empty_transcript_is_rejected(self):
        with self.assertRaises(NoteError) as cm:
            GroqNoteWriter("k", "m", completer=fake(GOOD)).write("   ")
        self.assertEqual(cm.exception.code, "empty_transcript")


DEAD_URL = "http://127.0.0.1:9"  # nothing listens here
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_READY = ollama_status(OLLAMA_URL, OLLAMA_MODEL)[0]


class NoteEndpointTests(EnvCase):
    def test_no_key_means_unavailable_never_a_sample_note(self):
        os.environ["NOTE_PROVIDER"] = "groq"
        h = self.client.get("/api/health").json()
        self.assertFalse(h["note_ai"]["enabled"])
        self.assertIn("GROQ_API_KEY is not set", h["note_ai"]["reason"])
        r = self.client.post("/api/note", json={"transcript": TRANSCRIPT})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"], "note_ai_unavailable")
        self.assertNotIn("note", r.json())

    def test_auto_without_key_or_ollama_is_unavailable(self):
        os.environ["OLLAMA_URL"] = DEAD_URL
        na = self.client.get("/api/health").json()["note_ai"]
        self.assertFalse(na["enabled"])
        self.assertIn("Ollama is not running", na["reason"])
        r = self.client.post("/api/note", json={"transcript": TRANSCRIPT})
        self.assertEqual(r.status_code, 503)

    def test_auto_prefers_groq_when_key_is_set(self):
        os.environ["GROQ_API_KEY"] = "gsk_dummy"
        self.assertEqual(self.client.get("/api/health").json()["note_ai"]["provider"], "groq")

    @unittest.skipUnless(OLLAMA_READY, "Ollama with llama3.1:8b is not running")
    def test_auto_without_key_uses_ollama(self):
        na = self.client.get("/api/health").json()["note_ai"]
        self.assertEqual((na["enabled"], na["provider"], na["model"]), (True, "ollama", OLLAMA_MODEL))

    def test_ollama_missing_model_is_reported(self):
        os.environ["NOTE_PROVIDER"] = "ollama"
        os.environ["OLLAMA_NOTE_MODEL"] = "no-such-model:1b"
        na = self.client.get("/api/health").json()["note_ai"]
        self.assertFalse(na["enabled"])

    def test_note_provider_off(self):
        os.environ["GROQ_API_KEY"] = "gsk_dummy"
        with mock.patch.dict(os.environ, {"NOTE_PROVIDER": "off"}):
            self.assertFalse(self.client.get("/api/health").json()["note_ai"]["enabled"])

    def test_endpoint_with_fake_model(self):
        os.environ["GROQ_API_KEY"] = "gsk_dummy"
        orig = main.resolve_note_writer
        with mock.patch.object(main, "resolve_note_writer",
                               lambda s: (GroqNoteWriter("k", "fake-model", completer=fake(GOOD)), "")):
            r = self.client.post("/api/note", json={"transcript": TRANSCRIPT, "language": "hi"})
        self.assertIs(main.resolve_note_writer, orig)
        j = r.json()
        self.assertEqual(r.status_code, 200, j)
        self.assertEqual(j["model"], "fake-model")
        self.assertEqual([m["name"] for m in j["note"]["medicines"]], ["Tab Pantoprazole", "Tab Aceclofenac"])
        self.assertEqual(len(j["dropped"]), 2)
        self.assertTrue(j["medicines"], "clinic-list matches come back too, for unconfirmed suggestions")


@unittest.skipUnless(REAL_KEY, "GROQ_API_KEY is empty in .env, so the live AI note path is UNTESTED")
class NoteLiveTests(EnvCase):
    def test_live_note_from_hindi_transcript(self):
        os.environ["GROQ_API_KEY"] = REAL_KEY
        r = self.client.post("/api/note", json={"transcript": TRANSCRIPT, "language": "hi"})
        j = r.json()
        self.assertEqual(r.status_code, 200, j.get("message"))
        print("\n--- live AI note ---\n" + json.dumps(j["note"], ensure_ascii=False, indent=2), "\ndropped:", j["dropped"])
        names = " ".join(m["name"].lower() for m in j["note"]["medicines"])
        self.assertIn("pantoprazole", names)
        self.assertIn("aceclofenac", names)
        self.assertNotIn("calcium", names, "patient's own calcium is history, not a prescription")
        self.assertIn("knee", j["note"]["chief_complaint"].lower())


if __name__ == "__main__":
    unittest.main()
