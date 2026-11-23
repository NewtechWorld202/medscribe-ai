"""Backend tests. Run from the project folder:

    .venv/bin/python -m unittest backend.tests.test_backend -v

The live Groq tests run only when the project's .env has a GROQ_API_KEY;
otherwise they are skipped and say so. Set MEDSCRIBE_SKIP_NETWORK=1 to skip
the bad-key test, which calls Groq with a deliberately invalid key.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend import config
from backend.diarization import label_speakers
from backend.main import app
from backend.medicines import match_medicines
from backend.transcription import Segment, local_whisper_status

AUDIO = Path(__file__).resolve().parent / "audio"
HINDI = AUDIO / "hindi_consult.m4a"
MARATHI = AUDIO / "marathi_consult.m4a"
ENV_KEYS = ("TRANSCRIPTION_PROVIDER", "GROQ_API_KEY", "GROQ_STT_MODEL", "MEDSCRIBE_ENV_FILE",
            "NOTE_PROVIDER", "OLLAMA_URL", "OLLAMA_NOTE_MODEL", "LOCAL_WHISPER_MODEL")
REAL_ENV = config._read_env_file(config.DEFAULT_ENV_FILE)
REAL_KEY = os.environ.get("GROQ_API_KEY") or REAL_ENV.get("GROQ_API_KEY", "")
LOCAL_MODEL = "mlx-community/whisper-large-v3-turbo"
LOCAL_READY, LOCAL_REASON = local_whisper_status(LOCAL_MODEL)


class EnvCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        os.environ["MEDSCRIBE_ENV_FILE"] = str(AUDIO / "no-such.env")  # isolate from the real .env
        for k in ("TRANSCRIPTION_PROVIDER", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        self.client = TestClient(app)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def post_audio(self, path: Path, language: str):
        with open(path, "rb") as f:
            return self.client.post("/api/transcribe", files={"audio": (path.name, f, "audio/mp4")},
                                    data={"language": language})


class MockProviderTests(EnvCase):
    def test_mock_mode_end_to_end_hindi_and_marathi(self):
        os.environ["TRANSCRIPTION_PROVIDER"] = "mock"
        h = self.client.get("/api/health").json()
        self.assertEqual((h["provider"], h["demo"]), ("mock", True))
        for path, lang in ((HINDI, "hi"), (MARATHI, "mr-IN")):
            r = self.post_audio(path, lang)
            self.assertEqual(r.status_code, 200, r.text)
            j = r.json()
            self.assertTrue(j["demo"])
            self.assertIn("DEMO DATA", j["notice"])
            speakers = [s["speaker"] for s in j["segments"]]
            self.assertEqual(speakers, ["Doctor", "Patient", "Doctor", "Patient", "Doctor"], f"{lang}: {speakers}")
            self.assertTrue(j["transcript"].startswith("Doctor: "))
            self.assertTrue(j["medicines"], "mock dialogue names two medicines")
            for m in j["medicines"]:
                self.assertFalse(m["confirmed"])
                self.assertTrue(m["needs_doctor_confirmation"])
                self.assertEqual(m["suggestion_source"], "clinic_default")
                self.assertIn(m["mentioned_as"], j["transcript"])

    def test_groq_without_key_uses_mock_and_says_why(self):
        os.environ["TRANSCRIPTION_PROVIDER"] = "groq"
        h = self.client.get("/api/health").json()
        self.assertEqual(h["provider"], "mock")
        self.assertIn("GROQ_API_KEY is not set", h["reason"])
        self.assertFalse(h["groq_key_present"])

    def test_auto_without_key_or_local_model_is_flagged_demo(self):
        os.environ["LOCAL_WHISPER_MODEL"] = "mlx-community/no-such-whisper-model"
        h = self.client.get("/api/health").json()
        self.assertEqual((h["provider"], h["demo"]), ("mock", True))
        self.assertIn("not downloaded", h["reason"])

    def test_local_forced_but_unavailable_is_flagged_demo(self):
        os.environ["TRANSCRIPTION_PROVIDER"] = "local"
        os.environ["LOCAL_WHISPER_MODEL"] = "mlx-community/no-such-whisper-model"
        r = self.post_audio(HINDI, "hi")
        self.assertTrue(r.json()["demo"], "an unavailable local model must never look like a real transcript")

    def test_unknown_provider_is_a_config_error(self):
        os.environ["TRANSCRIPTION_PROVIDER"] = "whisperx"
        r = self.post_audio(HINDI, "hi")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"], "config_error")

    def test_bad_inputs(self):
        os.environ["TRANSCRIPTION_PROVIDER"] = "mock"
        r = self.client.post("/api/transcribe", files={"audio": ("a.m4a", b"", "audio/mp4")}, data={"language": "hi"})
        self.assertEqual((r.status_code, r.json()["error"]), (400, "empty_audio"))
        r = self.post_audio(HINDI, "fr")
        self.assertEqual((r.status_code, r.json()["error"]), (400, "unsupported_language"))


@unittest.skipIf(os.environ.get("MEDSCRIBE_SKIP_NETWORK") == "1", "network tests disabled")
class GroqFailureTests(EnvCase):
    def test_bad_key_returns_error_not_mock(self):
        fake = "gsk_" + "invalid0test0key0for0medscribe0suite"
        os.environ["TRANSCRIPTION_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = fake
        r = self.post_audio(HINDI, "hi")
        self.assertNotEqual(r.status_code, 200, "a failed Groq call must not return a transcript")
        j = r.json()
        self.assertFalse(j["ok"])
        self.assertEqual(j["error"], "groq_auth_failed", j)
        self.assertNotIn("segments", j)
        self.assertNotIn("invalid0test0key", r.text, "key must never be echoed back")


@unittest.skipUnless(REAL_KEY, "GROQ_API_KEY is empty in .env, so the live Groq path is UNTESTED")
class GroqLiveTests(EnvCase):
    def setUp(self):
        super().setUp()
        os.environ["TRANSCRIPTION_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = REAL_KEY

    def _check(self, path, lang):
        r = self.post_audio(path, lang)
        self.assertEqual(r.status_code, 200, r.json().get("message"))
        j = r.json()
        self.assertEqual((j["provider"], j["demo"]), ("groq", False))
        self.assertTrue(j["segments"])
        self.assertTrue(all(s["speaker"] in ("Doctor", "Patient") for s in j["segments"]))
        print(f"\n--- Groq {lang} ({j['model']}, detected={j['detected_language']}) ---\n{j['transcript']}")
        print("medicines:", [(m["name"], m["spoken"], m["suggested"], m["confirmed"]) for m in j["medicines"]])
        return j

    def test_hindi_clip(self):
        self._check(HINDI, "hi")

    def test_marathi_clip(self):
        self._check(MARATHI, "mr")


@unittest.skipUnless(LOCAL_READY, f"local Whisper is not ready ({LOCAL_REASON})")
class LocalWhisperTests(EnvCase):
    def test_auto_without_key_uses_local(self):
        h = self.client.get("/api/health").json()
        self.assertEqual((h["provider"], h["demo"]), ("local", False))

    def _check(self, path, lang):
        r = self.post_audio(path, lang)
        j = r.json()
        self.assertEqual(r.status_code, 200, j.get("message"))
        self.assertEqual((j["provider"], j["demo"]), ("local", False))
        self.assertTrue(j["segments"])
        print(f"\n--- local Whisper {lang} ({j['model']}) ---\n{j['transcript']}")
        print("medicines:", [(m["name"], m["spoken"]) for m in j["medicines"]])

    def test_hindi_clip(self):
        self._check(HINDI, "hi")

    def test_marathi_clip(self):
        self._check(MARATHI, "mr")


class DiarizationTests(unittest.TestCase):
    def test_labels_from_cues_and_turn_taking(self):
        segs = label_speakers([
            Segment(0, 2, "कब से दर्द है?"),
            Segment(2.2, 4, "मुझे दो हफ्ते से घुटने में दर्द है।"),
            Segment(5, 7, "ठीक है।"),                      # no cue, long gap -> speaker changes
            Segment(7.1, 8, "अच्छा।"),                       # no cue, short gap -> same speaker
            Segment(9, 11, "जड वस्तू उचलल्यावर जास्त दुखतं का?"),  # Marathi question by doctor
        ])
        self.assertEqual([s.speaker for s in segs], ["Doctor", "Patient", "Doctor", "Doctor", "Doctor"])
        self.assertEqual([s.speaker_basis for s in segs][2:4], ["turn-taking", "turn-taking"])


class MedicineMatchTests(unittest.TestCase):
    def test_spoken_details_are_kept_and_defaults_are_unconfirmed(self):
        meds = match_medicines("Doctor: Tab Pan 40 khali pet lijiye 10 din. Dolo 650 zarurat ho to.")
        pan, dolo = meds
        self.assertEqual(pan["spoken"], {"strength": "40 mg", "timing": "Before food", "duration": "10 days"})
        self.assertEqual(pan["suggested"], {"frequency": "OD (1-0-0)"})
        self.assertEqual(dolo["spoken"]["frequency"], "SOS")
        for m in meds:
            self.assertFalse(m["confirmed"])

    def test_fully_spoken_medicine_needs_no_suggestion(self):
        [m] = match_medicines("Etoricoxib 90 mg din mein ek baar khane ke baad 5 din")
        self.assertEqual(m["suggested"], {})
        self.assertFalse(m["needs_doctor_confirmation"])

    def test_negated_and_unlisted_medicines_are_not_added(self):
        self.assertEqual(match_medicines("Calcium nahi chahiye. Tab Amoxicillin 500 lijiye."), [])
        self.assertEqual(match_medicines("मला काही औषध नको, फक्त व्यायाम करा."), [])

    def test_extract_endpoint(self):
        r = TestClient(app).post("/api/extract-medicines", json={"transcript": "मी पॅरासिटामॉल देतो, जेवणानंतर घ्या."})
        [m] = r.json()["medicines"]
        self.assertEqual(m["clinic_id"], "paracetamol_650")
        self.assertEqual(m["spoken"], {"timing": "After food"})
        self.assertIn("frequency", m["suggested"])


if __name__ == "__main__":
    unittest.main()
