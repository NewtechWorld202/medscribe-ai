"""Doctor / Patient labelling for transcript segments.

Whisper (Groq) returns timed segments but no speaker identity. This step
labels each segment as "Doctor" or "Patient" from what was said:
- doctor cues: questions about history, instructions, prescribing words
- patient cues: first-person complaints ("मुझे", "मला", "my knee")
When a segment has no clear cue, a pause of PAUSE_TURN_SEC or more is taken as
a change of speaker, and a shorter gap keeps the same speaker.

This works from text, not from the voices, so labels can be wrong. Each
segment's `speaker_basis` says how its label was chosen ("cue" or
"turn-taking") so the review screen can be read with that in mind.
"""
from __future__ import annotations

import re
from typing import List

from .transcription import Segment

PAUSE_TURN_SEC = 0.6
B = r"(?<![\wऀ-ॿ])"   # word boundary that also respects Devanagari
E = r"(?![\wऀ-ॿ])"

DOCTOR_CUES = [
    # prescribing / dosage
    (r"tablets?|tab|capsules?|cap|injection|syrup|gel|ointment|गोली|टैबलेट|टेबलेट|कैप्सूल|इंजेक्शन|सिरप|गोळ्या|गोळी|मलम", 2),
    (r"mg|एमजी|दिन में (?:दो|तीन|एक) बार|दिवसातून|रोज़|रोज", 1),
    (r"x-?ray|एक्स-?रे|mri|एमआरआई|test|टेस्ट|चाचणी|रिपोर्ट|report", 1),
    # instructions (Hindi / Marathi / English imperatives)
    (r"लीजिए|लीजिये|लें|लेना है|कीजिए|कीजिये|करें|करना है|मत|बचें|दिखाइए|आइए|घ्या|करा|टाळा|लावा", 2),
    (r"lijiye|kijiye|kariye|mat|take|apply|avoid|follow[- ]?up|come back|prescrib\w*|लिख रहा|लिख रही|देतो|देते", 2),
    # history-taking questions
    (r"कब से|कितने दिन|क्या तकलीफ|बताइए|बताइये|कधीपासून|किती दिवस|काय त्रास|सांगा|since when|how long|what brings|any other", 2),
]
PATIENT_CUES = [
    (r"मुझे|मेरा|मेरे|मेरी|मैं|मला|माझ्या|माझा|माझी|मी|mujhe|mera|mere|meri|\bmain\b|\bi\b|\bmy\b|i'm|i have", 1),
    (r"दर्द (?:है|हो|होता|रहता)|दुखत[ऀ-ॿ]*|त्रास होतो|takleef|dard ho|hurts|it hurts|pain in my", 2),
    (r"डॉक्टर साहब|डॉक्टर साहेब|डॉक्टर,|doctor sahab|doctor,", 2),
    (r"हाँ|हां|नहीं|हो,|नाही|\byes\b|\bno\b", 1),
]


def _score(text: str, cues) -> int:
    low = text.lower()
    total = 0
    for pattern, weight in cues:
        if re.search(B + "(?:" + pattern + ")" + E, low, flags=re.IGNORECASE):
            total += weight
    return total


def label_speakers(segments: List[Segment]) -> List[Segment]:
    prev_speaker = ""
    prev_end = None
    for seg in segments:
        d = _score(seg.text, DOCTOR_CUES)
        p = _score(seg.text, PATIENT_CUES)
        if seg.text.strip().endswith("?"):
            d += 2  # in a consultation, questions are far more often the doctor's history-taking
            if re.search(r"(?:का|क्या|kya|ka)\s*\?\s*$", seg.text.strip(), flags=re.IGNORECASE):
                d += 1  # yes/no question particle ("... दुखतं का?", "... है क्या?")
        if d != p:
            seg.speaker = "Doctor" if d > p else "Patient"
            seg.speaker_basis = "cue"
        else:
            if not prev_speaker:
                seg.speaker = "Doctor"
            else:
                gap = seg.start - prev_end if prev_end is not None else PAUSE_TURN_SEC
                same = gap < PAUSE_TURN_SEC
                seg.speaker = prev_speaker if same else ("Patient" if prev_speaker == "Doctor" else "Doctor")
            seg.speaker_basis = "turn-taking"
        prev_speaker = seg.speaker
        prev_end = seg.end
    return segments
