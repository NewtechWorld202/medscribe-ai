"""Match a transcript against the clinic medicine list.

Rules (deliberate, do not relax):
- A medicine is returned only if it was actually named in the transcript.
- Anything heard in the transcript near the mention (strength, frequency,
  timing, duration) goes under `spoken`.
- A clinic default is used only for a field that was NOT heard. It goes under
  `suggested`, marked `confirmed: false`, and must be confirmed by the doctor
  in the approval step. It is never presented as what the doctor said.
- A mention followed by a negation ("nahi", "नहीं", "नको", "don't") is skipped.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

LIST_PATH = Path(__file__).resolve().parent / "clinic_medicines.json"
DEV = "ऀ-ॿ"
B = rf"(?<![\w{DEV}])"
E = rf"(?![\w{DEV}])"
WINDOW = 120  # characters after a mention that count as its instructions

FIELDS = ("strength", "frequency", "timing", "duration")

FREQ = [
    (r"weekly|hafte mein ek|हफ्ते में एक|हफ़्ते में एक|आठवड्यातून एक", "Once weekly"),
    (r"\bsos\b|zarurat|जरूरत|ज़रूरत|गरज असेल|if needed|when required", "SOS"),
    (r"teen baar|thrice|\btds\b|din mein 3|तीन बार|दिवसातून तीन|तीनदा|1-1-1", "TDS (1-1-1)"),
    (r"subah[\s,-]*(aur\s+)?shaam|twice|\bbd\b|do baar|दो बार|सुबह.{0,8}शाम|दिवसातून दोन|दोनदा|सकाळ.{0,10}संध्याकाळ|1-0-1", "BD (1-0-1)"),
    (r"raat|night|bedtime|sone se pehle|रात|सोने से पहले|रात्री|झोपण्यापूर्वी|0-0-1", "HS (0-0-1)"),
    (r"ek baar|once|\bod\b|daily|roz|रोज़|रोज|एक बार|दिवसातून एक|एकदा|1-0-0", "OD (1-0-0)"),
]
TIMING = [
    (r"khaa?ne ke baad|after (food|meal)|खाने के बाद|भोजन के बाद|nashte ke baad|जेवणानंतर|जेवणा नंतर|जेवल्यानंतर|जेवणाच्या नंतर", "After food"),
    (r"khaa?ne se pehle|khaa?li pet|before (food|meal)|empty stomach|खाने से पहले|खाली पेट|जेवणापूर्वी|जेवणाआधी|उपाशी पोटी", "Before food"),
    (r"sone se pehle|bedtime|सोने से पहले|झोपण्यापूर्वी", "At bedtime"),
    (r"lagaiye|lagana|apply|लगाइए|लगाना|लावा|massage|मालिश", "Local application"),
]
NUM = {"ek": 1, "do": 2, "teen": 3, "chaar": 4, "char": 4, "paanch": 5, "panch": 5, "saat": 7, "das": 10, "pandrah": 15,
       "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "seven": 7, "ten": 10, "fifteen": 15,
       "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "सात": 7, "दस": 10, "पंद्रह": 15,
       "दोन": 2, "पाच": 5, "दहा": 10, "पंधरा": 15}
NUM_ALT = r"\d+|" + "|".join(sorted(map(re.escape, NUM), key=len, reverse=True))
UNIT = [
    (r"din|dino|days?|दिन|दिनों|दिवस|दिवसांसाठी|दिवसां", "day"),
    (r"hafte|hafta|hafton|weeks?|हफ्ते|हफ़्ते|हफ्ता|सप्ताह|आठवडे|आठवड्यां|आठवडा", "week"),
    (r"mahine|mahina|months?|महीने|महीना|महिने|महिना", "month"),
]
DUR_RE = re.compile(B + r"(" + NUM_ALT + r")\s*(" + "|".join(u for u, _ in UNIT) + r")", re.IGNORECASE)
STRENGTH_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mg|mcg|iu|ml|g|एमजी|मिलीग्राम|मिलिग्रॅम)?(?![\w" + DEV + r"])", re.IGNORECASE)
NEGATION_RE = re.compile(r"^[\s,]{0,3}(?:\S+\s+){0,3}?(nahi|nahin|mat|band|stop|don't|do not|नहीं|नही|मत|बंद|नको|नका|देऊ नका)(?![\w" + DEV + r"])", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_clinic_list() -> dict:
    return json.loads(LIST_PATH.read_text(encoding="utf-8"))


def _num(token: str) -> int:
    return int(token) if token.isdigit() else NUM.get(token.lower(), NUM.get(token, 0))


def _unit(token: str) -> str:
    for pattern, unit in UNIT:
        if re.fullmatch(pattern, token, flags=re.IGNORECASE):
            return unit
    return "day"


def _first(patterns, text: str) -> str:
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return ""


def _spoken_details(after: str) -> dict:
    out = {f: "" for f in FIELDS}
    lead = after.lstrip()
    m = STRENGTH_RE.match(lead)
    if m and not m.group(2) and re.match(r"\s*(" + "|".join(u for u, _ in UNIT) + r"|baar|बार|times|वेळा|दा)", lead[m.end():], re.IGNORECASE):
        m = None  # "5 din" / "2 baar" is a duration or frequency, not a strength
    if m and (m.group(2) or int(float(m.group(1))) >= 5):
        unit = (m.group(2) or "mg").lower().replace("एमजी", "mg").replace("मिलीग्राम", "mg").replace("मिलिग्रॅम", "mg")
        out["strength"] = f"{m.group(1)} {unit}"
    out["frequency"] = _first(FREQ, after)
    out["timing"] = _first(TIMING, after)
    for d in DUR_RE.finditer(after):
        tail = after[d.end(): d.end() + 12]
        if re.match(r"\s*(ke\s+)?(baad|बाद|later|नंतर)", tail, flags=re.IGNORECASE):
            continue  # "come after 2 weeks" is a follow-up, not a course length
        n = _num(d.group(1))
        if n:
            u = _unit(d.group(2))
            out["duration"] = f"{n} {u}{'s' if n > 1 else ''}"
            break
    return out


def _alias_regex(alias: str) -> re.Pattern:
    parts = [re.escape(p) for p in alias.lower().split()]
    return re.compile(B + r"[\s\-]*".join(parts) + E, re.IGNORECASE)


def match_medicines(transcript: str, clinic: Optional[dict] = None) -> List[dict]:
    clinic = clinic or load_clinic_list()
    text = transcript or ""
    low = text.lower()
    hits = []
    for med in clinic["medicines"]:
        best = None
        for alias in sorted(med["aliases"], key=len, reverse=True):
            for m in _alias_regex(alias).finditer(low):
                if best is None or m.start() < best[0].start():
                    best = (m, alias)
                break
        if best:
            hits.append((best[0].start(), best[0].end(), best[1], med))
    hits.sort(key=lambda h: h[0])

    # Drop a shorter alias match that sits inside a longer one ("diclofenac" inside "diclofenac gel").
    filtered = []
    for h in hits:
        if any(o is not h and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in hits):
            continue
        filtered.append(h)

    results = []
    for i, (start, end, alias, med) in enumerate(filtered):
        next_start = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        after = text[end: min(next_start, end + WINDOW)].split("\n")[0]
        # Instructions often follow the LAST medicine in a list ("X and Y, after food").
        if i + 1 < len(filtered) and not after.strip(" ,और and आणिअनि"):
            after = text[filtered[-1][1]: filtered[-1][1] + WINDOW].split("\n")[0]
        if NEGATION_RE.match(text[end: end + 40]):
            continue
        spoken = _spoken_details(after)
        # A strength said as part of the name ("Pan 40", "Dolo 650") was spoken.
        said_num = re.search(r"\d+", text[start:end])
        if said_num and not spoken["strength"]:
            spoken["strength"] = med["strength"] if med["strength"].startswith(said_num.group()) else f"{said_num.group()} mg"
        defaults = {"strength": med["strength"], "frequency": med["default_dose"],
                    "timing": med["timing"], "duration": med["duration"]}
        suggested = {f: defaults[f] for f in FIELDS if not spoken[f] and defaults.get(f)}
        results.append({
            "clinic_id": med["id"],
            "name": med["name"],
            "mentioned_as": text[start:end],
            "spoken": {f: v for f, v in spoken.items() if v},
            "suggested": suggested,
            "suggestion_source": "clinic_default" if suggested else "",
            "confirmed": False,
            "needs_doctor_confirmation": bool(suggested),
        })
    return results
