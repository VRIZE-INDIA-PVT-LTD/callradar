"""LLM analysis with turn-id citations.

The scoring rubric for this project is blunt: a claim with no evidence scores
zero, and evidence that does not support the claim scores negative. So the
design goal here is not "good summaries", it is "impossible to fabricate a
citation".

How that is achieved:
  1. The model is shown numbered turns and NO timestamps at all.
  2. It may only cite turn ids.
  3. Every cited id is checked against the real transcript. Ids that don't
     exist, or mood shifts attributed to an agent turn, fail validation.
  4. Quotes are checked to be verbatim substrings of the turn they cite.
  5. Timestamps are resolved from the turn in Python, afterwards.

A model that hallucinates simply fails step 3/4 and gets one retry with the
errors fed back; a second failure marks the call `needs_review` rather than
shipping a bad citation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import config
from .taxonomy import ISSUE_TAGS, MOOD_LABELS
from .transcribe import Turn, render_for_llm

SYSTEM_PROMPT = """You are a call-centre quality analyst. You analyse a single \
support call for a consumer bank and return STRICT JSON.

CITATION RULES - these matter more than anything else:
- You are shown the transcript as numbered turns: [1], [2], [3] ...
- Every judgement you make MUST cite the turn number that proves it.
- You must NEVER invent a timestamp. You will not be shown any timestamps.
- Every "quote" must be copied VERBATIM from the turn you cite. Do not
  paraphrase, do not reword, do not join words from different turns.
- A mood shift can only be cited on a CUSTOMER turn.
- If something genuinely is not present in the call (for example the customer's
  mood never shifts), say so honestly: set the value to null. Do NOT invent a
  shift to fill the field. Claiming a shift that did not happen is far worse
  than reporting none.
- A mood shift means a REAL change in how the customer feels about their problem or
  the service: relief after a fix, frustration at being blocked, anger at a refusal.
  Ordinary conversational politeness is NOT a mood shift. "Thank you", "thanks very
  much", "have a nice day", "okay great" at the end of a call are closing pleasantries
  that appear in almost every call regardless of mood. Never cite them as a shift.
- Most routine calls that go smoothly have NO mood shift. Returning
  "mood_shift_turn": null is the expected answer for them, not a failure.

Return ONLY a JSON object. No markdown, no backticks, no commentary."""

USER_TEMPLATE = """Analyse this support call.

ISSUE TAG must be exactly one of:
{tags}

MOOD LABELS should come from:
{moods}

MOOD SCORES are 0-100 where 0 is furious and 100 is delighted. Below 40 is a
negative mood.

TRANSCRIPT:
{transcript}

Return JSON with exactly this shape:
{{
  "intent": "one sentence: what the customer wanted",
  "issue_tag": "<one tag from the list>",
  "resolved": true or false,
  "summary": "<= 40 words, plain, factual",
  "mood_before": "<label>",
  "mood_after": "<label>",
  "mood_shift_turn": <customer turn number, or null if no shift>,
  "mood_timeline": [
    {{"turn": <customer turn number>, "mood": <0-100>, "label": "<label>"}}
  ],
  "escalation_requested": true or false,
  "churn_risk": true or false,
  "repeat_contact_mentioned": true or false,
  "agent_issues": ["repeated_question" | "no_ownership" | "long_silence" | "interrupted_customer"],
  "evidence": {{
    "intent":    {{"turn": <n>, "quote": "verbatim", "rationale": "why this proves the intent"}},
    "moodShift": {{"turn": <n or null>, "quote": "verbatim or empty", "rationale": "..."}},
    "outcome":   {{"turn": <n>, "quote": "verbatim", "rationale": "why this proves resolved/unresolved"}},
    "attention": {{"turn": <n>, "quote": "verbatim", "rationale": "why this call does or does not need a manager"}}
  }}
}}"""

EVIDENCE_KEYS = ["intent", "moodShift", "outcome", "attention"]

# Closing pleasantries. These appear in nearly every call regardless of how it
# went, so citing one as a mood shift is an unsupported claim - the schema has a
# slot for a shift and the model fills it rather than leaving it null.
_PLEASANTRIES = {
    "okay", "ok", "alright", "bye", "goodbye",
    "no that will be all", "no that would be all",
}
_PLEASANTRY_MAX_EXTRA_TOKENS = 6


def _is_closing_pleasantry(text: str) -> bool:
    """True when a turn is nothing but politeness at the end of a call."""
    words = _norm(text).split()
    if not words:
        return False
    if words[0].startswith("thank"):
        return True
    for phrase in _PLEASANTRIES:
        pw = phrase.split()
        if words[: len(pw)] == pw and len(words) - len(pw) <= _PLEASANTRY_MAX_EXTRA_TOKENS:
            return True
    return False


class AnalysisError(RuntimeError):
    pass


@dataclass
class Analysis:
    raw: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    needs_review: bool = False


# ------------------------------------------------------------------ helpers
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise AnalysisError(f"No JSON object in model output: {text[:200]}")
        return json.loads(text[start : end + 1])


def word_count(s: str) -> int:
    return len([w for w in (s or "").split() if w.strip()])


def trim_words(s: str, limit: int = 40) -> str:
    words = (s or "").split()
    if len(words) <= limit:
        return (s or "").strip()
    return " ".join(words[:limit]).rstrip(",.;:") + "."


# --------------------------------------------------------------- validation
def validate(data: dict, turns: list[Turn]) -> list[str]:
    """Return a list of human-readable problems. Empty list == clean."""
    problems: list[str] = []
    by_id = {t.id: t for t in turns}
    valid_ids = set(by_id)

    if not isinstance(data, dict):
        return ["response was not a JSON object"]

    for key in ("intent", "issue_tag", "summary", "resolved", "evidence"):
        if key not in data:
            problems.append(f"missing required field '{key}'")

    if data.get("issue_tag") not in ISSUE_TAGS:
        problems.append(
            f"issue_tag {data.get('issue_tag')!r} is not in the allowed taxonomy"
        )

    if word_count(data.get("summary", "")) > 40:
        problems.append("summary is longer than 40 words")

    shift = data.get("mood_shift_turn")
    if shift is not None:
        if shift not in valid_ids:
            problems.append(f"mood_shift_turn {shift} does not exist")
        elif by_id[shift].speaker != "customer":
            problems.append(
                f"mood_shift_turn {shift} is an agent turn; a mood shift must cite a customer turn"
            )
        elif _is_closing_pleasantry(by_id[shift].text):
            problems.append(
                f"mood_shift_turn {shift} is a closing pleasantry, not a mood shift; "
                f"use null instead"
            )

    for point in data.get("mood_timeline") or []:
        tid = point.get("turn")
        if tid not in valid_ids:
            problems.append(f"mood_timeline cites turn {tid}, which does not exist")
        elif by_id[tid].speaker != "customer":
            problems.append(f"mood_timeline turn {tid} is not a customer turn")
        mood = point.get("mood")
        if not isinstance(mood, (int, float)) or not 0 <= mood <= 100:
            problems.append(f"mood_timeline turn {tid} has mood {mood!r}, expected 0-100")

    ev = data.get("evidence") or {}
    for key in EVIDENCE_KEYS:
        item = ev.get(key)
        if not isinstance(item, dict):
            problems.append(f"evidence.{key} is missing")
            continue
        tid = item.get("turn")
        if key == "moodShift" and tid is None:
            continue  # honest "no shift" is allowed
        if tid not in valid_ids:
            problems.append(f"evidence.{key} cites turn {tid}, which does not exist")
            continue
        quote = item.get("quote") or ""
        if quote and _norm(quote) not in _norm(by_id[tid].text):
            problems.append(
                f"evidence.{key} quote is not verbatim from turn {tid}"
            )
    return problems


# ------------------------------------------------------------------ backends
def _call_groq(system: str, user: str) -> str:
    from .transcribe import _groq

    resp = _groq().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def _call_mock(system: str, user: str) -> str:
    """Offline stand-in that respects the citation contract.

    It reads the numbered transcript back out of the prompt and cites real
    turns, so validation genuinely exercises itself in mock mode.
    """
    turns = re.findall(r"^\[(\d+)\] (AGENT|CUSTOMER): (.*)$", user, flags=re.MULTILINE)
    cust = [(int(i), t) for i, s, t in turns if s == "CUSTOMER"]
    agent = [(int(i), t) for i, s, t in turns if s == "AGENT"]
    if not cust:
        cust = [(1, turns[0][2] if turns else "")]
    first_c, first_text = cust[0]
    last_c, last_text = cust[-1]
    last_a, last_a_text = agent[-1] if agent else (first_c, first_text)

    def first_words(text: str, n: int = 8) -> str:
        return " ".join((text or "").split()[:n])

    payload = {
        "intent": "Customer reports being charged twice and wants the duplicate reversed.",
        "issue_tag": "duplicate-charge-dispute",
        "resolved": False,
        "summary": (
            "Customer disputed a duplicate charge on a repeat call. Agent raised a "
            "dispute case and promised back-office follow-up within two business "
            "days; customer remained dissatisfied and trust was damaged."
        ),
        "mood_before": "frustrated",
        "mood_after": "angry",
        "mood_shift_turn": last_c,
        "mood_timeline": [
            {"turn": first_c, "mood": 44, "label": "frustrated"},
            {"turn": last_c, "mood": 27, "label": "angry"},
        ],
        "escalation_requested": False,
        "churn_risk": True,
        "repeat_contact_mentioned": True,
        "agent_issues": ["no_ownership"],
        "evidence": {
            "intent": {
                "turn": first_c,
                "quote": first_words(first_text),
                "rationale": "Customer states the duplicate charge as the reason for calling.",
            },
            "moodShift": {
                "turn": last_c,
                "quote": first_words(last_text),
                "rationale": "Language escalates from frustration to loss of trust.",
            },
            "outcome": {
                "turn": last_a,
                "quote": first_words(last_a_text),
                "rationale": "Action was initiated but the outcome is still pending.",
            },
            "attention": {
                "turn": last_c,
                "quote": first_words(last_text),
                "rationale": "Unresolved financial dispute with explicit loss of trust.",
            },
        },
    }
    return json.dumps(payload)


_LLM_BACKENDS = {"groq": _call_groq, "mock": _call_mock}


# -------------------------------------------------------------------- driver
def analyse(turns: list[Turn], backend: str | None = None) -> Analysis:
    backend = backend or config.LLM_BACKEND
    if backend not in _LLM_BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND {backend!r}")
    call = _LLM_BACKENDS[backend]

    user = USER_TEMPLATE.format(
        tags="\n".join(f"  - {t}" for t in ISSUE_TAGS),
        moods=", ".join(MOOD_LABELS),
        transcript=render_for_llm(turns),
    )

    last_problems: list[str] = []
    for attempt in range(config.LLM_MAX_RETRIES):
        prompt = user
        if last_problems:
            prompt += (
                "\n\nYour previous answer was rejected for these reasons:\n"
                + "\n".join(f"  - {p}" for p in last_problems)
                + "\nFix them. Cite only turn numbers that appear above, and copy "
                  "quotes verbatim."
            )
        try:
            data = _extract_json(call(SYSTEM_PROMPT, prompt))
        except (AnalysisError, json.JSONDecodeError) as exc:
            last_problems = [f"output was not valid JSON: {exc}"]
            continue

        problems = validate(data, turns)
        if not problems:
            data["summary"] = trim_words(data.get("summary", ""), 40)
            return Analysis(raw=data, warnings=[], needs_review=False)
        last_problems = problems

    # Both attempts failed. Keep whatever we have but flag it loudly rather
    # than silently shipping an unsupported citation.
    return Analysis(raw=data if "data" in dir() else {}, warnings=last_problems, needs_review=True)


# --------------------------------------------------- resolve ids -> seconds
def resolve_quote_time(turn: Turn, quote: str) -> float:
    """Turn id -> a real timestamp.

    If the quote sits partway through a long turn, interpolate proportionally
    by character position so the player seeks to roughly the right moment
    rather than the start of the whole turn.
    """
    if not quote:
        return turn.startSec
    haystack, needle = _norm(turn.text), _norm(quote)
    idx = haystack.find(needle)
    if idx <= 0 or not haystack:
        return turn.startSec
    span = max(turn.endSec - turn.startSec, 0.0)
    return round(turn.startSec + span * (idx / len(haystack)), 2)


def build_evidence(data: dict, turns: list[Turn]) -> dict[str, dict]:
    """Produce the frontend's `CallAnalysisEvidence` with real timestamps."""
    by_id = {t.id: t for t in turns}
    ev_in = data.get("evidence") or {}
    out: dict[str, dict] = {}

    for key in EVIDENCE_KEYS:
        item = ev_in.get(key) or {}
        tid = item.get("turn")
        turn = by_id.get(tid)
        if turn is None:
            out[key] = {
                "timestampSec": config.NO_MOOD_SHIFT_SEC if key == "moodShift" else 0,
                "quote": "",
                "rationale": (
                    "No mood shift detected in this call."
                    if key == "moodShift"
                    else "No supporting turn was cited for this judgement."
                ),
            }
            continue
        quote = (item.get("quote") or "").strip()
        # never ship a quote we could not verify against the transcript
        if quote and _norm(quote) not in _norm(turn.text):
            quote = turn.text
        out[key] = {
            "timestampSec": resolve_quote_time(turn, quote),
            "quote": quote or turn.text,
            "rationale": (item.get("rationale") or "").strip(),
        }
    return out


def build_mood_timeline(data: dict, turns: list[Turn]) -> list[dict]:
    """`MoodPoint[]` for the chart.

    Calls in this dataset average 45-60 seconds, so integer-minute bucketing
    collapses almost every call to a single point. Default mode is therefore
    one point per customer turn, with `minute` expressed as a fractional
    minute (still a number, still plots correctly). Set
    MOOD_TIMELINE_MODE=minute for the original integer-bucket behaviour.
    """
    by_id = {t.id: t for t in turns}
    points: list[tuple[float, float, str]] = []
    for p in data.get("mood_timeline") or []:
        turn = by_id.get(p.get("turn"))
        if turn is None:
            continue
        points.append((turn.startSec, float(p.get("mood", 50)), p.get("label") or "neutral"))

    points.sort(key=lambda x: x[0])
    if not points:
        return []

    if config.MOOD_TIMELINE_MODE == "minute":
        buckets: dict[int, list[tuple[float, str]]] = {}
        for sec, mood, label in points:
            buckets.setdefault(int(sec // 60), []).append((mood, label))
        return [
            {
                "minute": minute,
                "mood": round(sum(m for m, _ in vals) / len(vals)),
                "label": vals[-1][1],
            }
            for minute, vals in sorted(buckets.items())
        ]

    return [
        {"minute": round(sec / 60, 2), "mood": round(mood), "label": label}
        for sec, mood, label in points
    ]


def mood_shift_seconds(data: dict, turns: list[Turn]) -> float:
    by_id = {t.id: t for t in turns}
    turn = by_id.get(data.get("mood_shift_turn"))
    return turn.startSec if turn else config.NO_MOOD_SHIFT_SEC
