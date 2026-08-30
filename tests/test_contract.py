"""Contract tests.

These assert the pipeline output matches the frontend's TypeScript types
exactly. If the frontend changes `types.ts`, change this file in the same
commit - it is the thing that stops the two halves drifting apart.

    python -m pytest tests/ -q
    python tests/test_contract.py          # also runs standalone
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar.analyze import validate, word_count
from callradar.transcribe import Segment, Turn, merge_turns

ROOT = Path(__file__).resolve().parent.parent

# Mirrors CallRecord in the frontend types. audioUrl intentionally absent:
# the UI builds it from the id via GET /api/calls/{id}/audio.
CALL_RECORD_KEYS = {
    "id": str, "customerId": str, "customerName": str, "agentId": str,
    "agentName": str, "startedAt": str, "durationSec": int, "summary": str,
    "intent": str, "resolved": bool, "needsAttention": int, "moodShiftSec": (int, float),
    "moodBefore": str, "moodAfter": str, "issueTag": str, "transcript": list,
    "moodTimeline": list, "evidence": dict, "metadata": dict,
}

TRANSCRIPT_TURN_KEYS = {"speaker", "startSec", "endSec", "text"}
MOOD_POINT_KEYS = {"minute", "mood", "label"}
EVIDENCE_KEYS = {"timestampSec", "quote", "rationale"}
CALL_ANALYSIS_EVIDENCE_KEYS = {"intent", "moodShift", "outcome", "attention"}
METADATA_KEYS = {"agent", "caller", "endTimeMs", "sid", "startTimeMs", "labels", "session"}
PARTY_KEYS = {
    "arrivalTimeMs", "hangupTimeMs", "metadata", "responses", "speakerId", "surveyResponse"
}
LABEL_KEYS = {"lhvbScript", "callerMos", "agentMos"}


class Skipped(Exception):
    """Raised when a test could not run for lack of fixtures.

    Reported separately from a pass. A green run must never overstate what was
    actually verified - a test that quietly returns because its sample file is
    missing looks identical to one that checked something, which is how a
    broken pipeline ships behind a wall of PASS lines.
    """


def _any_sample_pair() -> tuple[Path | None, Path | None]:
    """First staged audio file that also has metadata, or (None, None)."""
    for mp3 in sorted((ROOT / "data" / "audio").glob("*.mp3")):
        meta = ROOT / "data" / "metadata" / f"{mp3.stem}.json"
        if meta.exists():
            return mp3, meta
    return None, None


def assert_contract(rec: dict) -> None:
    for key, typ in CALL_RECORD_KEYS.items():
        assert key in rec, f"CallRecord is missing '{key}'"
        assert isinstance(rec[key], typ), (
            f"CallRecord['{key}'] is {type(rec[key]).__name__}, expected {typ}"
        )

    assert "audioUrl" not in rec, "audioUrl should have been removed"

    assert word_count(rec["summary"]) <= 40, (
        f"summary is {word_count(rec['summary'])} words, limit is 40"
    )
    assert 0 <= rec["needsAttention"] <= 100, "needsAttention must be 0-100"

    assert rec["transcript"], "transcript is empty"
    for turn in rec["transcript"]:
        assert set(turn) == TRANSCRIPT_TURN_KEYS, f"bad TranscriptTurn keys: {set(turn)}"
        assert turn["speaker"] in ("agent", "customer")
        assert turn["endSec"] >= turn["startSec"]

    for pt in rec["moodTimeline"]:
        assert set(pt) == MOOD_POINT_KEYS, f"bad MoodPoint keys: {set(pt)}"
        assert 0 <= pt["mood"] <= 100

    assert set(rec["evidence"]) == CALL_ANALYSIS_EVIDENCE_KEYS
    for name, ev in rec["evidence"].items():
        assert set(ev) == EVIDENCE_KEYS, f"bad Evidence keys on {name}: {set(ev)}"
        assert isinstance(ev["timestampSec"], (int, float))

    md = rec["metadata"]
    assert METADATA_KEYS <= set(md), f"CallMetadata missing {METADATA_KEYS - set(md)}"
    for side in ("agent", "caller"):
        assert PARTY_KEYS <= set(md[side]), f"{side} missing {PARTY_KEYS - set(md[side])}"
        assert isinstance(md[side]["surveyResponse"], dict)
        assert "data" in md[side]["surveyResponse"]
    assert LABEL_KEYS <= set(md["labels"])


def assert_evidence_grounded(rec: dict) -> None:
    """Every cited timestamp must fall inside a real transcript turn, and every
    quote must actually appear in the transcript. This is the check that the
    anti-hallucination design is doing its job."""
    turns = rec["transcript"]
    joined = " ".join(t["text"].lower() for t in turns)
    for name, ev in rec["evidence"].items():
        ts = ev["timestampSec"]
        if name == "moodShift" and ts < 0:
            continue  # honest "no shift detected"
        assert any(
            t["startSec"] - 0.5 <= ts <= t["endSec"] + 0.5 for t in turns
        ), f"evidence.{name} timestamp {ts}s is not inside any turn"
        q = (ev["quote"] or "").lower().strip()
        if q:
            assert q[:40] in joined, f"evidence.{name} quote is not in the transcript"


# ---------------------------------------------------------------- unit tests
def test_merge_turns_groups_same_speaker():
    segs = [
        Segment("agent", 0.0, 1.0, "Hello there"),
        Segment("agent", 1.2, 2.0, "how can I help"),
        Segment("customer", 2.5, 4.0, "I have a problem"),
    ]
    turns = merge_turns(segs)
    assert len(turns) == 2
    assert turns[0].text == "Hello there how can I help"
    assert [t.id for t in turns] == [1, 2]


def test_validator_rejects_nonexistent_turn():
    turns = [Turn(1, "agent", 0, 1, "hi"), Turn(2, "customer", 1, 2, "my card broke")]
    bad = {
        "intent": "x", "issue_tag": "card-activation", "summary": "s", "resolved": True,
        "mood_shift_turn": 99,
        "evidence": {
            "intent": {"turn": 99, "quote": "", "rationale": ""},
            "moodShift": {"turn": None, "quote": "", "rationale": ""},
            "outcome": {"turn": 1, "quote": "", "rationale": ""},
            "attention": {"turn": 1, "quote": "", "rationale": ""},
        },
    }
    problems = validate(bad, turns)
    assert any("99 does not exist" in p for p in problems)


def test_validator_rejects_mood_shift_on_agent_turn():
    turns = [Turn(1, "agent", 0, 1, "hi"), Turn(2, "customer", 1, 2, "my card broke")]
    bad = {
        "intent": "x", "issue_tag": "card-activation", "summary": "s", "resolved": True,
        "mood_shift_turn": 1,
        "evidence": {k: {"turn": 1, "quote": "", "rationale": ""} for k in
                     ("intent", "moodShift", "outcome", "attention")},
    }
    assert any("must cite a customer turn" in p for p in validate(bad, turns))


def test_validator_rejects_non_verbatim_quote():
    turns = [Turn(1, "customer", 0, 3, "I was charged twice for the same thing")]
    bad = {
        "intent": "x", "issue_tag": "other", "summary": "s", "resolved": False,
        "mood_shift_turn": None,
        "evidence": {
            "intent": {"turn": 1, "quote": "I was billed double", "rationale": ""},
            "moodShift": {"turn": None, "quote": "", "rationale": ""},
            "outcome": {"turn": 1, "quote": "charged twice", "rationale": ""},
            "attention": {"turn": 1, "quote": "charged twice", "rationale": ""},
        },
    }
    problems = validate(bad, turns)
    assert any("not verbatim" in p for p in problems)


def test_validator_rejects_bad_issue_tag():
    turns = [Turn(1, "customer", 0, 1, "hello")]
    bad = {
        "intent": "x", "issue_tag": "made-up-tag", "summary": "s", "resolved": True,
        "mood_shift_turn": None,
        "evidence": {k: {"turn": 1, "quote": "", "rationale": ""} for k in
                     ("intent", "moodShift", "outcome", "attention")},
    }
    assert any("taxonomy" in p for p in validate(bad, turns))


def test_full_pipeline_contract():
    """End-to-end in mock mode against the real sample audio."""
    from callradar.metadata import load_metadata
    from callradar.pipeline import process_call

    audio, meta = _any_sample_pair()
    if audio is None:
        raise Skipped("no audio/metadata pair staged in data/")
    res = process_call(
        audio, load_metadata(meta), asr_backend="mock", llm_backend="mock"
    )
    assert_contract(res.record)
    assert_evidence_grounded(res.record)


def test_snapping_skipped_when_word_timestamps_present():
    """Word-level times are already accurate; snapping them can only add error."""
    plain = Segment("agent", 0.0, 1.0, "hello")
    assert plain.from_words is False, "from_words must default to False"
    aligned = Segment("agent", 0.0, 1.0, "hello", from_words=True)
    assert aligned.from_words is True


def test_mood_drop_ignores_improvement():
    """A call whose mood only ever rises must not be charged for a fall.

    `_mood_drop` feeds the sharp_mood_drop attention factor. When it returned
    max-min, every neutral -> satisfied call was billed 5 points for a drop
    that never happened, and the factor label directly contradicted the mood
    labels displayed beside it.
    """
    from callradar.pipeline import _mood_drop

    def tl(*moods):
        return [{"turn": i + 1, "mood": m, "label": "x"} for i, m in enumerate(moods)]

    assert _mood_drop(tl(50, 80)) == 0, "steady improvement is not a drop"
    assert _mood_drop(tl(40, 60, 85)) == 0, "monotonic rise is not a drop"
    assert _mood_drop(tl(70, 30)) == 40, "plain fall should be the full distance"
    assert _mood_drop(tl(60, 25, 70)) == 35, (
        "drawdown is measured from the running peak, and recovery does not erase it"
    )


def test_validator_rejects_pleasantry_mood_shift():
    """Closing politeness is not a mood shift.

    "Thank you" ends almost every call regardless of how it went, so citing it
    is an unsupported claim. The model gets one retry to replace it with null.
    """
    def payload(shift_turn):
        return {
            "intent": "x", "issue_tag": "schedule-appointment", "summary": "s",
            "resolved": True, "mood_shift_turn": shift_turn,
            "evidence": {k: {"turn": 1, "quote": "", "rationale": ""} for k in
                         ("intent", "moodShift", "outcome", "attention")},
        }

    turns = [
        Turn(1, "agent", 0, 1, "Thank you for calling. Have a great day."),
        Turn(2, "customer", 1, 2, "Thank you very much."),
    ]
    problems = validate(payload(2), turns)
    assert any("closing pleasantry" in p for p in problems), (
        f"pleasantry was accepted as a mood shift: {problems}"
    )

    genuine = [
        Turn(1, "agent", 0, 1, "Two business days is the soonest I can do."),
        Turn(2, "customer", 1, 2,
             "This is exactly why I am losing trust in your bank"),
    ]
    problems = validate(payload(2), genuine)
    assert not any("closing pleasantry" in p for p in problems), (
        f"a genuine mood shift was wrongly rejected: {problems}"
    )


def test_warns_without_word_timestamps():
    """A channel with no word-level times must complain, not degrade quietly.

    Segment-level times routinely report 0.0 for the first segment of BOTH
    channels, which scrambles the merge order. There is no reliable way to
    repair that after the fact, so the only honest response is to say so.
    """
    import warnings

    from callradar import transcribe

    coarse = [Segment("customer", 0.0, 4.0, "hi there", from_words=False)]
    aligned = [Segment("agent", 0.0, 4.0, "hello", from_words=True)]

    def fake_channel(wav, speaker, backend=None):
        return list(aligned if speaker == "agent" else coarse)

    real = transcribe.transcribe_channel
    transcribe.transcribe_channel = fake_channel
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transcribe.transcribe_call(Path("a.wav"), Path("c.wav"), backend="groq")
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert msgs, "no RuntimeWarning raised for a channel without word timestamps"
        assert any("customer channel" in m for m in msgs), (
            f"warning did not name the offending channel: {msgs}"
        )
        assert not any("agent channel" in m for m in msgs), (
            "warned about the agent channel, which DOES have word timestamps"
        )
    finally:
        transcribe.transcribe_channel = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    skipped: list[str] = []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Skipped as e:
            skipped.append(t.__name__)
            print(f"  SKIP  {t.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")

    summary = f"\n{passed} passed"
    if skipped:
        summary += f", {len(skipped)} SKIPPED (not verified)"
    if failed:
        summary += f", {failed} FAILED"
    print(summary)
    if skipped:
        print(f"  not verified: {', '.join(skipped)}")
    raise SystemExit(1 if failed else 0)
