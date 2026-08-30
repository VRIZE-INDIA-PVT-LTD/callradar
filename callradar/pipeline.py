"""End-to-end pipeline for a single call.

    mp3 + metadata.json
        -> split + loudness-normalise channels        (audio.py)
        -> ASR per channel, merge into numbered turns (transcribe.py)
        -> LLM analysis citing turn ids, validated    (analyze.py)
        -> resolve ids to real timestamps             (analyze.py)
        -> deterministic attention score              (scoring.py)
        -> CallRecord                                 (here)

The output matches the frontend's `CallRecord` TypeScript type exactly, with
`audioUrl` removed as agreed - the UI builds the audio URL from the call id
via GET /api/calls/{id}/audio.
"""
from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import analyze, audio, config, db, scoring
from .metadata import CallFacts, extract_facts
from .transcribe import transcribe_call, turns_to_public


@dataclass
class ProcessResult:
    record: dict[str, Any]
    needs_review: bool
    warnings: list[str]
    debug: dict[str, Any]


def _final_mood(timeline: list[dict]) -> float | None:
    return float(timeline[-1]["mood"]) if timeline else None


def _mood_drop(timeline: list[dict]) -> float:
    """Largest fall from a running peak.

    Must be a DROP, not a range: a call where mood only improves has to score
    zero. The previous version returned max-min, so `neutral -> satisfied` was
    charged 5 attention points for a 30-point "fall" that never happened - and
    the factor label then contradicted the mood labels shown right above it,
    which is exactly the unsupported-evidence case the rubric penalises.
    """
    if len(timeline) < 2:
        return 0.0
    moods = [float(p["mood"]) for p in timeline]
    worst, peak = 0.0, moods[0]
    for m in moods[1:]:
        peak = max(peak, m)
        worst = max(worst, peak - m)
    return worst


def process_call(
    audio_path: str | Path,
    raw_metadata: dict[str, Any],
    *,
    conn: sqlite3.Connection | None = None,
    work_dir: str | Path | None = None,
    asr_backend: str | None = None,
    llm_backend: str | None = None,
) -> ProcessResult:
    facts: CallFacts = extract_facts(raw_metadata)
    call_id = facts.call_id or Path(audio_path).stem

    tmp_holder: tempfile.TemporaryDirectory | None = None
    if work_dir is None:
        tmp_holder = tempfile.TemporaryDirectory(prefix="callradar_")
        work_dir = tmp_holder.name

    try:
        prepared = audio.prepare(audio_path, work_dir, call_id)
        turns = transcribe_call(
            prepared.agent_wav, prepared.customer_wav, backend=asr_backend
        )
        if not turns:
            raise RuntimeError("transcription produced no turns")

        analysis = analyze.analyse(turns, backend=llm_backend)
        data = analysis.raw

        evidence = analyze.build_evidence(data, turns)
        mood_timeline = analyze.build_mood_timeline(data, turns)
        shift_sec = analyze.mood_shift_seconds(data, turns)

        repeat_contact = False
        if conn is not None:
            started_ms = raw_metadata.get("start_time_ms")
            repeat_contact = (
                db.prior_calls_for_customer(
                    conn, facts.customer_id, started_ms, config.REPEAT_CONTACT_WINDOW_DAYS
                )
                > 0
                if started_ms
                else False
            )
        repeat_contact = repeat_contact or bool(data.get("repeat_contact_mentioned"))

        att = scoring.compute(
            resolved=bool(data.get("resolved")),
            final_mood=_final_mood(mood_timeline),
            issue_tag=data.get("issue_tag", "other"),
            escalation_requested=bool(data.get("escalation_requested")),
            churn_risk=bool(data.get("churn_risk")),
            repeat_contact=repeat_contact,
            agent_handle_sec=facts.agent_handle_sec,
            caller_wait_sec=facts.caller_wait_sec,
            dead_air_frac=prepared.stats.dead_air_frac,
            mood_drop=_mood_drop(mood_timeline),
            agent_issues=data.get("agent_issues") or [],
        )

        duration = facts.duration_sec or int(round(prepared.duration_sec))

        record: dict[str, Any] = {
            "id": call_id,
            "customerId": facts.customer_id,
            "customerName": facts.customer_name,
            "agentId": facts.agent_id,
            "agentName": facts.agent_name,
            "startedAt": facts.started_at,
            "durationSec": duration,
            "summary": data.get("summary", ""),
            "intent": data.get("intent", ""),
            "resolved": bool(data.get("resolved")),
            "needsAttention": att.score,
            "moodShiftSec": shift_sec,
            "moodBefore": data.get("mood_before", "neutral"),
            "moodAfter": data.get("mood_after", "neutral"),
            "issueTag": data.get("issue_tag", "other"),
            "transcript": turns_to_public(turns),
            "moodTimeline": mood_timeline,
            "evidence": evidence,
            "metadata": facts.normalised,
            # --- additive, safe to ignore -----------------------------------
            # Not in the original TypeScript type, but extra keys don't break
            # a structurally-typed consumer. This is what lets the UI answer
            # "why is this an 87?" - drop it if the frontend prefers.
            "needsAttentionFactors": att.as_dict()["factors"],
            "needsReview": analysis.needs_review,
        }

        return ProcessResult(
            record=record,
            needs_review=analysis.needs_review,
            warnings=analysis.warnings,
            debug={
                "audio": prepared.as_dict(),
                "turn_count": len(turns),
                "repeat_contact": repeat_contact,
                "partner_rating": facts.partner_rating,
            },
        )
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


def store(conn: sqlite3.Connection, result: ProcessResult, raw_metadata: dict) -> None:
    db.upsert_call(
        conn,
        result.record,
        started_ms=raw_metadata.get("start_time_ms"),
        issue_tag=result.record["issueTag"],
        needs_review=result.needs_review,
        warnings=result.warnings,
    )
