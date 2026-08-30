"""Parse the raw call metadata JSON and normalise it to the shape the
frontend expects.

The raw files are awkward on purpose:
  - snake_case at the top level, but the frontend type is camelCase
  - the customer's name lives under the key "first and last name" (with spaces)
  - timestamps are epoch milliseconds
  - the agent's survey_response.data is often empty

Join on speaker_id, never on name: names collide across 1,441 calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _party(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise one side (agent or caller) to camelCase."""
    survey = raw.get("survey_response") or {}
    return {
        "arrivalTimeMs": raw.get("arrival_time_ms"),
        "hangupTimeMs": raw.get("hangup_time_ms"),
        # inner metadata keys are left EXACTLY as-is on purpose:
        # "agent_name" and "first and last name" are part of the contract
        "metadata": raw.get("metadata") or {},
        "responses": [
            {"submitTimeMs": r.get("submit_time_ms")} for r in (raw.get("responses") or [])
        ],
        "speakerId": raw.get("speaker_id"),
        "surveyResponse": {
            "submitTimeMs": survey.get("submit_time_ms"),
            "data": survey.get("data") or {},
        },
    }


def normalise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Raw on-disk JSON -> the `CallMetadata` shape the frontend types declare."""
    labels = raw.get("labels") or {}
    return {
        "agent": _party(raw.get("agent") or {}),
        "caller": _party(raw.get("caller") or {}),
        "endTimeMs": raw.get("end_time_ms"),
        "sid": raw.get("sid"),
        "startTimeMs": raw.get("start_time_ms"),
        "labels": {
            "lhvbScript": labels.get("lhvb_script"),
            "callerMos": labels.get("caller_mos"),
            "agentMos": labels.get("agent_mos"),
        },
        "session": raw.get("session"),
    }


@dataclass
class CallFacts:
    """The handful of things derived from metadata that the rest of the
    pipeline actually reasons about."""

    call_id: str
    customer_id: str
    customer_name: str
    agent_id: str
    agent_name: str
    started_at: str | None
    duration_sec: int
    # handle time = agent picked up -> agent hung up
    agent_handle_sec: float
    # how long the caller waited before the conversation started
    caller_wait_sec: float
    partner_rating: int | None
    ease_of_connection: int | None
    caller_mos: float | None
    agent_mos: float | None
    session: str | None
    normalised: dict[str, Any] = field(repr=False, default_factory=dict)


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def extract_facts(raw: dict[str, Any]) -> CallFacts:
    agent = raw.get("agent") or {}
    caller = raw.get("caller") or {}
    labels = raw.get("labels") or {}

    start_ms = raw.get("start_time_ms")
    end_ms = raw.get("end_time_ms")
    duration = int(round((end_ms - start_ms) / 1000)) if start_ms and end_ms else 0

    a_arr, a_hang = agent.get("arrival_time_ms"), agent.get("hangup_time_ms")
    handle = (a_hang - a_arr) / 1000 if a_arr and a_hang else 0.0

    c_arr = caller.get("arrival_time_ms")
    wait = (start_ms - c_arr) / 1000 if start_ms and c_arr else 0.0

    survey = ((caller.get("survey_response") or {}).get("data")) or {}

    return CallFacts(
        call_id=str(raw.get("sid") or ""),
        customer_id=str(caller.get("speaker_id") or ""),
        # the space in this key is not a typo - it is what the files contain
        customer_name=(caller.get("metadata") or {}).get("first and last name") or "Unknown",
        agent_id=str(agent.get("speaker_id") or ""),
        agent_name=(agent.get("metadata") or {}).get("agent_name") or "Unknown",
        started_at=_iso(start_ms),
        duration_sec=duration,
        agent_handle_sec=round(handle, 1),
        caller_wait_sec=round(wait, 1),
        partner_rating=_int_or_none(survey.get("partner_rating")),
        ease_of_connection=_int_or_none(survey.get("ease_of_connection")),
        caller_mos=labels.get("caller_mos"),
        agent_mos=labels.get("agent_mos"),
        session=raw.get("session"),
        normalised=normalise_metadata(raw),
    )


def load_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
