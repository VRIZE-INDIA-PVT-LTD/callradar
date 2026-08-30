"""SQLite storage.

1,441 rows is a rounding error, so a single file beats a database server here:
zero setup for whoever runs the repo, it can be committed alongside the code,
and FTS5 gives full-text search over every transcript for free.

The whole `CallRecord` is stored as JSON in `calls.record`, with the fields the
API filters or sorts on lifted into real columns.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customer (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calls (
    id                TEXT PRIMARY KEY,
    customer_id       TEXT REFERENCES customer(id),
    agent_id          TEXT REFERENCES agent(id),
    audio_mp3         BLOB,
    started_at        TEXT,
    started_ms        INTEGER,
    duration_sec      INTEGER,
    intent            TEXT,
    issue_tag         TEXT,
    resolved          INTEGER,
    needs_attention   INTEGER,
    mood_shift_sec    REAL,
    needs_review      INTEGER DEFAULT 0,
    warnings          TEXT,
    record            TEXT NOT NULL,

    -- Nested/array fields stored as JSON text
    transcript      TEXT    NOT NULL CHECK (json_valid(transcript)),
    moodTimeline    TEXT    NOT NULL CHECK (json_valid(moodTimeline)),
    evidence        TEXT    NOT NULL CHECK (json_valid(evidence)),
    metadata        TEXT    NOT NULL CHECK (json_valid(metadata)),

    createdAtUtc    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    call_id   TEXT PRIMARY KEY,
    status    TEXT NOT NULL,       -- pending | done | failed
    error     TEXT,
    updated_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts USING fts5(
    call_id UNINDEXED,
    speaker UNINDEXED,
    text
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_call(
    conn: sqlite3.Connection,
    record: dict[str, Any],
    *,
    started_ms: int | None,
    issue_tag: str,
    audio_mp3: bytes | None = None,
    needs_review: bool = False,
    warnings: list[str] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO calls (id, customer_id, agent_id, audio_mp3,
                           started_at, started_ms, duration_sec, intent, issue_tag,
                           resolved, needs_attention, mood_shift_sec,
                           needs_review, warnings, record,
                           transcript, moodTimeline, evidence, metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            customer_id=excluded.customer_id,
            agent_id=excluded.agent_id,
            audio_mp3=excluded.audio_mp3,
            started_at=excluded.started_at, started_ms=excluded.started_ms,
            duration_sec=excluded.duration_sec, intent=excluded.intent,
            issue_tag=excluded.issue_tag, resolved=excluded.resolved,
            needs_attention=excluded.needs_attention,
            mood_shift_sec=excluded.mood_shift_sec,
            needs_review=excluded.needs_review, warnings=excluded.warnings,
            record=excluded.record,
            transcript=excluded.transcript,
            moodTimeline=excluded.moodTimeline,
            evidence=excluded.evidence,
            metadata=excluded.metadata
        """,
        (
            record["id"], record["customerId"],
            record["agentId"], audio_mp3,
            record["startedAt"],
            started_ms, record["durationSec"], record["intent"], issue_tag,
            1 if record["resolved"] else 0, record["needsAttention"],
            record["moodShiftSec"], 1 if needs_review else 0,
            json.dumps(warnings or []), json.dumps(record),
            json.dumps(record.get("transcript", [])),
            json.dumps(record.get("moodTimeline", [])),
            json.dumps(record.get("evidence", [])),
            json.dumps(record.get("metadata", {})),
        ),
    )
    conn.execute("DELETE FROM transcripts WHERE call_id = ?", (record["id"],))
    conn.executemany(
        "INSERT INTO transcripts (call_id, speaker, text) VALUES (?,?,?)",
        [(record["id"], t["speaker"], t["text"]) for t in record.get("transcript", [])],
    )
    conn.commit()


def get_call(conn: sqlite3.Connection, call_id: str) -> dict | None:
    row = conn.execute("SELECT record FROM calls WHERE id = ?", (call_id,)).fetchone()
    return json.loads(row["record"]) if row else None


def all_calls(conn: sqlite3.Connection) -> list[dict]:
    return [
        json.loads(r["record"])
        for r in conn.execute("SELECT record FROM calls ORDER BY started_ms DESC")
    ]


def prior_calls_for_customer(
    conn: sqlite3.Connection, customer_id: str, before_ms: int, window_days: int
) -> int:
    """Used by the repeat-contact factor in the attention score."""
    if not customer_id or before_ms is None:
        return 0
    since = before_ms - window_days * 86400 * 1000
    row = conn.execute(
        "SELECT COUNT(*) c FROM calls WHERE customer_id = ? AND started_ms < ? AND started_ms >= ?",
        (customer_id, before_ms, since),
    ).fetchone()
    return int(row["c"])


# -------------------------------------------------------------------- jobs
def set_job(conn: sqlite3.Connection, call_id: str, status: str, error: str | None = None) -> None:
    conn.execute(
        """INSERT INTO jobs (call_id, status, error, updated_at)
           VALUES (?,?,?,datetime('now'))
           ON CONFLICT(call_id) DO UPDATE SET
             status=excluded.status, error=excluded.error, updated_at=excluded.updated_at""",
        (call_id, status, error),
    )
    conn.commit()


def done_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["call_id"] for r in conn.execute("SELECT call_id FROM jobs WHERE status='done'")}


def search_transcripts(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        """SELECT t.call_id, t.speaker, snippet(transcripts, 2, '[', ']', '...', 12) AS snip
           FROM transcripts t WHERE transcripts MATCH ? LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]
