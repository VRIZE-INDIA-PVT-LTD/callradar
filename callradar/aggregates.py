"""Cross-call views: trending issues and per-agent performance.

These only work because intents come from a frozen taxonomy. Free-text intents
would give ~900 unique strings and nothing countable.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def trending_issues(conn: sqlite3.Connection, window_days: int = 7) -> list[dict]:
    """`TrendingIssue[]`: count this window vs the one before it."""
    rows = conn.execute("SELECT issue_tag, started_at FROM calls").fetchall()
    now = datetime.now(timezone.utc)
    cur_start = now - timedelta(days=window_days)
    prev_start = now - timedelta(days=window_days * 2)

    # If the corpus is historical (this dataset is from 2020), anchor the
    # windows to the newest call instead of wall-clock now, so the demo shows
    # real movement rather than two empty windows.
    stamps = [d for d in (_parse(r["started_at"]) for r in rows) if d]
    if stamps:
        newest = max(stamps)
        if newest < cur_start:
            now = newest
            cur_start = now - timedelta(days=window_days)
            prev_start = now - timedelta(days=window_days * 2)

    cur: dict[str, int] = {}
    prev: dict[str, int] = {}
    for r in rows:
        tag = r["issue_tag"] or "other"
        when = _parse(r["started_at"])
        if when is None:
            continue
        if when >= cur_start:
            cur[tag] = cur.get(tag, 0) + 1
        elif when >= prev_start:
            prev[tag] = prev.get(tag, 0) + 1

    if not cur and not prev:  # no usable dates: fall back to a flat count
        for r in rows:
            tag = r["issue_tag"] or "other"
            cur[tag] = cur.get(tag, 0) + 1

    return sorted(
        (
            {
                "issueTag": tag,
                "count": count,
                "deltaVsLastWeek": count - prev.get(tag, 0),
            }
            for tag, count in cur.items()
        ),
        key=lambda d: d["count"],
        reverse=True,
    )


def agent_metrics(conn: sqlite3.Connection) -> list[dict]:
    """`AgentMetric[]`: volume, average handle time, resolution rate."""
    rows = conn.execute(
        """SELECT c.agent_id AS agent_id, a.name AS agent_name,
                  COUNT(*)                       AS call_volume,
                  AVG(c.duration_sec)            AS avg_handle,
                  AVG(CASE WHEN c.resolved=1 THEN 1.0 ELSE 0.0 END) AS resolved_pct,
                  SUM(CASE WHEN c.needs_attention >= 70 THEN 1 ELSE 0 END) AS escalations
           FROM calls c
           JOIN agent a ON a.id = c.agent_id
           GROUP BY c.agent_id, a.name
           ORDER BY call_volume DESC"""
    ).fetchall()
    return [
        {
            "agentId": r["agent_id"],
            "agentName": r["agent_name"],
            "callVolume": r["call_volume"],
            "avgHandleTimeSec": round(r["avg_handle"] or 0),
            "resolvedPct": round((r["resolved_pct"] or 0) * 100, 1),
            "escalations": r["escalations"] or 0,
        }
        for r in rows
    ]


def customers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT c.customer_id AS customer_id, cu.name AS customer_name,
                  COUNT(*) AS calls,
                  MAX(c.needs_attention) AS worst
           FROM calls c
           JOIN customer cu ON cu.id = c.customer_id
           GROUP BY c.customer_id, cu.name
           ORDER BY cu.name"""
    ).fetchall()
    return [
        {
            "id": r["customer_id"],
            "name": r["customer_name"],
            "callCount": r["calls"],
            "worstAttention": r["worst"],
        }
        for r in rows
    ]


def agents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT DISTINCT c.agent_id AS agent_id, a.name AS agent_name
           FROM calls c JOIN agent a ON a.id = c.agent_id
           ORDER BY a.name"""
    ).fetchall()
    return [{"id": r["agent_id"], "name": r["agent_name"]} for r in rows]
