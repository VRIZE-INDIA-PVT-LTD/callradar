#!/usr/bin/env python3
"""Read a processed call back out of the database, in human-readable form.

This is the thing to run after `make sample`. It answers two questions the
numbers cannot:

  1. Did the turn ordering come out right? (agent greeting first, customer
     reply after it, timestamps that increase)
  2. Are the channels the right way round? The greeting belongs to the AGENT.
     If it lands on the customer side, set LEFT_CHANNEL_SPEAKER=customer.

    python scripts/show_transcript.py                 # top 3 by attention
    python scripts/show_transcript.py --limit 10
    python scripts/show_transcript.py --id 0a37e848a9c04e0c --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar import db

# Phrases only an agent says. Used for the channel sanity check at the end.
GREETING_PHRASES = (
    "thank you for calling",
    "this is",
    "how can i help",
    "my name is",
)


def show_call(rec: dict, full: bool) -> None:
    print("=" * 78)
    print(f"call {rec.get('id')}")
    print(f"  customer      : {rec.get('customerName')}")
    print(f"  agent         : {rec.get('agentName')}")
    print(f"  intent        : {rec.get('intent')}")
    print(f"  issueTag      : {rec.get('issueTag')}")
    print(f"  resolved      : {rec.get('resolved')}")
    print(f"  needsAttention: {rec.get('needsAttention')}")

    shift = rec.get("moodShiftSec", -1)
    mood = f"{rec.get('moodBefore')} -> {rec.get('moodAfter')}"
    if shift is None or shift < 0:
        print(f"  mood          : {mood}  (shift: none detected)")
    else:
        print(f"  mood          : {mood}  (shift at {shift:.1f}s)")

    print(f"  summary       : {rec.get('summary')}")

    turns = rec.get("transcript") or []
    shown = turns if full else turns[:8]
    print(f"\n  transcript ({len(shown)} of {len(turns)} turns):")
    for t in shown:
        print(
            f"    {t['speaker']:<9} {t['startSec']:>6.1f}s  {t['text'][:95]}"
        )
    if not full and len(turns) > len(shown):
        print(f"    ... {len(turns) - len(shown)} more (use --full)")

    if full:
        print("\n  evidence:")
        for name, ev in (rec.get("evidence") or {}).items():
            print(f"    {name}:")
            print(f"      at       : {ev.get('timestampSec')}s")
            print(f"      quote    : {ev.get('quote')}")
            print(f"      rationale: {ev.get('rationale')}")

        factors = rec.get("needsAttentionFactors") or []
        if factors:
            print("\n  needsAttentionFactors:")
            for f in factors:
                print(f"    {f.get('points'):>+4}  {f.get('key')}: {f.get('label')}")
    print()


def channel_sanity_check(records: list[dict]) -> None:
    """Count agent-style greeting phrases on each side.

    They belong to the agent. If the customer-labelled channel is winning,
    the stereo split is the wrong way round.
    """
    agent_hits = customer_hits = 0
    for rec in records:
        for t in rec.get("transcript") or []:
            text = (t.get("text") or "").lower()
            hits = sum(1 for phrase in GREETING_PHRASES if phrase in text)
            if t.get("speaker") == "agent":
                agent_hits += hits
            else:
                customer_hits += hits

    print("=" * 78)
    print("channel sanity check (agent greeting phrases)")
    print(f"  on the 'agent' side    : {agent_hits}")
    print(f"  on the 'customer' side : {customer_hits}")
    if customer_hits > agent_hits:
        print("\n  >>> WARNING: the greeting is landing on the CUSTOMER channel.")
        print("  >>> The channels look swapped. Set LEFT_CHANNEL_SPEAKER=customer")
        print("  >>> in .env and re-run the pipeline.")
    else:
        print("  looks right: the greeting is on the agent channel.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="call_id", help="show one specific call id")
    ap.add_argument("--limit", type=int, default=3, help="how many calls (default 3)")
    ap.add_argument("--full", action="store_true", help="all turns + evidence + factors")
    args = ap.parse_args()

    conn = db.connect()

    if args.call_id:
        rec = db.get_call(conn, args.call_id)
        if rec is None:
            print(f"No call with id {args.call_id!r} in the database.")
            return 1
        records = [rec]
    else:
        rows = conn.execute(
            "SELECT id FROM calls ORDER BY needs_attention DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        if not rows:
            print("Database is empty. Run: make sample")
            return 1
        records = [r for r in (db.get_call(conn, row["id"]) for row in rows) if r]

    for rec in records:
        show_call(rec, args.full)

    channel_sanity_check(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
