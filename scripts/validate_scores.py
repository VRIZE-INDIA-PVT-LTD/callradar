#!/usr/bin/env python3
"""Check the attention score against the customer's own survey rating.

`caller.survey_response.data.partner_rating` is the closest thing this dataset
has to ground truth, and it is deliberately NOT an input to the score. That
makes it a genuine independent check.

The line you want for the demo is something like:
    "our top-50 flagged calls average a customer rating of 4.1;
     everything else averages 8.7"

    python scripts/validate_scores.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar import db


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _factor_firing_rates(rows) -> None:
    """How often each attention factor actually fires.

    A factor present on every call is a constant offset, not a signal - it
    shifts every score equally and separates nothing. One that almost never
    fires is dead weight. Either way the threshold behind it wants revisiting.
    """
    counts: Counter[str] = Counter()
    total = 0
    for r in rows:
        rec = json.loads(r["record"])
        total += 1
        for f in rec.get("needsAttentionFactors") or []:
            key = f.get("key")
            if key:
                counts[key] += 1
    if not total:
        return

    print(f"\nattention factor firing rates (n={total})")
    if not counts:
        print("  no factors recorded - are these calls from an older schema?")
        return
    for key, n in counts.most_common():
        pct = 100.0 * n / total
        flag = ""
        if pct > 80:
            flag = "  <-- fires on nearly every call: no discriminative signal"
        elif pct < 2:
            flag = "  <-- almost never fires: dead weight"
        print(f"  {key:<28} {n:>5}/{total}  {pct:5.1f}%{flag}")
    print("  Rule of thumb: a factor above ~80% or below ~2% is not separating")
    print("  calls from one another. Revisit its threshold in callradar/config.py.")


def main() -> int:
    conn = db.connect()
    rows = conn.execute(
        "SELECT needs_attention, record FROM calls ORDER BY needs_attention DESC"
    ).fetchall()
    if not rows:
        print("No calls in the database yet. Run scripts/run_batch.py first.")
        return 1

    scored = []
    for r in rows:
        rec = json.loads(r["record"])
        rating = (
            ((rec.get("metadata") or {}).get("caller") or {})
            .get("surveyResponse", {})
            .get("data", {})
            .get("partner_rating")
        )
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            continue
        scored.append((r["needs_attention"], rating))

    print(f"calls with a customer rating: {len(scored)}/{len(rows)}")
    if not scored:
        print("No survey ratings present - skip this check.")
        return 0

    top = scored[: min(50, len(scored))]
    rest = scored[len(top) :]
    print(f"\ntop {len(top)} by attention score : avg customer rating {mean(r for _, r in top):.2f}")
    if rest:
        print(f"remaining {len(rest):>4} calls        : avg customer rating {mean(r for _, r in rest):.2f}")

    # crude rank correlation; should be clearly negative
    n = len(scored)
    if n > 2:
        a = sorted(range(n), key=lambda i: scored[i][0])
        b = sorted(range(n), key=lambda i: scored[i][1])
        ra = {v: i for i, v in enumerate(a)}
        rb = {v: i for i, v in enumerate(b)}
        d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
        rho = 1 - (6 * d2) / (n * (n * n - 1))
        print(f"\nSpearman correlation (score vs rating): {rho:+.3f}")
        print("Expect a NEGATIVE number: higher attention should mean lower satisfaction.")

    _factor_firing_rates(rows)

    buckets: dict[str, list[int]] = {}
    for score, rating in scored:
        key = f"{score // 20 * 20:>3}-{score // 20 * 20 + 19}"
        buckets.setdefault(key, []).append(rating)
    print("\nattention band -> avg customer rating")
    for key in sorted(buckets):
        vals = buckets[key]
        print(f"  {key:>7} : {mean(vals):.2f}   (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
