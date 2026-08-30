"""Needs-attention score.

Deliberately NOT produced by the LLM. Two reasons:
  1. Language models are bad at calibrated 0-100 numbers; ask twice and you get
     two different answers.
  2. A judge will ask "why is this call an 87?". A formula answers that; a
     model's opinion does not.

Every point is attributable to a named factor, and the factors are stored so
the UI can show the breakdown with each one linked to its evidence.

Note on `partner_rating`: the customer's own survey score is deliberately kept
OUT of the score. It is the closest thing this dataset has to ground truth, so
it is far more valuable as an independent check (see scripts/validate_scores.py)
than as an input. Using it both ways would be circular.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .taxonomy import HIGH_SEVERITY_TAGS


@dataclass
class ScoreFactor:
    key: str
    points: int
    label: str


@dataclass
class AttentionScore:
    score: int
    factors: list[ScoreFactor] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "factors": [
                {"key": f.key, "points": f.points, "label": f.label} for f in self.factors
            ],
        }


def compute(
    *,
    resolved: bool,
    final_mood: float | None,
    issue_tag: str,
    escalation_requested: bool,
    churn_risk: bool,
    repeat_contact: bool,
    agent_handle_sec: float,
    caller_wait_sec: float,
    dead_air_frac: float,
    mood_drop: float,
    agent_issues: list[str] | None = None,
) -> AttentionScore:
    factors: list[ScoreFactor] = []

    def add(key: str, points: int, label: str) -> None:
        factors.append(ScoreFactor(key, points, label))

    if not resolved:
        add("unresolved", 30, "Issue was not resolved on the call")

    if final_mood is not None and final_mood < config.LOW_MOOD_THRESHOLD:
        add("negative_final_mood", 20, f"Customer ended the call unhappy ({int(final_mood)}/100)")

    if escalation_requested:
        add("escalation_requested", 15, "Customer asked to escalate or speak to a manager")

    if repeat_contact:
        add(
            "repeat_contact",
            15,
            f"Same customer called again within {config.REPEAT_CONTACT_WINDOW_DAYS} days",
        )

    if churn_risk:
        add("churn_risk", 10, "Customer signalled they may leave the bank")

    if issue_tag in HIGH_SEVERITY_TAGS:
        add("high_severity_issue", 10, f"High-severity issue type ({issue_tag})")

    if agent_handle_sec > config.LONG_HANDLE_TIME_SEC:
        add("long_handle_time", 5, f"Long handle time ({int(agent_handle_sec)}s)")

    if dead_air_frac > config.DEAD_AIR_THRESHOLD:
        add("dead_air", 5, f"{int(dead_air_frac * 100)}% of the call was silence")

    if mood_drop >= 20:
        add("sharp_mood_drop", 5, f"Mood fell {int(mood_drop)} points during the call")

    if caller_wait_sec > 30:
        add("long_wait", 5, f"Customer waited {int(caller_wait_sec)}s before an agent joined")

    for issue in agent_issues or []:
        if issue in {"repeated_question", "no_ownership", "interrupted_customer"}:
            add(f"agent_{issue}", 5, f"Agent behaviour: {issue.replace('_', ' ')}")
            break  # cap agent-behaviour contribution at one factor

    total = min(sum(f.points for f in factors), 100)
    return AttentionScore(score=total, factors=factors)
