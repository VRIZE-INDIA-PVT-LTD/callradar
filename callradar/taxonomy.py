"""Fixed issue taxonomy.

Free-text intents give you ~900 unique strings across 1,441 calls and nothing
you can count. A frozen list gives a real trending chart with week-over-week
deltas. The LLM must pick one of these tags; anything else is rejected by the
validator and falls back to "other".

Derive/extend this by sampling ~100 real calls first, then freeze it before
the bulk run.
"""

ISSUE_TAGS = [
    "schedule-appointment",
    "cancel-or-reschedule-appointment",
    "transfer-funds",
    "check-balance",
    "recent-transactions-query",
    "pay-bill",
    "order-checks",
    "replace-card",
    "card-activation",
    "report-lost-or-stolen-card",
    "update-contact-details",
    "open-account",
    "close-account",
    "branch-hours-or-location",
    "loan-or-mortgage-enquiry",
    "password-or-online-access",
    "duplicate-charge-dispute",
    "unauthorised-transaction",
    "failed-or-pending-transfer",
    "overdraft-or-nsf-fee",
    "fraud-alert-verification",
    "general-enquiry",
    "other",
]

# Tags that raise the needs-attention score on their own.
HIGH_SEVERITY_TAGS = {
    "unauthorised-transaction",
    "fraud-alert-verification",
    "report-lost-or-stolen-card",
    "duplicate-charge-dispute",
    "close-account",
}

MOOD_LABELS = [
    "angry",
    "frustrated",
    "tense",
    "anxious",
    "neutral",
    "cautiously calm",
    "calm",
    "satisfied",
    "delighted",
]
