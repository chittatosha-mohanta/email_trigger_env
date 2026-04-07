from __future__ import annotations

from typing import Dict, List, Tuple

from app.models import SupportAction, TaskDescriptor, Ticket


TASKS: Dict[str, Dict] = {
    "easy_priority_routing": {
        "descriptor": TaskDescriptor(
            id="easy_priority_routing",
            title="Basic Billing Priority Routing",
            difficulty="easy",
            goal=(
                "Classify a billing issue, set appropriate priority/team, and send a polite"
                " response with an actionable next step."
            ),
        ),
        "ticket": Ticket(
            id="TKT-1001",
            customer_tier="pro",
            subject="Charged twice for monthly plan",
            body=(
                "I upgraded to Pro yesterday and now I see two charges. Please fix this"
                " and confirm how long it will take."
            ),
            channel="email",
            region="US",
            metadata={"duplicate_charge": "true"},
        ),
        "rubric": {
            "priority": "high",
            "team": "billing",
            "must_tags": ["refund_check"],
            "must_not_escalate": True,
            "can_resolve": False,
            "required_phrases": ["sorry", "investigate", "24"],
        },
        "max_steps": 5,
    },
    "medium_policy_escalation": {
        "descriptor": TaskDescriptor(
            id="medium_policy_escalation",
            title="Suspicious Account Activity Escalation",
            difficulty="medium",
            goal=(
                "Identify suspicious behavior and route/escalate to Trust & Safety while"
                " avoiding premature resolution."
            ),
        ),
        "ticket": Ticket(
            id="TKT-2002",
            customer_tier="enterprise",
            subject="Multiple failed admin logins from unknown IPs",
            body=(
                "We noticed many admin login failures overnight from regions we do not"
                " operate in. We suspect account compromise."
            ),
            channel="web_form",
            region="DE",
            metadata={"security_incident": "suspected"},
        ),
        "rubric": {
            "priority": "urgent",
            "team": "trust_safety",
            "must_tags": ["account_takeover", "security_review"],
            "must_escalate": True,
            "can_resolve": False,
            "required_phrases": ["secure", "review", "timeline"],
        },
        "max_steps": 6,
    },
    "hard_multi_constraint_resolution": {
        "descriptor": TaskDescriptor(
            id="hard_multi_constraint_resolution",
            title="Cross-Team Incident with Compliance Constraints",
            difficulty="hard",
            goal=(
                "Handle a severe incident balancing urgency, compliance language,"
                " partial remediation, and correct escalation policy."
            ),
        ),
        "ticket": Ticket(
            id="TKT-3003",
            customer_tier="enterprise",
            subject="PII export mismatch and delayed incident response",
            body=(
                "Our compliance team found that one data export included extra user rows."
                " We need immediate mitigation steps and a written incident timeline."
            ),
            channel="email",
            region="IN",
            metadata={"compliance": "gdpr_like", "pii_exposure": "possible"},
        ),
        "rubric": {
            "priority": "urgent",
            "team": "trust_safety",
            "must_tags": ["compliance", "pii_review", "incident_timeline"],
            "must_escalate": True,
            "can_resolve": False,
            "required_phrases": ["mitigation", "timeline", "compliance", "review"],
        },
        "max_steps": 7,
    },
}


def list_task_descriptors() -> List[TaskDescriptor]:
    return [entry["descriptor"] for entry in TASKS.values()]


def get_task_bundle(task_id: str) -> Dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def grade_action(task_id: str, action: SupportAction) -> Tuple[float, Dict[str, float], str]:
    bundle = get_task_bundle(task_id)
    rubric = bundle["rubric"]
    details: Dict[str, float] = {}

    details["priority"] = 1.0 if action.priority == rubric["priority"] else 0.0
    details["team"] = 1.0 if action.team == rubric["team"] else 0.0

    required_tags = set(rubric.get("must_tags", []))
    action_tags = set(tag.lower().strip() for tag in action.tags)
    if not required_tags:
        details["tags"] = 1.0
    else:
        overlap = len(required_tags.intersection(action_tags))
        details["tags"] = overlap / max(1, len(required_tags))

    if rubric.get("must_escalate", False):
        details["escalate"] = 1.0 if action.escalate else 0.0
    elif rubric.get("must_not_escalate", False):
        details["escalate"] = 1.0 if not action.escalate else 0.0
    else:
        details["escalate"] = 1.0

    can_resolve = rubric.get("can_resolve", True)
    details["resolve"] = 1.0 if (action.resolve == can_resolve) else 0.0

    response = action.response_text.lower()
    phrases = rubric.get("required_phrases", [])
    if not phrases:
        details["response"] = 1.0
    else:
        matched = sum(1 for phrase in phrases if phrase in response)
        details["response"] = matched / len(phrases)

    # Weighted deterministic score -> 0..1
    weights = {
        "priority": 0.20,
        "team": 0.20,
        "tags": 0.20,
        "escalate": 0.15,
        "resolve": 0.10,
        "response": 0.15,
    }
    score = 0.0
    for k, w in weights.items():
        score += details.get(k, 0.0) * w
    score = max(0.0, min(1.0, score))

    if score >= 0.95:
        reason = "Excellent triage decision."
    elif score >= 0.75:
        reason = "Good triage with minor gaps."
    elif score >= 0.5:
        reason = "Partial progress; multiple fields still incorrect."
    else:
        reason = "Triage decision misses core requirements."

    return score, details, reason
