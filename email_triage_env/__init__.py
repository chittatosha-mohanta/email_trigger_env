"""Email Triage OpenEnv Environment.

A real-world RL environment where an AI agent triages emails:
categorize, prioritize, and draft responses.
"""

from .models import EmailTriageAction, EmailTriageObservation, EmailTriageState

__all__ = [
    "EmailTriageAction",
    "EmailTriageObservation",
    "EmailTriageState",
]
