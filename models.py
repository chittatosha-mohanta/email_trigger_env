"""Typed Pydantic models for the Email Triage environment.

Defines Action, Observation, and State contracts following the OpenEnv spec.
Uses dataclasses since openenv-core provides base model support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# --- Action: What the agent can do ---

@dataclass
class EmailTriageAction:
    """Agent's triage action for a single email.

    Attributes:
        action_type: One of 'categorize', 'prioritize', 'respond', 'skip', 'triage'.
                     'triage' combines all three operations in one step.
        category: Email category — one of:
                  'spam', 'newsletter', 'personal', 'work', 'urgent'
        priority: Urgency level from 1 (lowest) to 5 (highest)
        response_draft: Optional drafted response text.
        metadata: Extra key-value pairs (optional).
    """
    action_type: str = "triage"
    category: str = ""
    priority: int = 3
    response_draft: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Observation: What the agent sees ---

@dataclass
class EmailTriageObservation:
    """What the agent observes after each step.

    Attributes:
        done: Whether the episode is finished.
        reward: Reward for the current step (None on reset).
        email_id: Unique identifier of the current email.
        email_subject: Subject line of the current email.
        email_from: Sender address.
        email_body: Full email body text.
        email_timestamp: When the email was received (ISO format).
        current_task: Description of the current task requirements.
        task_id: Task number (1=easy, 2=medium, 3=hard).
        inbox_remaining: Number of emails left in the inbox.
        total_emails: Total emails in this episode.
        score_so_far: Running score (0.0-1.0) for the episode.
        feedback: Textual feedback on the agent's last action.
        metadata: Extra key-value pairs.
    """
    done: bool = False
    reward: Optional[float] = None
    email_id: str = ""
    email_subject: str = ""
    email_from: str = ""
    email_body: str = ""
    email_timestamp: str = ""
    current_task: str = ""
    task_id: int = 1
    inbox_remaining: int = 0
    total_emails: int = 0
    score_so_far: float = 0.0
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- State: Episode metadata ---

@dataclass
class EmailTriageState:
    """Episode metadata for the Email Triage environment.

    Attributes:
        episode_id: Unique episode identifier.
        step_count: Number of steps taken so far.
        task_id: Which task is being run (1, 2, or 3).
        total_emails: Total emails in the inbox.
        processed_emails: How many emails have been processed.
        correct_categorizations: Number of correct category assignments.
        correct_priorities: Number of correct priority assignments.
        response_scores: Sum of response quality scores.
    """
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: int = 1
    total_emails: int = 0
    processed_emails: int = 0
    correct_categorizations: int = 0
    correct_priorities: int = 0
    response_scores: float = 0.0
