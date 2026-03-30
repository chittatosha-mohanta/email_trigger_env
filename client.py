"""Client for the Email Triage OpenEnv environment.

Provides a typed Python client for interacting with the environment
via HTTP or directly.
"""

import requests
from typing import Optional
from dataclasses import asdict

from .models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageEnvClient:
    """HTTP client for the Email Triage environment.

    Usage:
        client = EmailTriageEnvClient(base_url="http://localhost:8000")
        obs = client.reset(task_id=1)
        while not obs.done:
            action = EmailTriageAction(category="work", priority=3, ...)
            obs = client.step(action)
        print(f"Final score: {obs.reward}")
    """

    def __init__(self, base_url: str = "http://localhost:8000", session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def reset(self, task_id: int = 1, seed: Optional[int] = None,
              episode_id: Optional[str] = None) -> EmailTriageObservation:
        """Reset the environment and start a new episode."""
        payload = {
            "task_id": task_id,
            "seed": seed,
            "episode_id": episode_id,
            "session_id": self.session_id,
        }
        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id", self.session_id)
        return self._parse_observation(data)

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Take a step in the environment."""
        payload = {
            "session_id": self.session_id,
            "action_type": action.action_type,
            "category": action.category,
            "priority": action.priority,
            "response_draft": action.response_draft,
        }
        resp = requests.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def state(self) -> EmailTriageState:
        """Get the current state of the environment."""
        payload = {"session_id": self.session_id}
        resp = requests.post(f"{self.base_url}/state", json=payload)
        resp.raise_for_status()
        return self._parse_state(resp.json())

    def health(self) -> dict:
        """Check if the server is healthy."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def _parse_observation(self, data: dict) -> EmailTriageObservation:
        """Parse server response into an EmailTriageObservation."""
        obs_data = data.get("observation", {})
        return EmailTriageObservation(
            done=data.get("done", False),
            reward=data.get("reward"),
            email_id=obs_data.get("email_id", ""),
            email_subject=obs_data.get("email_subject", ""),
            email_from=obs_data.get("email_from", ""),
            email_body=obs_data.get("email_body", ""),
            email_timestamp=obs_data.get("email_timestamp", ""),
            current_task=obs_data.get("current_task", ""),
            task_id=obs_data.get("task_id", 1),
            inbox_remaining=obs_data.get("inbox_remaining", 0),
            total_emails=obs_data.get("total_emails", 0),
            score_so_far=obs_data.get("score_so_far", 0.0),
            feedback=obs_data.get("feedback", ""),
        )

    def _parse_state(self, data: dict) -> EmailTriageState:
        """Parse server response into an EmailTriageState."""
        return EmailTriageState(
            episode_id=data.get("episode_id"),
            step_count=data.get("step_count", 0),
            task_id=data.get("task_id", 1),
            total_emails=data.get("total_emails", 0),
            processed_emails=data.get("processed_emails", 0),
            correct_categorizations=data.get("correct_categorizations", 0),
            correct_priorities=data.get("correct_priorities", 0),
            response_scores=data.get("response_scores", 0.0),
        )
