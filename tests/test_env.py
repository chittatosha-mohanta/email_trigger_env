import sys
import os
from pathlib import Path

# Add root directory to sys.path to resolve server and models
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from server.environment import EmailTriageEnvironment
from models import EmailTriageAction

def test_environment_lifecycle():
    """Test the basic lifecycle of the Email Triage environment."""
    env = EmailTriageEnvironment()
    
    # Test reset (Task 1: Easy)
    obs = env.reset(task_id=1)
    assert obs.task_id == 1
    assert obs.email_id == "e001"
    assert "Easy" in obs.feedback
    assert obs.done is False

    # Test step
    action = EmailTriageAction(
        action_type="triage",
        category="spam",
        priority=1,
        response_draft=""
    )
    res = env.step(action)
    
    # Verify progression
    assert res.reward > 0  # Should get some reward for correct spam categorization
    assert res.inbox_remaining < obs.inbox_remaining
    assert res.done is False

def test_environment_completion():
    """Test running a task to completion."""
    env = EmailTriageEnvironment()
    obs = env.reset(task_id=1)
    
    # Task 1 has 5 emails
    for _ in range(5):
        action = EmailTriageAction(
            action_type="triage",
            category="spam",
            priority=1,
            response_draft=""
        )
        res = env.step(action)
        if res.done:
            break
            
    assert res.done is True
    assert "Final score" in res.feedback

def test_invalid_task_id():
    """Test reset with an out-of-range task ID."""
    env = EmailTriageEnvironment()
    # Should clamp to task 1 or 3
    obs = env.reset(task_id=99)
    assert obs.task_id == 3 
