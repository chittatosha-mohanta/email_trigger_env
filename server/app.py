"""FastAPI application for the Email Triage environment.

Creates all OpenEnv endpoints: /ws, /reset, /step, /state, /health, /docs.
"""

import sys
import os
import json
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure parent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from server.environment import EmailTriageEnvironment


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage Environment",
    description="An OpenEnv RL environment for email triage — categorize, prioritize, and respond to emails.",
    version="1.0.0",
)

# Store environments per session
_sessions: dict[str, EmailTriageEnvironment] = {}


def _get_or_create_env(session_id: str) -> EmailTriageEnvironment:
    """Get or create an environment for the given session."""
    if session_id not in _sessions:
        _sessions[session_id] = EmailTriageEnvironment()
    return _sessions[session_id]


def _obs_to_dict(obs: EmailTriageObservation) -> dict:
    """Convert observation to JSON-serializable dict."""
    return {
        "done": obs.done,
        "reward": obs.reward,
        "observation": {
            "email_id": obs.email_id,
            "email_subject": obs.email_subject,
            "email_from": obs.email_from,
            "email_body": obs.email_body,
            "email_timestamp": obs.email_timestamp,
            "current_task": obs.current_task,
            "task_id": obs.task_id,
            "inbox_remaining": obs.inbox_remaining,
            "total_emails": obs.total_emails,
            "score_so_far": obs.score_so_far,
            "feedback": obs.feedback,
        },
    }


def _state_to_dict(state: EmailTriageState) -> dict:
    """Convert state to JSON-serializable dict."""
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "task_id": state.task_id,
        "total_emails": state.total_emails,
        "processed_emails": state.processed_emails,
        "correct_categorizations": state.correct_categorizations,
        "correct_priorities": state.correct_priorities,
        "response_scores": state.response_scores,
    }


# --- HTTP endpoints ---

class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: int = 1


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action_type: str = "triage"
    category: str = ""
    priority: int = 3
    response_draft: str = ""


class StateRequest(BaseModel):
    session_id: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "email_triage"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    request = request or ResetRequest()
    """Reset the environment and start a new episode."""
    session_id = request.session_id or str(uuid.uuid4())
    env = _get_or_create_env(session_id)
    obs = env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    result = _obs_to_dict(obs)
    result["session_id"] = session_id
    return JSONResponse(content=result)


@app.post("/step")
async def step(request: Optional[StepRequest] = None):
    request = request or StepRequest()
    """Take a step in the environment."""
    session_id = request.session_id or "default"
    env = _get_or_create_env(session_id)
    action = EmailTriageAction(
        action_type=request.action_type,
        category=request.category,
        priority=request.priority,
        response_draft=request.response_draft,
    )
    obs = env.step(action)
    result = _obs_to_dict(obs)
    result["session_id"] = session_id
    return JSONResponse(content=result)


@app.post("/state")
async def get_state(request: Optional[StateRequest] = None):
    request = request or StateRequest()
    """Get the current state of the environment."""
    session_id = request.session_id or "default"
    env = _get_or_create_env(session_id)
    return JSONResponse(content=_state_to_dict(env.state))


# --- WebSocket endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time interaction."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = _get_or_create_env(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            method = msg.get("method", "")

            if method == "reset":
                obs = env.reset(
                    seed=msg.get("seed"),
                    episode_id=msg.get("episode_id"),
                    task_id=msg.get("task_id", 1),
                )
                result = _obs_to_dict(obs)
                result["session_id"] = session_id
                await websocket.send_json(result)

            elif method == "step":
                action = EmailTriageAction(
                    action_type=msg.get("action_type", "triage"),
                    category=msg.get("category", ""),
                    priority=msg.get("priority", 3),
                    response_draft=msg.get("response_draft", ""),
                )
                obs = env.step(action)
                result = _obs_to_dict(obs)
                result["session_id"] = session_id
                await websocket.send_json(result)

            elif method == "state":
                await websocket.send_json(_state_to_dict(env.state))

            else:
                await websocket.send_json({"error": f"Unknown method: {method}"})

    except WebSocketDisconnect:
        if session_id in _sessions:
            del _sessions[session_id]

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
