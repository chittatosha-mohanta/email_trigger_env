# 📧 Email Triage — OpenEnv Environment

An OpenEnv reinforcement learning environment that simulates **real-world email triage**. An AI agent must categorize incoming emails, assign priority levels, and draft appropriate responses — a task humans perform daily.

## 🎯 Motivation

Email triage is one of the most common knowledge-worker tasks: categorizing messages, deciding what's urgent, and crafting responses. This environment goes beyond simple classification by incorporating:

- **Email threads** — multi-message conversations requiring contextual reasoning
- **SLA/deadline awareness** — time-sensitive items with urgency decay
- **Phishing detection** — deceptive emails disguised as legitimate messages
- **Response generation** — drafting professional, actionable replies

## 📦 Environment Overview

| Property | Value |
|----------|-------|
| **Domain** | Email triage (categorize, prioritize, respond) |
| **Action Type** | Text-based (category, priority, response draft) |
| **Observation Type** | Structured email data with thread context |
| **Episode Length** | 5–15 steps (one step per email) |
| **Tasks** | 3 (Easy → Medium → Hard) |
| **Score Range** | 0.0 – 1.0 |
| **Novel Mechanics** | Email threads, SLA tracking, phishing detection |

## 🎮 Action Space

```python
@dataclass
class EmailTriageAction:
    action_type: str    # "triage", "categorize", "prioritize", "respond", "skip"
    category: str       # "spam", "newsletter", "personal", "work", "urgent"
    priority: int       # 1 (lowest) to 5 (highest)
    response_draft: str # Optional drafted response text
```

### Categories
| Category | Description |
|----------|-------------|
| `spam` | Unsolicited, scam, phishing, junk |
| `newsletter` | Subscriptions, digests, automated updates |
| `personal` | From friends/family, non-work |
| `work` | Professional, colleagues, projects |
| `urgent` | Time-sensitive, critical, immediate action needed |

### Priority Levels
| Level | Meaning | SLA Mapping |
|-------|---------|-------------|
| 1 | Can be ignored (spam, notifications) | No SLA or >72h |
| 2 | Low importance (newsletters, FYI) | 24-72h SLA |
| 3 | Normal importance (routine work) | 8-24h SLA |
| 4 | Important, handle today (deadlines) | 2-8h SLA |
| 5 | Critical, handle immediately | <2h SLA |

## 👀 Observation Space

```python
@dataclass
class EmailTriageObservation:
    done: bool                  # Episode finished?
    reward: Optional[float]     # Step/final reward
    email_id: str               # Current email ID
    email_subject: str          # Subject line
    email_from: str             # Sender address
    email_body: str             # Body (includes thread context in Task 3)
    email_timestamp: str        # ISO timestamp
    current_task: str           # Task description (includes SLA info)
    task_id: int                # Task number (1/2/3)
    inbox_remaining: int        # Emails left
    total_emails: int           # Total in episode
    score_so_far: float         # Running score 0.0-1.0
    feedback: str               # Grader feedback
```

### Novel: Thread Context in Task 3
For threaded emails, the `email_body` includes a `--- THREAD CONTEXT ---` section showing previous messages in the conversation. The agent must reason about the full thread to correctly categorize (e.g., a "PRODUCTION DOWN" email becomes "work" once resolved).

## 📋 Tasks

### Task 1: Easy — Single Email Categorization
- **Emails**: 5 clearly-labeled emails (one per category)
- **Grading**: Category accuracy only
- **Expected difficulty**: ~0.8-1.0 score for LLMs

### Task 2: Medium — Categorize & Prioritize
- **Emails**: 10 emails including ambiguous ones + phishing
- **Grading**: Category + priority accuracy
- **Includes**: A deceptive phishing email (Netflix impersonation)
- **Expected difficulty**: ~0.6-0.8 score

### Task 3: Hard — Full Triage with Threads & SLA
- **Emails**: 15 emails with conversation threads and SLA deadlines
- **Grading**: Category + priority + response quality + SLA awareness
- **Includes**:
  - 🧵 4 email threads (outage→update→resolved, security→patched, timeline→client, moving→change)
  - ⏱️ SLA deadlines (1h to 168h)
  - 🎣 Phishing email mixed in
  - ✍️ Response draft quality grading
- **Expected difficulty**: ~0.4-0.6 score — **genuinely challenges frontier models**

## 🏆 Reward Function

Multi-dimensional reward with **partial progress signals** (not just binary end-of-episode):

| Component | Description | Signal |
|-----------|-------------|--------|
| **Category** | Exact match | 1.0 correct, 0.0 wrong |
| **Priority** | Proximity match | 1.0 exact, 0.5 for ±1, 0.2 for ±2 |
| **Response** | Multi-signal grading | Hint keywords (30%), email keywords (20%), length (20%), tone (30%) |
| **SLA awareness** | Priority ↔ deadline alignment | 1.0 if priority ≥ SLA-implied minimum |
| **Efficiency** | Step penalty | -0.02 per step |
| **Skip penalty** | Missing important emails | -0.3 for skipping priority ≥ 4 |
| **Performance bonus** | High overall score | +0.05 if score > 0.7 |

Weights distribute **dynamically** based on active grading dimensions per task (e.g., Task 1 only grades category, so category gets 100% weight).

## 🚀 Setup & Usage

### Installation

```bash
cd Hackathon
pip install -e email_triage_env/
```

### Direct Python Usage

```python
from email_triage_env.server.environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction

env = EmailTriageEnvironment()
obs = env.reset(task_id=1)

while not obs.done:
    action = EmailTriageAction(
        category="work", priority=3,
        action_type="triage",
        response_draft="Thank you, noted."
    )
    obs = env.step(action)

print(f"Final score: {obs.reward}")
```

### HTTP Server

```bash
uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000 --reload
curl http://localhost:8000/health
```

### Docker

```bash
# From project root (Hackathon/)
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Baseline Inference

```bash
export OPENAI_API_KEY="your-key-here"
python email_triage_env/baseline.py --model gpt-4o-mini --episodes 3
```

## 📊 Baseline Scores

| Task | Difficulty | Model | Expected Mean | Notes |
|------|-----------|-------|---------------|-------|
| 1 | Easy | gpt-4o-mini | ~0.85 | Clear-cut categorization |
| 2 | Medium | gpt-4o-mini | ~0.65 | Phishing detection tested |
| 3 | Hard | gpt-4o-mini | ~0.50 | Thread reasoning + SLA + responses |

*Scores are approximate. Run `baseline.py` for exact reproducible results.*

## 📁 Project Structure

```
Hackathon/
├── Dockerfile                    # Container definition (project root)
├── .dockerignore
└── email_triage_env/
    ├── __init__.py               # Package exports
    ├── models.py                 # Action, Observation, State types
    ├── client.py                 # HTTP client for remote usage
    ├── baseline.py               # Baseline inference (OpenAI API)
    ├── openenv.yaml              # Environment manifest
    ├── pyproject.toml            # Package metadata
    ├── README.md                 # This file
    └── server/
        ├── __init__.py
        ├── environment.py        # Core simulation (35+ emails, threads, SLA)
        ├── app.py                # FastAPI + WebSocket server
        ├── requirements.txt
        └── Dockerfile            # Alternative Dockerfile
```

## 🔧 OpenEnv Spec Compliance

- ✅ Typed `Action`, `Observation`, `State` models (Pydantic-compatible dataclasses)
- ✅ `step(action)` → returns observation, reward, done, info
- ✅ `reset()` → returns initial observation with clean state
- ✅ `state()` → returns current episode metadata
- ✅ `openenv.yaml` with environment metadata and task definitions
- ✅ 3 tasks with programmatic graders (Easy → Medium → Hard, scores 0.0–1.0)
- ✅ Meaningful reward function with partial progress signals
- ✅ Baseline inference script with OpenAI API
- ✅ Working Dockerfile
- ✅ Deploys to Hugging Face Spaces (port 7860)

## 📄 License

MIT
