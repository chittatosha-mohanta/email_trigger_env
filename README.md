# Support Triage OpenEnv

Support Triage OpenEnv is a real-world environment that simulates customer support triage workflows.  
An agent must inspect incoming tickets, assign priority/team, decide escalation/resolution, and craft policy-compliant responses.

## Why this environment

Customer support triage is a high-value operational task in SaaS, fintech, and e-commerce teams. This environment is useful for:

- evaluating planning + policy-following behavior;
- measuring action quality under partial information;
- benchmarking reward shaping for practical agent workflows.

## Environment design

- **Domain**: customer support operations
- **Episode unit**: one ticket triage workflow
- **Stateful actions**: each step updates ticket and audit state
- **Reward**: dense partial-credit shaping across multiple criteria
- **Termination**:
  - successful completion of required triage workflow, or
  - maximum step budget reached

## Action and observation spaces

Typed via Pydantic models in `app/models.py`.

- **Observation (`SupportObservation`)**
  - `goal`: task objective text
  - `task_id`: current task identifier
  - `step_count`, `max_steps`
  - `inbox`: compact list of ticket summaries
  - `active_ticket`: full ticket currently being processed
  - `history`: action/reward trail
  - `last_error`: validation/runtime feedback

- **Action (`SupportAction`)**
  - `ticket_id`: ticket to operate on
  - `priority`: `low|medium|high|urgent`
  - `team`: `billing|tech|trust_safety|logistics|general`
  - `tags`: list of tags
  - `escalate`: bool
  - `resolve`: bool
  - `response_text`: drafted response
  - `note`: rationale/log note

- **Reward (`SupportReward`)**
  - `value`: float score for step
  - `components`: deterministic per-criterion breakdown
  - `penalty`: negative contribution from bad behavior
  - `done_bonus`: terminal bonus for high-quality completion

## Tasks and graders (easy → medium → hard)

Implemented in `app/tasks.py` with deterministic rubric graders (`0.0` to `1.0`):

1. `easy_priority_routing`
   - classify a simple billing issue and route correctly.
2. `medium_policy_escalation`
   - detect suspicious behavior, route to trust/safety, escalate, and include required tags.
3. `hard_multi_constraint_resolution`
   - apply nuanced decisions: urgency, escalation logic, partial refund language, and compliance wording.

Each grader uses deterministic checks with weighted scoring; no randomness in grading.

## Project structure

- `app/models.py`: typed observation/action/reward/state models
- `app/tasks.py`: task definitions and grader logic
- `app/environment.py`: `reset()`, `step()`, `state()` implementation
- `app/server.py`: FastAPI endpoints for environment interaction
- `openenv.yaml`: metadata + interface description
- `inference.py`: baseline script using OpenAI client
- `Dockerfile`: container build for local/HF Spaces

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run local API:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

## Baseline inference

Set required environment variables:

```bash
export OPENAI_API_KEY="..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_or_api_token"
```

Run:

```bash
python inference.py
```

The script evaluates all 3 tasks and prints per-task + average reproducible scores.

## Docker

```bash
docker build -t support-triage-openenv .
docker run --rm -p 7860:7860 support-triage-openenv
```

## Hugging Face Spaces

- Space SDK: `docker`
- Tag your Space with `openenv`
- Push this repo with the included `Dockerfile`

Health check endpoint:

- `GET /health`

OpenEnv endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

## Validation checklist

- `openenv validate` passes on `openenv.yaml`
- API returns valid typed payloads
- Docker build + run succeeds
- `inference.py` runs under 20 minutes on modest CPU
- all graders return scores in `0.0 .. 1.0`
