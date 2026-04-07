---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Email Triage OpenEnv

An OpenEnv environment for email triage. The agent must categorize, prioritize, and draft responses to emails in a simulated inbox.

## Features
- 30+ synthetic emails across 5 categories (spam, newsletter, personal, work, urgent)
- Email threads with conversation context
- SLA/deadline tracking with urgency awareness
- 3 difficulty levels (Easy, Medium, Hard)

## Action Space
Typed via `models.EmailTriageAction`:
- `action_type`: "triage"
- `category`: `spam|newsletter|personal|work|urgent`
- `priority`: `1|2|3|4|5`
- `response_draft`: string

## Tasks
1. **Easy**: Categorize 5 clearly-labeled emails.
2. **Medium**: Categorize and prioritize 10 emails (some ambiguous).
3. **Hard**: Full triage of 15 emails including threads and SLAs.

## Local Setup
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Inference
```bash
python inference.py
```
