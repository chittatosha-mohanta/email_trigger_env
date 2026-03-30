#!/usr/bin/env python3
"""Baseline inference script for the Email Triage OpenEnv environment.

Uses the OpenAI API to run a language model against all 3 tasks.
Produces reproducible baseline scores.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python baseline.py

    # Or specify model and episodes:
    python baseline.py --model gpt-4o-mini --episodes 3
"""

import os
import sys
import json
import argparse
import statistics
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailTriageAction, EmailTriageObservation
from server.environment import EmailTriageEnvironment

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


def build_system_prompt(task_description: str) -> str:
    """Build the system prompt for the LLM agent."""
    return f"""You are an AI email triage assistant. Your task: {task_description}

For each email, you must respond with a JSON object containing:
- "action_type": always "triage"
- "category": one of "spam", "newsletter", "personal", "work", "urgent"
- "priority": integer from 1 (lowest) to 5 (highest)
- "response_draft": a brief, professional response (if the task requires it, otherwise empty string)

Category guidelines:
- spam: Unsolicited, scam, phishing, or junk emails. Watch for: misspelled domains (e.g. netf1ix), urgency tactics, requests for personal data.
- newsletter: Subscriptions, digests, automated updates from legitimate sources.
- personal: From friends/family, non-work related.
- work: Professional, from colleagues, about projects/meetings. Includes follow-ups and resolved incidents.
- urgent: Time-sensitive, ACTIVE critical issues requiring immediate action. Note: once an incident is RESOLVED, it's "work" not "urgent".

Priority guidelines:
- 1: Can be ignored or handled whenever (spam, shipping notifications)
- 2: Low importance, read when convenient (newsletters, FYI emails, read-later items)
- 3: Normal importance, handle during work hours (routine work, social plans)
- 4: Important, handle today (deadlines, reviews, onboarding, follow-ups requiring action)
- 5: Critical, handle immediately (active production issues, security threats, urgent deadlines)

IMPORTANT RULES:
1. If THREAD CONTEXT is provided, read the full conversation history before making your decision. The category and priority may change as a thread evolves (e.g. an outage that gets resolved).
2. If SLA information is mentioned, ensure your priority reflects the urgency of the deadline.
3. Watch for phishing: check sender domains carefully, look for urgency tactics, suspicious links, or requests for sensitive information.
4. Response drafts should be professional, acknowledge the email's content, and include relevant action items.

Respond ONLY with the JSON object, no other text."""


def build_email_prompt(obs: EmailTriageObservation) -> str:
    """Build the user prompt showing the current email."""
    return f"""Email #{obs.total_emails - obs.inbox_remaining + 1} of {obs.total_emails}

From: {obs.email_from}
Subject: {obs.email_subject}
Date: {obs.email_timestamp}

{obs.email_body}

---
Remaining emails: {obs.inbox_remaining}
Current score: {obs.score_so_far:.3f}
Feedback: {obs.feedback}"""


def parse_llm_response(response_text: str) -> EmailTriageAction:
    """Parse the LLM's JSON response into an EmailTriageAction."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        return EmailTriageAction(
            action_type=data.get("action_type", "triage"),
            category=data.get("category", "work"),
            priority=int(data.get("priority", 3)),
            response_draft=data.get("response_draft", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  ⚠️ Failed to parse LLM response: {e}")
        print(f"  Raw response: {response_text[:200]}")
        # Return a default action
        return EmailTriageAction(
            action_type="triage",
            category="work",
            priority=3,
            response_draft="",
        )


def run_episode(env: EmailTriageEnvironment, client: OpenAI, model: str,
                task_id: int, seed: int) -> float:
    """Run a single episode and return the final score."""
    obs = env.reset(seed=seed, task_id=task_id)
    system_prompt = build_system_prompt(obs.current_task)

    step = 0
    while not obs.done:
        step += 1
        email_prompt = build_email_prompt(obs)

        # Call the LLM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": email_prompt},
            ],
            temperature=0.1,  # Low temperature for reproducibility
            max_tokens=500,
        )

        response_text = response.choices[0].message.content or ""
        action = parse_llm_response(response_text)

        print(f"    Step {step}: category={action.category}, priority={action.priority}")
        obs = env.step(action)

    print(f"    → Episode score: {obs.reward:.4f}")
    return obs.reward


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference on Email Triage environment")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # Check API configuration
    api_key = os.environ.get("HF_TOKEN")
    base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    if not api_key or not base_url or not model_name:
        print("ERROR: HF_TOKEN, API_BASE_URL, and MODEL_NAME environment variables must be set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    env = EmailTriageEnvironment()

    print("=" * 60)
    print(f"Email Triage Baseline — Model: {model_name}")
    print(f"Episodes per task: {args.episodes}")
    print("=" * 60)

    results: Dict[int, List[float]] = {}

    for task_id in [1, 2, 3]:
        print(f"\n📋 Task {task_id}:")
        task_scores = []

        for ep in range(args.episodes):
            seed = args.base_seed + task_id * 100 + ep
            print(f"  Episode {ep + 1}/{args.episodes} (seed={seed}):")
            score = run_episode(env, client, model_name, task_id, seed)
            task_scores.append(score)

        results[task_id] = task_scores

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BASELINE RESULTS")
    print("=" * 60)
    print(f"{'Task':<8} {'Difficulty':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}
    all_scores = []

    for task_id, scores in results.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        mn = min(scores)
        mx = max(scores)
        all_scores.extend(scores)
        print(f"Task {task_id:<4} {difficulties[task_id]:<12} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    print("-" * 60)
    overall_mean = statistics.mean(all_scores)
    overall_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
    print(f"{'Overall':<20} {overall_mean:>8.4f} {overall_std:>8.4f}")
    print("=" * 60)
    print(f"\nModel: {model_name} | Episodes: {args.episodes} | Seed: {args.base_seed}")


if __name__ == "__main__":
    main()
