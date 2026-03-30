"""Email Triage Environment — Core simulation logic.

Implements reset(), step(), and state for the OpenEnv 3-method interface.
Generates synthetic but realistic emails and grades agent triage actions.

Features:
- 30+ synthetic emails across 5 categories (spam, newsletter, personal, work, urgent)
- Email threads with conversation context (agent must reason about full threads)
- SLA/deadline tracking with urgency decay (novel mechanic)
- Multi-dimensional grading: category, priority, response quality, SLA awareness
- 3 tasks: Easy (categorize), Medium (categorize + prioritize), Hard (full triage + threads + SLA)
"""

import random
import uuid
import math
from typing import List, Dict, Optional, Set


from ..models import EmailTriageAction, EmailTriageObservation, EmailTriageState


# ---------------------------------------------------------------------------
# Synthetic email data (realistic workplace / personal emails)
# ---------------------------------------------------------------------------

EMAILS_DB: List[Dict] = [
    # --- SPAM ---
    {
        "id": "e001", "subject": "You've WON $1,000,000!!!", "from": "winner@prize-lottery.xyz",
        "body": "Congratulations! You have been selected as the winner of our international lottery. Click here to claim your prize. Send us your bank details immediately.",
        "timestamp": "2025-03-15T08:23:00Z", "category": "spam", "priority": 1,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["scam", "lottery", "prize"],
    },
    {
        "id": "e002", "subject": "Discount Viagra - 90% OFF!", "from": "deals@pharma-best.ru",
        "body": "Limited time offer! Get the best deals on pharmaceutical products. No prescription needed. Order now and save big!",
        "timestamp": "2025-03-15T09:01:00Z", "category": "spam", "priority": 1,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["pharma", "discount", "spam"],
    },
    {
        "id": "e003", "subject": "Your account has been compromised!", "from": "security@bankk-alert.com",
        "body": "Dear customer, we detected unusual activity. Please verify your identity by clicking this link and entering your password and SSN immediately.",
        "timestamp": "2025-03-15T09:30:00Z", "category": "spam", "priority": 1,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["phishing", "compromised", "verify"],
    },
    {
        "id": "e004", "subject": "Make $5000/week working from home", "from": "opportunity@quickcash.biz",
        "body": "No experience needed! Start earning thousands weekly with our proven system. Just send $99 to get started.",
        "timestamp": "2025-03-15T10:00:00Z", "category": "spam", "priority": 1,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["work from home", "earn", "scam"],
    },

    # --- NEWSLETTERS ---
    {
        "id": "e010", "subject": "This Week in AI - March Newsletter", "from": "newsletter@ai-weekly.com",
        "body": "Top stories this week: New GPT-5 benchmarks released, OpenAI announces enterprise pricing, Meta open-sources new model. Read more in our curated digest.",
        "timestamp": "2025-03-15T07:00:00Z", "category": "newsletter", "priority": 2,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["AI", "newsletter", "digest"],
    },
    {
        "id": "e011", "subject": "Your Monthly Python Digest", "from": "digest@python.org",
        "body": "Python 3.13 beta is out! New features include improved error messages, a JIT compiler preview, and enhanced typing support. Plus: top community packages this month.",
        "timestamp": "2025-03-14T18:00:00Z", "category": "newsletter", "priority": 2,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["Python", "newsletter", "release"],
    },
    {
        "id": "e012", "subject": "TechCrunch Daily - Startup Funding Round-up", "from": "daily@techcrunch.com",
        "body": "Today's top funding rounds: Series B for AI startup ($50M), healthtech raises $30M seed, new fintech unicorn emerges in Southeast Asia.",
        "timestamp": "2025-03-15T06:30:00Z", "category": "newsletter", "priority": 2,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["startup", "funding", "news"],
    },
    {
        "id": "e013", "subject": "Dev.to Weekly: Best Articles", "from": "weekly@dev.to",
        "body": "Trending this week: 'Why I switched from React to Svelte', 'Building microservices with Go', 'The future of WebAssembly'. Plus community highlights.",
        "timestamp": "2025-03-14T20:00:00Z", "category": "newsletter", "priority": 2,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["dev", "articles", "trending"],
    },

    # --- PERSONAL ---
    {
        "id": "e020", "subject": "Dinner this Saturday?", "from": "sarah.chen@gmail.com",
        "body": "Hey! It's been ages since we caught up. Are you free this Saturday evening? I was thinking we could try that new Italian place downtown. Let me know!",
        "timestamp": "2025-03-15T12:30:00Z", "category": "personal", "priority": 3,
        "expected_response": "confirm or suggest alternative plans",
        "thread_id": None, "sla_hours": 48,
        "keywords": ["dinner", "Saturday", "plans"],
    },
    {
        "id": "e021", "subject": "Happy Birthday! 🎂", "from": "mom@family.com",
        "body": "Happy birthday sweetheart! I hope you have a wonderful day. Dad and I are sending your gift — it should arrive by Thursday. We love you! Call us when you can.",
        "timestamp": "2025-03-15T00:01:00Z", "category": "personal", "priority": 3,
        "expected_response": "thank and acknowledge",
        "thread_id": None, "sla_hours": 24,
        "keywords": ["birthday", "thank", "gift"],
    },
    {
        "id": "e022", "subject": "Photos from last weekend", "from": "jake.miller@outlook.com",
        "body": "Hey, I finally got around to editing the photos from the hiking trip. Here's the Google Drive link. Let me know which ones you want me to print!",
        "timestamp": "2025-03-14T22:15:00Z", "category": "personal", "priority": 2,
        "expected_response": "thank and review photos",
        "thread_id": None, "sla_hours": 72,
        "keywords": ["photos", "hiking", "print"],
    },
    {
        "id": "e023", "subject": "Re: Moving help needed", "from": "roommate_alex@gmail.com",
        "body": "Thanks for offering to help! I'm moving next Sunday. Can you bring your car around 10am? I'll order pizza for everyone helping. Let me know if that works.",
        "timestamp": "2025-03-15T11:00:00Z", "category": "personal", "priority": 3,
        "expected_response": "confirm availability and time",
        "thread_id": "thread_moving", "sla_hours": 48,
        "keywords": ["moving", "Sunday", "confirm", "help"],
    },

    # --- WORK ---
    {
        "id": "e030", "subject": "Q1 Report - Review needed by Friday", "from": "manager@company.com",
        "body": "Hi team, please review the attached Q1 report and add your section's data by end of day Friday. The board meeting is next Monday. Let me know if you have questions.",
        "timestamp": "2025-03-15T09:00:00Z", "category": "work", "priority": 4,
        "expected_response": "acknowledge, confirm will review by deadline",
        "thread_id": None, "sla_hours": 24,
        "keywords": ["report", "review", "Friday", "deadline", "board"],
    },
    {
        "id": "e031", "subject": "Sprint Planning - Updated backlog", "from": "scrum-master@company.com",
        "body": "Team, I've updated the sprint backlog with new stories. Please review and add estimates before tomorrow's planning meeting at 10am. Link to Jira board attached.",
        "timestamp": "2025-03-15T08:30:00Z", "category": "work", "priority": 3,
        "expected_response": "acknowledge and confirm will review",
        "thread_id": None, "sla_hours": 16,
        "keywords": ["sprint", "backlog", "estimates", "planning"],
    },
    {
        "id": "e032", "subject": "Code Review Request: PR #1247", "from": "dev-team@company.com",
        "body": "Alex has submitted PR #1247 for the authentication refactor. Please review when you get a chance. No rush — target merge by end of sprint.",
        "timestamp": "2025-03-15T10:45:00Z", "category": "work", "priority": 3,
        "expected_response": "acknowledge and plan review",
        "thread_id": None, "sla_hours": 48,
        "keywords": ["code review", "PR", "authentication", "refactor"],
    },
    {
        "id": "e033", "subject": "Team lunch tomorrow - RSVP", "from": "admin@company.com",
        "body": "Hi everyone, we're organizing a team lunch tomorrow at 12:30pm at Sakura restaurant. Please reply with any dietary restrictions. Hope to see you there!",
        "timestamp": "2025-03-15T11:30:00Z", "category": "work", "priority": 2,
        "expected_response": "RSVP with attendance and dietary needs",
        "thread_id": None, "sla_hours": 12,
        "keywords": ["lunch", "RSVP", "dietary", "restaurant"],
    },
    {
        "id": "e034", "subject": "New hire onboarding - your mentee starts Monday", "from": "hr@company.com",
        "body": "Just a reminder that your new mentee, Priya Sharma, starts on Monday. Please prepare the onboarding checklist and schedule a 1:1 for her first day. IT has set up her accounts.",
        "timestamp": "2025-03-14T16:00:00Z", "category": "work", "priority": 4,
        "expected_response": "acknowledge and confirm preparation plan",
        "thread_id": None, "sla_hours": 48,
        "keywords": ["onboarding", "mentee", "Monday", "prepare", "checklist"],
    },
    {
        "id": "e035", "subject": "Expense report approval needed", "from": "finance@company.com",
        "body": "Your expense report for the DevConf trip ($1,247.50) has been submitted. Please verify the amounts and approve in the portal by March 20th.",
        "timestamp": "2025-03-15T14:00:00Z", "category": "work", "priority": 3,
        "expected_response": "acknowledge and verify",
        "thread_id": None, "sla_hours": 120,
        "keywords": ["expense", "approve", "verify", "portal"],
    },

    # --- URGENT ---
    {
        "id": "e040", "subject": "🚨 PRODUCTION DOWN - API 500 errors", "from": "alerts@company.com",
        "body": "CRITICAL: Production API is returning 500 errors. Error rate spiked to 45% at 14:23 UTC. Database connection pool exhausted. Need immediate investigation. PagerDuty incident #4521 created.",
        "timestamp": "2025-03-15T14:25:00Z", "category": "urgent", "priority": 5,
        "expected_response": "acknowledge immediately, join incident response",
        "thread_id": "thread_outage", "sla_hours": 1,
        "keywords": ["production", "down", "500", "critical", "incident", "immediate"],
    },
    {
        "id": "e041", "subject": "URGENT: Client demo in 2 hours - slides broken", "from": "sales-lead@company.com",
        "body": "The demo slides for Acme Corp are showing old data. The client meeting is at 4pm today. Can you update the product metrics on slides 8-12 ASAP? This is a $2M deal.",
        "timestamp": "2025-03-15T14:00:00Z", "category": "urgent", "priority": 5,
        "expected_response": "acknowledge urgency, update slides immediately",
        "thread_id": None, "sla_hours": 2,
        "keywords": ["urgent", "demo", "client", "ASAP", "slides", "update"],
    },
    {
        "id": "e042", "subject": "Security vulnerability found in auth module", "from": "security@company.com",
        "body": "A critical SQL injection vulnerability has been discovered in the login endpoint (CVE-2025-XXXX). All user sessions should be invalidated. Patch ready for review in PR #1250. Deploy ASAP.",
        "timestamp": "2025-03-15T13:15:00Z", "category": "urgent", "priority": 5,
        "expected_response": "acknowledge, review and deploy patch immediately",
        "thread_id": "thread_security", "sla_hours": 4,
        "keywords": ["security", "vulnerability", "SQL injection", "patch", "deploy", "critical"],
    },
    {
        "id": "e043", "subject": "Flight cancelled - need rebooking NOW", "from": "travel@company.com",
        "body": "Your flight AA1234 to the NYC client meeting tomorrow has been cancelled due to weather. Please rebook ASAP — the meeting cannot be rescheduled. Alternative flights available at 6am and 8pm.",
        "timestamp": "2025-03-15T15:30:00Z", "category": "urgent", "priority": 5,
        "expected_response": "rebook immediately, notify meeting attendees",
        "thread_id": None, "sla_hours": 3,
        "keywords": ["flight", "cancelled", "rebook", "ASAP", "meeting"],
    },

    # --- AMBIGUOUS / TRICKY ---
    {
        "id": "e050", "subject": "Re: Project timeline update", "from": "contractor@freelance.io",
        "body": "Hi, just wanted to flag that the deliverables for phase 2 might slip by a week. The design reviews took longer than expected. Should we adjust the client timeline or add resources?",
        "timestamp": "2025-03-15T11:45:00Z", "category": "work", "priority": 4,
        "expected_response": "discuss options, make decision on timeline",
        "thread_id": "thread_timeline", "sla_hours": 8,
        "keywords": ["timeline", "slip", "deliverables", "adjust", "resources"],
    },
    {
        "id": "e051", "subject": "Your order has shipped!", "from": "noreply@amazon.com",
        "body": "Your order #114-7732891 has shipped and will arrive by March 18. Track your package with the link below. Thank you for shopping with us!",
        "timestamp": "2025-03-15T10:20:00Z", "category": "personal", "priority": 1,
        "expected_response": "", "thread_id": None,
        "sla_hours": None, "keywords": ["order", "shipped", "tracking"],
    },
    {
        "id": "e052", "subject": "Invitation: Board Strategy Meeting", "from": "ceo@company.com",
        "body": "You're invited to present the technical roadmap at next month's board meeting April 15. Please prepare a 20-min presentation covering Q1 achievements and Q2 plans. This is high-visibility.",
        "timestamp": "2025-03-15T09:45:00Z", "category": "work", "priority": 5,
        "expected_response": "acknowledge importance, confirm and start preparation",
        "thread_id": None, "sla_hours": 48,
        "keywords": ["board", "presentation", "roadmap", "CEO", "high-visibility"],
    },
    {
        "id": "e053", "subject": "Hey, quick question about the API", "from": "intern@company.com",
        "body": "Hey, I'm trying to use the users endpoint but getting a 403. I think I need different permissions? Also, is there documentation for the v2 API? Thanks!",
        "timestamp": "2025-03-15T13:00:00Z", "category": "work", "priority": 3,
        "expected_response": "help with permissions and point to docs",
        "thread_id": None, "sla_hours": 24,
        "keywords": ["API", "permissions", "403", "documentation", "help"],
    },
    {
        "id": "e054", "subject": "Conference speaking opportunity", "from": "events@pycon.org",
        "body": "We'd love to have you speak at PyCon 2025! Based on your blog posts about ML pipelines, we think you'd be a great fit. CFP deadline is April 1. Interested?",
        "timestamp": "2025-03-14T14:00:00Z", "category": "work", "priority": 3,
        "expected_response": "express interest, ask for details",
        "thread_id": None, "sla_hours": 168,
        "keywords": ["conference", "speaking", "PyCon", "interest", "deadline"],
    },
    {
        "id": "e055", "subject": "Subscription renewal reminder", "from": "billing@jetbrains.com",
        "body": "Your PyCharm Professional subscription expires on March 31. Renew now to keep your settings and avoid interruption. Early renewal discount: 20% off.",
        "timestamp": "2025-03-14T12:00:00Z", "category": "work", "priority": 2,
        "expected_response": "decide on renewal",
        "thread_id": None, "sla_hours": 384,
        "keywords": ["subscription", "renewal", "expires", "discount"],
    },

    # --- EMAIL THREADS (multi-message context) ---
    {
        "id": "e060", "subject": "Re: 🚨 PRODUCTION DOWN - Update", "from": "devops@company.com",
        "body": "Update: We've identified the issue — a misconfigured connection pool limit in the new deployment. Rolling back now. ETA for fix: 30 minutes. Keep monitoring.",
        "timestamp": "2025-03-15T14:45:00Z", "category": "urgent", "priority": 5,
        "expected_response": "acknowledge update, confirm monitoring, ask if help needed",
        "thread_id": "thread_outage", "sla_hours": 1,
        "keywords": ["update", "rollback", "fix", "monitoring", "production"],
    },
    {
        "id": "e061", "subject": "Re: 🚨 PRODUCTION DOWN - Resolved", "from": "devops@company.com",
        "body": "All clear! Production is back to normal. Root cause: connection pool was set to 10 instead of 100 in the Helm chart. Post-mortem scheduled for tomorrow at 2pm. Please attend.",
        "timestamp": "2025-03-15T15:15:00Z", "category": "work", "priority": 4,
        "expected_response": "acknowledge resolution, confirm will attend post-mortem",
        "thread_id": "thread_outage", "sla_hours": 24,
        "keywords": ["resolved", "post-mortem", "root cause", "attend"],
    },
    {
        "id": "e062", "subject": "Re: Security vulnerability - Patch deployed", "from": "security@company.com",
        "body": "The security patch (PR #1250) has been deployed to production. All user sessions have been invalidated. Please verify the fix by testing login on staging. Report any issues immediately.",
        "timestamp": "2025-03-15T16:00:00Z", "category": "work", "priority": 4,
        "expected_response": "confirm will test on staging, report results",
        "thread_id": "thread_security", "sla_hours": 4,
        "keywords": ["patch", "deployed", "verify", "testing", "staging"],
    },
    {
        "id": "e063", "subject": "Re: Project timeline - Client notified", "from": "pm@company.com",
        "body": "I've notified the client about the potential 1-week delay. They're concerned but understanding. They want a revised timeline by Wednesday. Can you provide updated estimates for your tasks?",
        "timestamp": "2025-03-15T14:30:00Z", "category": "work", "priority": 4,
        "expected_response": "provide revised estimates by Wednesday, acknowledge client concern",
        "thread_id": "thread_timeline", "sla_hours": 24,
        "keywords": ["timeline", "revised", "estimates", "Wednesday", "client"],
    },
    {
        "id": "e064", "subject": "Re: Moving help needed - Change of plans", "from": "roommate_alex@gmail.com",
        "body": "Hey, change of plans — the movers can only come on Saturday now, not Sunday. Can you still help? Same time, 10am. Also, my friend Mike will have a truck so we won't need your car after all.",
        "timestamp": "2025-03-15T16:30:00Z", "category": "personal", "priority": 3,
        "expected_response": "confirm availability for Saturday, acknowledge car not needed",
        "thread_id": "thread_moving", "sla_hours": 24,
        "keywords": ["Saturday", "change", "plans", "confirm", "availability"],
    },

    # --- DECEPTIVE / HARD-TO-CLASSIFY ---
    {
        "id": "e070", "subject": "Important: Your Netflix payment failed", "from": "billing@netf1ix-support.com",
        "body": "We were unable to process your payment for Netflix Premium. Your account will be suspended in 24 hours. Please update your payment information by clicking the button below.",
        "timestamp": "2025-03-15T08:00:00Z", "category": "spam", "priority": 1,
        "expected_response": "",
        "thread_id": None, "sla_hours": None,
        "keywords": ["phishing", "Netflix", "payment", "suspicious"],
    },
    {
        "id": "e071", "subject": "Quick sync about architecture decision?", "from": "tech-lead@company.com",
        "body": "Hey, I've been thinking about our discussion on microservices vs monolith for the new platform. I'm leaning towards starting monolithic and extracting services later. Can we grab 15 min today to align? The VP is asking for our recommendation by EOD.",
        "timestamp": "2025-03-15T10:30:00Z", "category": "urgent", "priority": 4,
        "expected_response": "schedule meeting, share initial thoughts on architecture",
        "thread_id": None, "sla_hours": 4,
        "keywords": ["architecture", "sync", "decision", "EOD", "VP", "recommendation"],
    },
    {
        "id": "e072", "subject": "Fw: Interesting AI paper - thoughts?", "from": "colleague@company.com",
        "body": "Saw this paper on retrieval-augmented generation for enterprise search. Their approach to chunking strategies is similar to what we discussed for our knowledge base project. Might be relevant for the Q2 roadmap. Worth a read when you have time.",
        "timestamp": "2025-03-15T09:15:00Z", "category": "work", "priority": 2,
        "expected_response": "acknowledge, plan to review paper",
        "thread_id": None, "sla_hours": 168,
        "keywords": ["paper", "RAG", "research", "knowledge base", "roadmap"],
    },
]


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    1: {
        "name": "Easy — Single Email Categorization",
        "description": "Categorize 5 clearly-labeled emails into the correct category (spam, newsletter, personal, work, urgent). Priority and response not required.",
        "email_ids": ["e001", "e010", "e020", "e030", "e040"],
        "grade_category": True,
        "grade_priority": False,
        "grade_response": False,
        "grade_sla": False,
        "include_threads": False,
    },
    2: {
        "name": "Medium — Categorize & Prioritize",
        "description": "Categorize AND assign priority (1-5) for 10 emails, including some ambiguous ones. Consider sender context and content signals. Response drafts not required.",
        "email_ids": ["e002", "e011", "e021", "e031", "e041", "e050", "e051", "e052", "e033", "e070"],
        "grade_category": True,
        "grade_priority": True,
        "grade_response": False,
        "grade_sla": False,
        "include_threads": False,
    },
    3: {
        "name": "Hard — Full Triage with Threads & SLA",
        "description": (
            "Full triage: categorize, prioritize, AND draft appropriate responses for 15 emails. "
            "This includes EMAIL THREADS — multiple messages in the same conversation that require "
            "contextual understanding. Some emails have SLA deadlines; your priority assignment should "
            "reflect urgency. A deceptive phishing email is mixed in. You must reason about conversation "
            "history, detect deception, and handle time-sensitive items appropriately."
        ),
        "email_ids": [
            # Outage thread (3 emails: incident → update → resolved)
            "e040", "e060", "e061",
            # Security thread (2 emails: vuln → patch deployed)
            "e042", "e062",
            # Timeline thread (2 emails: delay → client notified)
            "e050", "e063",
            # Moving thread (2 emails: help → change of plans)
            "e023", "e064",
            # Deceptive phishing (looks like real Netflix)
            "e070",
            # Tricky work emails
            "e071", "e072", "e052",
            # Regular fillers
            "e041", "e054",
        ],
        "grade_category": True,
        "grade_priority": True,
        "grade_response": True,
        "grade_sla": True,
        "include_threads": True,
    },
}


# ---------------------------------------------------------------------------
# Thread context builder
# ---------------------------------------------------------------------------

def _build_thread_context(email: Dict, all_emails: List[Dict], current_idx: int) -> str:
    """Build conversation thread context for emails that are part of a thread.

    For threaded emails, includes previous messages in the same thread
    that the agent has already seen. This tests the agent's ability to
    reason about conversation history.
    """
    thread_id = email.get("thread_id")
    if not thread_id:
        return ""

    # Find previous emails in same thread that appeared before current_idx
    prev_in_thread = []
    for i in range(current_idx):
        e = all_emails[i]
        if e.get("thread_id") == thread_id:
            prev_in_thread.append(e)

    if not prev_in_thread:
        return ""

    context_parts = ["\n--- THREAD CONTEXT (previous messages in this conversation) ---"]
    for prev in prev_in_thread:
        context_parts.append(
            f"\nFrom: {prev['from']}\n"
            f"Subject: {prev['subject']}\n"
            f"Date: {prev['timestamp']}\n"
            f"Body: {prev['body']}"
        )
    context_parts.append("\n--- END THREAD CONTEXT ---\n")

    return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def _grade_category(predicted: str, expected: str) -> float:
    """Score category prediction. Returns 1.0 for exact match, 0.0 otherwise."""
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def _grade_priority(predicted: int, expected: int) -> float:
    """Score priority prediction. Full marks for exact, partial for close."""
    diff = abs(predicted - expected)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    elif diff == 2:
        return 0.2
    return 0.0


def _grade_response(draft: str, expected_hint: str, keywords: List[str]) -> float:
    """Score response quality using multi-signal grading.

    Signals:
    - Keyword relevance (from email content keywords)
    - Action word matching (from expected response hint)
    - Length appropriateness
    - Professionalism/tone
    """
    if not expected_hint:
        # No response expected — giving one is neutral
        return 1.0 if not draft else 0.8

    if not draft or len(draft.strip()) < 10:
        return 0.0

    draft_lower = draft.lower()
    scores = []

    # 1. Action word matching from expected hint (30%)
    hint_words = set(expected_hint.lower().split())
    meaningful_words = {w for w in hint_words if len(w) > 3}
    if not meaningful_words:
        meaningful_words = hint_words
    matches = sum(1 for w in meaningful_words if w in draft_lower)
    hint_score = min(matches / max(len(meaningful_words), 1), 1.0)
    scores.append(("hint", 0.30, hint_score))

    # 2. Email-specific keyword relevance (20%)
    if keywords:
        kw_matches = sum(1 for kw in keywords if kw.lower() in draft_lower)
        kw_score = min(kw_matches / max(len(keywords), 1), 1.0)
    else:
        kw_score = 0.5  # neutral
    scores.append(("keywords", 0.20, kw_score))

    # 3. Length appropriateness (20%)
    length = len(draft.strip())
    if length < 20:
        length_score = 0.2
    elif length < 50:
        length_score = 0.6
    elif length <= 300:
        length_score = 1.0
    elif length <= 500:
        length_score = 0.7
    else:
        length_score = 0.4  # Too verbose
    scores.append(("length", 0.20, length_score))

    # 4. Professionalism and tone (30%)
    polite_signals = [
        "thank", "please", "appreciate", "happy to", "will do",
        "sure", "acknowledge", "confirm", "noted", "understood",
        "i'll", "i will", "right away", "on it", "absolutely",
    ]
    polite_count = sum(1 for s in polite_signals if s in draft_lower)
    polite_score = min(polite_count * 0.2, 1.0)

    # Negative signals
    negative_signals = ["whatever", "don't care", "idk", "nah", "lol", "lmao"]
    neg_count = sum(1 for s in negative_signals if s in draft_lower)
    polite_score = max(polite_score - neg_count * 0.3, 0.0)
    scores.append(("tone", 0.30, polite_score))

    total = sum(weight * score for _, weight, score in scores)
    return total


def _grade_sla_awareness(priority: int, sla_hours: Optional[float]) -> float:
    """Grade whether the agent's priority assignment reflects the SLA urgency.

    Emails with tight SLAs should get higher priority, while those with relaxed
    SLAs can be lower priority. This rewards agents that understand time-sensitivity.
    """
    if sla_hours is None:
        return 1.0  # No SLA, any priority is fine

    # Map SLA hours to expected minimum priority
    if sla_hours <= 2:
        expected_min_priority = 5
    elif sla_hours <= 8:
        expected_min_priority = 4
    elif sla_hours <= 24:
        expected_min_priority = 3
    elif sla_hours <= 72:
        expected_min_priority = 2
    else:
        expected_min_priority = 1

    if priority >= expected_min_priority:
        return 1.0
    else:
        # Penalty proportional to how far below expected
        diff = expected_min_priority - priority
        return max(1.0 - diff * 0.3, 0.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment:
    """Email Triage environment implementing the OpenEnv 3-method interface.

    Simulates an email inbox where the agent must triage each email:
    categorize it, assign priority, and optionally draft a response.

    Novel mechanics:
    - Email threads with conversation context (agent must reason about history)
    - SLA/deadline tracking (grades urgency awareness)
    - Deceptive phishing emails mixed in (tests detection ability)
    - Multi-signal response grading (keywords, tone, length, action words)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = EmailTriageState()
        self._emails: List[Dict] = []
        self._current_idx: int = 0
        self._step_rewards: List[float] = []
        self._emails_lookup: Dict[str, Dict] = {e["id"]: e for e in EMAILS_DB}
        self._task_config: Dict = {}
        self._thread_seen: Set[str] = set()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: int = 1, **kwargs) -> EmailTriageObservation:
        """Start a new triage episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional episode identifier.
            task_id: Task number (1=easy, 2=medium, 3=hard).

        Returns:
            Initial observation with the first email to triage.
        """
        if seed is not None:
            random.seed(seed)

        task_id = max(1, min(3, task_id))
        self._task_config = TASK_CONFIGS[task_id]
        self._emails = [self._emails_lookup[eid] for eid in self._task_config["email_ids"]]
        self._current_idx = 0
        self._step_rewards = []
        self._thread_seen = set()

        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_emails=len(self._emails),
            processed_emails=0,
            correct_categorizations=0,
            correct_priorities=0,
            response_scores=0.0,
        )

        email = self._emails[0]

        # Build thread context if applicable
        thread_context = ""
        if self._task_config.get("include_threads"):
            thread_context = _build_thread_context(email, self._emails, 0)
            if email.get("thread_id"):
                self._thread_seen.add(email["thread_id"])

        body_with_context = email["body"]
        if thread_context:
            body_with_context = thread_context + "\n" + email["body"]

        sla_info = ""
        if email.get("sla_hours") is not None and self._task_config.get("grade_sla"):
            sla_info = f" [SLA: respond within {email['sla_hours']}h]"

        return EmailTriageObservation(
            done=False,
            reward=None,
            email_id=email["id"],
            email_subject=email["subject"],
            email_from=email["from"],
            email_body=body_with_context,
            email_timestamp=email["timestamp"],
            current_task=self._task_config["description"] + sla_info,
            task_id=task_id,
            inbox_remaining=len(self._emails),
            total_emails=len(self._emails),
            score_so_far=0.0,
            feedback=f"Welcome! {self._task_config['name']}. You have {len(self._emails)} emails to process.",
        )

    def step(self, action: EmailTriageAction, timeout_s: Optional[float] = None,
             **kwargs) -> EmailTriageObservation:
        """Process the agent's triage action for the current email.

        Args:
            action: The agent's triage action.
            timeout_s: Optional timeout (unused, for API compatibility).

        Returns:
            Next observation with the next email or final results.
        """
        self._state.step_count += 1

        if self._current_idx >= len(self._emails):
            return self._make_final_observation("Episode already complete. Call reset() to start a new one.")

        email = self._emails[self._current_idx]
        step_reward = 0.0
        feedback_parts = []

        # Calculate dynamic weights based on active grading dimensions
        active_dimensions = []
        if self._task_config["grade_category"]:
            active_dimensions.append("category")
        if self._task_config["grade_priority"]:
            active_dimensions.append("priority")
        if self._task_config["grade_response"]:
            active_dimensions.append("response")
        if self._task_config.get("grade_sla"):
            active_dimensions.append("sla")

        num_active = len(active_dimensions)
        weight_per_dim = 1.0 / num_active if num_active > 0 else 0.0

        # --- Grade category ---
        if self._task_config["grade_category"]:
            cat_score = _grade_category(action.category, email["category"])
            step_reward += weight_per_dim * cat_score
            if cat_score == 1.0:
                self._state.correct_categorizations += 1
                feedback_parts.append(f"✅ Category '{action.category}' is correct!")
            else:
                feedback_parts.append(f"❌ Category '{action.category}' — expected '{email['category']}'.")

        # --- Grade priority ---
        if self._task_config["grade_priority"]:
            pri_score = _grade_priority(action.priority, email["priority"])
            step_reward += weight_per_dim * pri_score
            if pri_score == 1.0:
                self._state.correct_priorities += 1
                feedback_parts.append(f"✅ Priority {action.priority} is correct!")
            elif pri_score > 0:
                feedback_parts.append(f"⚠️ Priority {action.priority} — expected {email['priority']} (partial credit).")
            else:
                feedback_parts.append(f"❌ Priority {action.priority} — expected {email['priority']}.")

        # --- Grade response ---
        if self._task_config["grade_response"]:
            resp_score = _grade_response(
                action.response_draft,
                email.get("expected_response", ""),
                email.get("keywords", []),
            )
            step_reward += weight_per_dim * resp_score
            self._state.response_scores += resp_score
            if resp_score >= 0.7:
                feedback_parts.append("✅ Good response draft!")
            elif resp_score >= 0.3:
                feedback_parts.append("⚠️ Response could be improved.")
            else:
                feedback_parts.append("❌ Missing or poor response draft.")

        # --- Grade SLA awareness ---
        if self._task_config.get("grade_sla"):
            sla_score = _grade_sla_awareness(action.priority, email.get("sla_hours"))
            step_reward += weight_per_dim * sla_score
            if sla_score >= 0.9:
                feedback_parts.append("✅ SLA-appropriate priority!")
            elif sla_score >= 0.5:
                feedback_parts.append("⚠️ Priority could better reflect SLA urgency.")
            else:
                feedback_parts.append("❌ Priority doesn't match SLA urgency.")

        # --- Handle skip action ---
        if action.action_type == "skip":
            if email["priority"] >= 4:
                step_reward = max(step_reward - 0.3, 0.0)
                feedback_parts.append("⚠️ Skipped an important email! Penalty applied.")
            else:
                feedback_parts.append("Skipped (low-priority email).")

        # --- Step penalty for efficiency ---
        step_reward = max(step_reward - 0.02, 0.0)

        self._step_rewards.append(step_reward)
        self._state.processed_emails += 1
        self._current_idx += 1

        # Calculate running score
        max_possible_per_step = 1.0
        total_possible = max_possible_per_step * self._state.processed_emails
        running_score = sum(self._step_rewards) / total_possible if total_possible > 0 else 0.0

        feedback = " | ".join(feedback_parts)

        # Check if done
        done = self._current_idx >= len(self._emails)

        if done:
            total_possible_all = max_possible_per_step * len(self._emails)
            final_score = sum(self._step_rewards) / total_possible_all if total_possible_all > 0 else 0.0
            if final_score > 0.7:
                final_score = min(final_score + 0.05, 1.0)
            feedback += f" | 🏁 Episode complete! Final score: {final_score:.3f}"
            return EmailTriageObservation(
                done=True,
                reward=round(final_score, 4),
                email_id=email["id"],
                email_subject=email["subject"],
                email_from=email["from"],
                email_body=email["body"],
                email_timestamp=email["timestamp"],
                current_task=self._task_config["description"],
                task_id=self._state.task_id,
                inbox_remaining=0,
                total_emails=self._state.total_emails,
                score_so_far=round(final_score, 4),
                feedback=feedback,
            )

        # Show next email
        next_email = self._emails[self._current_idx]

        # Build thread context for next email
        thread_context = ""
        if self._task_config.get("include_threads"):
            thread_context = _build_thread_context(next_email, self._emails, self._current_idx)
            if next_email.get("thread_id"):
                self._thread_seen.add(next_email["thread_id"])

        body_with_context = next_email["body"]
        if thread_context:
            body_with_context = thread_context + "\n" + next_email["body"]

        sla_info = ""
        if next_email.get("sla_hours") is not None and self._task_config.get("grade_sla"):
            sla_info = f" [SLA: respond within {next_email['sla_hours']}h]"

        return EmailTriageObservation(
            done=False,
            reward=round(step_reward, 4),
            email_id=next_email["id"],
            email_subject=next_email["subject"],
            email_from=next_email["from"],
            email_body=body_with_context,
            email_timestamp=next_email["timestamp"],
            current_task=self._task_config["description"] + sla_info,
            task_id=self._state.task_id,
            inbox_remaining=len(self._emails) - self._current_idx,
            total_emails=self._state.total_emails,
            score_so_far=round(running_score, 4),
            feedback=feedback,
        )

    @property
    def state(self) -> EmailTriageState:
        """Return current episode metadata."""
        return self._state

    def _make_final_observation(self, message: str) -> EmailTriageObservation:
        """Create an observation for an already-finished episode."""
        total_possible = len(self._emails)
        final_score = sum(self._step_rewards) / total_possible if total_possible > 0 else 0.0
        return EmailTriageObservation(
            done=True,
            reward=round(final_score, 4),
            email_id="",
            email_subject="",
            email_from="",
            email_body="",
            email_timestamp="",
            current_task=self._task_config.get("description", ""),
            task_id=self._state.task_id,
            inbox_remaining=0,
            total_emails=self._state.total_emails,
            score_so_far=round(final_score, 4),
            feedback=message,
        )
