from app.environment import SupportTriageEnv
from app.models import SupportAction
from app.tasks import TASKS, grade_action


def test_grade_action_score_range_all_tasks() -> None:
    for task_id in TASKS.keys():
        action = SupportAction(
            ticket_id=TASKS[task_id]["ticket"].id,
            priority="high",
            team="billing",
            tags=[],
            escalate=False,
            resolve=False,
            response_text="sorry we will investigate within 24 hours",
            note="test",
        )
        score, details, _ = grade_action(task_id, action)
        assert 0.0 <= score <= 1.0
        for v in details.values():
            assert 0.0 <= v <= 1.0


def test_reset_step_state_determinism() -> None:
    env = SupportTriageEnv()
    obs1 = env.reset("easy_priority_routing").observation
    state1 = env.state()
    assert obs1.task_id == "easy_priority_routing"
    assert state1.task.id == "easy_priority_routing"

    action = SupportAction(
        ticket_id=obs1.active_ticket.id,
        priority="high",
        team="billing",
        tags=["refund_check"],
        escalate=False,
        resolve=False,
        response_text="Sorry, we will investigate and reply within 24 hours.",
        note="triage",
    )
    res1 = env.step(action)

    env2 = SupportTriageEnv()
    obs2 = env2.reset("easy_priority_routing").observation
    res2 = env2.step(action.model_copy(update={"ticket_id": obs2.active_ticket.id}))

    assert res1.info.grader_score == res2.info.grader_score
    assert res1.done == res2.done


def test_step_requires_reset() -> None:
    env = SupportTriageEnv()
    action = SupportAction(
        ticket_id="TKT-1001",
        priority="high",
        team="billing",
        tags=["refund_check"],
        escalate=False,
        resolve=False,
        response_text="Sorry, we will investigate and reply within 24 hours.",
        note="triage",
    )
    try:
        env.step(action)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Call reset() first" in str(exc)

