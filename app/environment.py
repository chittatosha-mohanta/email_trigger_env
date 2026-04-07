from __future__ import annotations

from typing import Optional

from app.models import (
    EpisodeState,
    ResetResponse,
    StepInfo,
    StepResult,
    SupportAction,
    SupportObservation,
    SupportReward,
    TaskDescriptor,
)
from app.tasks import get_task_bundle, grade_action, list_task_descriptors


class SupportTriageEnv:
    """OpenEnv-style environment for customer support triage."""

    def __init__(self) -> None:
        self._state: Optional[EpisodeState] = None
        self._task_order = [task.id for task in list_task_descriptors()]
        self._next_task_idx = 0

    def available_tasks(self) -> list[TaskDescriptor]:
        return list_task_descriptors()

    def reset(self, task_id: Optional[str] = None) -> ResetResponse:
        if task_id is None:
            task_id = self._task_order[self._next_task_idx % len(self._task_order)]
            self._next_task_idx += 1

        bundle = get_task_bundle(task_id)
        descriptor: TaskDescriptor = bundle["descriptor"]
        ticket = bundle["ticket"]
        max_steps = bundle["max_steps"]

        self._state = EpisodeState(
            task=descriptor,
            max_steps=max_steps,
            step_count=0,
            done=False,
            inbox=[ticket],
            active_ticket=ticket,
            history=[],
            cumulative_reward=0.0,
            last_error=None,
            last_action=None,
        )
        return ResetResponse(observation=self._observation())

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def step(self, action: SupportAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            reward = SupportReward(value=0.0, components={}, penalty=0.0, done_bonus=0.0)
            info = StepInfo(
                grader_score=0.0,
                reason="Episode is already done. Call reset() to start a new one.",
                reward_components={},
            )
            return StepResult(
                observation=self._observation(),
                reward=reward,
                done=True,
                info=info,
            )

        self._state.step_count += 1
        self._state.last_error = None

        penalty = 0.0
        if action.ticket_id != self._state.active_ticket.id:
            penalty -= 0.10
            self._state.last_error = "Action ticket_id does not match active ticket."

        score, details, reason = grade_action(self._state.task.id, action)

        # Dense reward shaping:
        # - base from grader score
        # - small time penalty
        # - quality bonus near completion
        time_penalty = -0.02
        done_bonus = 0.10 if score >= 0.90 else 0.0
        reward_value = (0.7 * score) + time_penalty + penalty + done_bonus
        reward_value = max(-1.0, min(1.0, reward_value))

        self._state.last_action = action
        self._state.cumulative_reward += reward_value
        self._state.history.append(
            (
                f"step={self._state.step_count} score={score:.2f} reward={reward_value:.2f} "
                f"priority={action.priority} team={action.team} escalate={action.escalate}"
            )
        )

        if score >= 0.90:
            self._state.done = True
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        reward = SupportReward(
            value=reward_value,
            components=details,
            penalty=penalty + time_penalty,
            done_bonus=done_bonus,
        )
        info = StepInfo(grader_score=score, reason=reason, reward_components=details)
        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=self._state.done,
            info=info,
        )

    def _observation(self) -> SupportObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return SupportObservation(
            goal=self._state.task.goal,
            task_id=self._state.task.id,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            inbox=self._state.inbox,
            active_ticket=self._state.active_ticket,
            history=self._state.history,
            last_error=self._state.last_error,
        )
