from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Priority = Literal["low", "medium", "high", "urgent"]
Team = Literal["billing", "tech", "trust_safety", "logistics", "general"]


class Ticket(BaseModel):
    id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    subject: str
    body: str
    channel: Literal["email", "chat", "web_form"]
    region: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class SupportAction(BaseModel):
    ticket_id: str
    priority: Priority
    team: Team
    tags: List[str] = Field(default_factory=list)
    escalate: bool = False
    resolve: bool = False
    response_text: str = ""
    note: str = ""


class SupportReward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    penalty: float = 0.0
    done_bonus: float = 0.0


class SupportObservation(BaseModel):
    goal: str
    task_id: str
    step_count: int
    max_steps: int
    inbox: List[Ticket] = Field(default_factory=list)
    active_ticket: Ticket
    history: List[str] = Field(default_factory=list)
    last_error: Optional[str] = None


class TaskDescriptor(BaseModel):
    id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    goal: str


class EpisodeState(BaseModel):
    task: TaskDescriptor
    max_steps: int
    step_count: int = 0
    done: bool = False
    inbox: List[Ticket]
    active_ticket: Ticket
    history: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    last_error: Optional[str] = None
    last_action: Optional[SupportAction] = None


class StepInfo(BaseModel):
    grader_score: float
    reason: str
    reward_components: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: SupportObservation
    reward: SupportReward
    done: bool
    info: StepInfo


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ResetResponse(BaseModel):
    observation: SupportObservation


class StateResponse(BaseModel):
    state: EpisodeState


class TaskScore(BaseModel):
    task_id: str
    score: float
    details: Dict[str, float] = Field(default_factory=dict)
