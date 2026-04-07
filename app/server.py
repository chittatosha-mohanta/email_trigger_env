from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.environment import SupportTriageEnv
from app.models import (
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResult,
    SupportAction,
    TaskDescriptor,
)

app = FastAPI(title="Support Triage OpenEnv", version="0.1.0")
env = SupportTriageEnv()


@app.exception_handler(RuntimeError)
def runtime_error_handler(_request, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "support-triage-openenv"}


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks", response_model=list[TaskDescriptor])
def tasks() -> list[TaskDescriptor]:
    return env.available_tasks()


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    try:
        return env.reset(task_id=req.task_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: SupportAction) -> StepResult:
    try:
        return env.step(action)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    try:
        return StateResponse(state=env.state())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
