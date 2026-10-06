"""FastAPI application: gateway between user and LLM."""

from collections import Counter
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.audit import AuditLog
from app.detectors.injection import InjectionScanner
from app.llm_client import LLMClient, LLMError, get_client
from app.pipeline import Pipeline
from app.policy import PolicyError

app = FastAPI(title="Privacy & Safety Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- lazy app.state factories (overridable in tests) ----------


def get_pipeline(request: Request) -> Pipeline:
    if not hasattr(request.app.state, "pipeline"):
        request.app.state.pipeline = Pipeline.default()
    return request.app.state.pipeline


def get_scanner(request: Request) -> InjectionScanner:
    if not hasattr(request.app.state, "injection_scanner"):
        request.app.state.injection_scanner = InjectionScanner()
    return request.app.state.injection_scanner


def get_audit_log(request: Request) -> AuditLog:
    if not hasattr(request.app.state, "audit_log"):
        request.app.state.audit_log = AuditLog()
    return request.app.state.audit_log


def get_llm_client(request: Request) -> LLMClient:
    if not hasattr(request.app.state, "llm_client"):
        request.app.state.llm_client = get_client()
    return request.app.state.llm_client


# ---------- request/response models ----------


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    profile: str | None = None


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    profile: str | None = None


# ---------- endpoints ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/profiles")
def profiles(pipeline: Pipeline = Depends(get_pipeline)) -> dict[str, Any]:
    return {
        "profiles": pipeline._policy.profile_names(),
        "default_profile": pipeline._policy._default_profile,
    }


@app.post("/v1/analyze")
def analyze(
    req: AnalyzeRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    scanner: InjectionScanner = Depends(get_scanner),
) -> dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="empty prompt")
    try:
        action_for = pipeline._policy.action_for(req.profile)
    except PolicyError as exc:
        raise HTTPException(status_code=404, detail="unknown profile") from exc

    injection = scanner.scan(req.prompt)
    detections = pipeline.analyze(req.prompt, req.profile)
    findings = [
        {
            "type": d.type,
            "start": d.start,
            "end": d.end,
            "confidence": d.confidence,
            "action": action_for(d.type),
        }
        for d in detections
    ]
    categories = sorted({m.category for m in injection.matches})
    return {
        "findings": findings,
        "injection": {
            "score": injection.score,
            "verdict": injection.verdict,
            "categories": categories,
        },
        "prompt_length": len(req.prompt),
    }


@app.post("/v1/chat")
def chat(
    req: ChatRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    scanner: InjectionScanner = Depends(get_scanner),
    audit_log: AuditLog = Depends(get_audit_log),
    llm: LLMClient = Depends(get_llm_client),
) -> dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="empty prompt")

    injection = scanner.scan(req.prompt)

    def _audit(decision: str, counts: dict[str, int], acts: dict[str, str]) -> None:
        audit_log.record(
            profile=req.profile or pipeline._policy._default_profile or "",
            finding_counts=counts,
            actions=acts,
            injection_score=injection.score,
            injection_verdict=injection.verdict,
            decision=decision,
            prompt_length=len(req.prompt),
        )

    if injection.verdict == "block":
        _audit("blocked", {}, {})
        return {
            "decision": "blocked",
            "reason": "prompt_injection",
            "injection_verdict": injection.verdict,
        }

    try:
        result, vault = pipeline.process(req.prompt, req.profile)
    except PolicyError as exc:
        raise HTTPException(status_code=404, detail="unknown profile") from exc

    counts = dict(Counter(t for t, *_ in result.findings))
    acts: dict[str, str] = {}
    for type_name, action in result.applied:
        acts[type_name] = action

    if result.blocked:
        _audit("blocked", counts, acts)
        return {
            "decision": "blocked",
            "blocked_types": result.blocked_types,
            "injection_verdict": injection.verdict,
        }

    try:
        raw_reply = llm.complete(result.protected_text)
    except LLMError as exc:
        _audit("blocked", counts, acts)
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    reply = pipeline.restore(vault, raw_reply)
    decision = "modified" if result.applied else "allowed"
    _audit(decision, counts, acts)
    return {
        "decision": decision,
        "reply": reply,
        "applied": [{"type": t, "action": a} for t, a in result.applied],
        "injection_verdict": injection.verdict,
    }


@app.get("/v1/audit/summary")
def audit_summary(audit_log: AuditLog = Depends(get_audit_log)) -> dict[str, Any]:
    return audit_log.summary()
