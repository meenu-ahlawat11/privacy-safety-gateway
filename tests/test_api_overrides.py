"""Tests for chat overrides. Mock LLM, tmp-path DB, tmp policy."""

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditLog
from app.detectors.injection import InjectionScanner
from app.detectors.regex_det import RegexDetector
from app.llm_client import MockLLMClient
from app.main import (
    app,
    get_audit_log,
    get_llm_client,
    get_pipeline,
    get_scanner,
)
from app.pipeline import Pipeline
from app.policy import Policy

POLICY_YAML = """
default_profile: student
profiles:
  student:
    default_action: mask
    actions:
      EMAIL: tokenize
      PHONE: redact
      CREDIT_CARD: block
"""

SAMPLE_EMAIL = "alice@example.com"
SAMPLE_CARD = "4111 1111 1111 1111"


class RecordingLLM(MockLLMClient):
    def __init__(self) -> None:
        self.received: list[str] = []

    def complete(self, prompt: str) -> str:
        self.received.append(prompt)
        return super().complete(prompt)


@pytest.fixture()
def env(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_YAML, encoding="utf-8")
    pipeline = Pipeline(Policy.load(policy_path), [RegexDetector()])
    audit = AuditLog(str(tmp_path / "audit.db"))
    llm = RecordingLLM()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_audit_log] = lambda: audit
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_scanner] = lambda: InjectionScanner()
    client = TestClient(app)
    yield client, llm, audit, tmp_path
    app.dependency_overrides.clear()


def test_override_tokenize_to_mask_changes_llm_input(env) -> None:
    client, llm, _audit, _tmp = env
    resp = client.post(
        "/v1/chat",
        json={
            "prompt": f"mail {SAMPLE_EMAIL} end",
            "overrides": {"EMAIL": "mask"},
        },
    )
    body = resp.json()
    assert body["decision"] == "modified"
    assert body["applied"] == [{"type": "EMAIL", "action": "mask"}]
    assert len(llm.received) == 1
    assert SAMPLE_EMAIL not in llm.received[0]
    assert "al***@example.com" in llm.received[0]


def test_override_cannot_weaken_policy_block(env) -> None:
    client, llm, _audit, _tmp = env
    resp = client.post(
        "/v1/chat",
        json={
            "prompt": f"card {SAMPLE_CARD} ok",
            "overrides": {"CREDIT_CARD": "allow"},
        },
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["decision"] == "blocked"
    assert body["blocked_types"] == ["CREDIT_CARD"]
    assert llm.received == []


def test_invalid_action_gives_422(env) -> None:
    client, _llm, _audit, _tmp = env
    resp = client.post(
        "/v1/chat",
        json={"prompt": "hi", "overrides": {"EMAIL": "explode"}},
    )
    assert resp.status_code == 422


def test_block_override_gives_422(env) -> None:
    client, _llm, _audit, _tmp = env
    resp = client.post(
        "/v1/chat",
        json={"prompt": "hi", "overrides": {"EMAIL": "block"}},
    )
    assert resp.status_code == 422


def test_audit_overrides_count_increases(env) -> None:
    client, _llm, _audit, _tmp = env
    client.post(
        "/v1/chat",
        json={
            "prompt": f"mail {SAMPLE_EMAIL} end",
            "overrides": {"EMAIL": "mask"},
        },
    )
    events = env[2].recent(limit=5)
    assert events[0]["overrides"] == 1

    client.post("/v1/chat", json={"prompt": f"mail {SAMPLE_EMAIL} end"})
    events = env[2].recent(limit=5)
    assert events[0]["overrides"] == 0


def test_no_overrides_behaves_as_before(env) -> None:
    client, llm, _audit, _tmp = env
    resp = client.post("/v1/chat", json={"prompt": f"mail {SAMPLE_EMAIL} end"})
    body = resp.json()
    assert body["decision"] == "modified"
    assert body["applied"] == [{"type": "EMAIL", "action": "tokenize"}]
    assert len(llm.received) == 1
    assert "<EMAIL_1>" in llm.received[0]
    assert SAMPLE_EMAIL in body["reply"]
