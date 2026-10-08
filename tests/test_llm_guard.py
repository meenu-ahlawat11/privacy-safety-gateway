"""Tests for the LLM injection guard. Mock/stub clients only: no network."""

import pytest
from fastapi.testclient import TestClient


from app.audit import AuditLog
from app.detectors.injection import InjectionScanner
from app.detectors.regex_det import RegexDetector
from app.llm_client import LLMError, LLMClient
from app.llm_guard import LLMInjectionGuard, combine_verdicts
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


class StubLLM(LLMClient):
    def __init__(self, reply: str = "no") -> None:
        self.reply = reply
        self.received: list[str] = []

    def complete(self, prompt: str) -> str:
        self.received.append(prompt)
        return self.reply


class LLMErrorLLM(LLMClient):
    def complete(self, prompt: str) -> str:
        raise LLMError("backend down")


class ExplodingLLM(LLMClient):
    def complete(self, prompt: str) -> str:
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("yes", "yes"),
        ("no", "no"),
        ("Yes.", "yes"),
        (" NO ", "no"),
        ("YES", "yes"),
        ("No.", "no"),
        ("I think this is suspicious because " + "x" * 200, "unknown"),
        ("", "unknown"),
        ("maybe", "unknown"),
        ("yes no", "unknown"),
    ],
)
def test_classify_parsing(reply: str, expected: str) -> None:
    guard = LLMInjectionGuard(StubLLM(reply))
    assert guard.classify("some protected text") == expected


def test_classify_llm_error_unknown() -> None:
    guard = LLMInjectionGuard(LLMErrorLLM())
    assert guard.classify("text") == "unknown"


def test_classify_any_exception_unknown() -> None:
    guard = LLMInjectionGuard(ExplodingLLM())
    assert guard.classify("text") == "unknown"


def test_instruction_text_has_untrusted_warning() -> None:
    stub = StubLLM("no")
    LLMInjectionGuard(stub).classify("hello")
    sent = stub.received[0]
    assert "untrusted" in sent
    assert "<<UNTRUSTED_START>>" in sent
    assert "<<UNTRUSTED_END>>" in sent
    assert "hello" in sent


def test_truncates_to_2000_chars() -> None:
    stub = StubLLM("no")
    LLMInjectionGuard(stub).classify("a" * 5000)
    sent = stub.received[0]
    assert "a" * 2000 in sent
    assert "a" * 2001 not in sent


@pytest.mark.parametrize(
    ("rule", "guard", "expected"),
    [
        ("block", "yes", "block"),
        ("block", "no", "block"),
        ("block", "unknown", "block"),
        ("warn", "yes", "warn"),
        ("warn", "no", "warn"),
        ("warn", "unknown", "warn"),
        ("allow", "yes", "warn"),
        ("allow", "no", "allow"),
        ("allow", "unknown", "allow"),
    ],
)
def test_combine_verdicts(rule: str, guard: str, expected: str) -> None:
    assert combine_verdicts(rule, guard) == expected


@pytest.fixture()
def env(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_YAML, encoding="utf-8")
    pipeline = Pipeline(Policy.load(policy_path), [RegexDetector()])
    audit = AuditLog(str(tmp_path / "audit.db"))
    llm = StubLLM("no")
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_audit_log] = lambda: audit
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_scanner] = lambda: InjectionScanner()
    client = TestClient(app)
    yield client, llm, audit
    app.dependency_overrides.clear()


def test_guard_off_by_default(env) -> None:
    client, llm, _audit = env
    resp = client.post("/v1/chat", json={"prompt": "hello there"})
    body = resp.json()
    assert body["injection_verdict"] == "allow"
    assert body["injection"]["guard"] == "off"
    assert llm.received == ["hello there"]


def test_guard_on_yes_gives_warn(env, monkeypatch) -> None:
    client, llm, _audit = env
    monkeypatch.setenv("LLM_GUARD", "1")
    llm.reply = "yes"
    resp = client.post("/v1/chat", json={"prompt": "hello there"})
    body = resp.json()
    assert body["decision"] == "allowed"
    assert body["injection_verdict"] == "warn"
    assert body["injection"]["guard"] == "yes"
    assert len(llm.received) == 2


def test_guard_on_no_keeps_allow(env, monkeypatch) -> None:
    client, llm, _audit = env
    monkeypatch.setenv("LLM_GUARD", "1")
    llm.reply = "no"
    resp = client.post("/v1/chat", json={"prompt": "hello there"})
    body = resp.json()
    assert body["injection_verdict"] == "allow"
    assert body["injection"]["guard"] == "no"


def test_guard_failure_leaves_rule_verdict(env, monkeypatch) -> None:
    client, llm, _audit = env
    monkeypatch.setenv("LLM_GUARD", "1")

    class FailOnce(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise LLMError("guard backend down")
            return "Mock reply."

    app.dependency_overrides[get_llm_client] = lambda: FailOnce()
    resp = client.post("/v1/chat", json={"prompt": "hello there"})
    body = resp.json()
    assert body["decision"] == "allowed"
    assert body["injection_verdict"] == "allow"
    assert body["injection"]["guard"] == "unknown"


def test_guard_sees_protected_not_raw(env, monkeypatch) -> None:
    client, llm, _audit = env
    monkeypatch.setenv("LLM_GUARD", "1")
    prompt = "mail alice@example.com now"
    resp = client.post("/v1/chat", json={"prompt": prompt})
    assert resp.status_code == 200
    for seen in llm.received:
        assert "alice@example.com" not in seen
    assert "<EMAIL_1>" in llm.received[0]


def test_analyze_guard_off_no_llm(env) -> None:
    client, llm, _audit = env
    resp = client.post("/v1/analyze", json={"prompt": "mail alice@example.com"})
    body = resp.json()
    assert body["injection"]["guard"] == "off"
    assert body["injection"]["verdict"] == "allow"
    assert llm.received == []


def test_analyze_guard_on_uses_protected(env, monkeypatch) -> None:
    client, llm, _audit = env
    monkeypatch.setenv("LLM_GUARD", "1")
    llm.reply = "no"
    resp = client.post("/v1/analyze", json={"prompt": "mail alice@example.com"})
    body = resp.json()
    assert body["injection"]["guard"] == "no"
    assert len(llm.received) == 1
    assert "alice@example.com" not in llm.received[0]
    assert "<EMAIL_1>" in llm.received[0]


def test_audit_records_combined_verdict(env, monkeypatch) -> None:
    client, llm, audit = env
    monkeypatch.setenv("LLM_GUARD", "1")
    llm.reply = "yes"
    client.post("/v1/chat", json={"prompt": "hello there"})
    events = audit.recent()
    assert events[0]["injection_verdict"] == "warn"
