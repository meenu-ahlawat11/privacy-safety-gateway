"""Tests for Pipeline. Synthetic data only."""

import pytest

from app.detectors.keyword_trie import KeywordDetector
from app.detectors.regex_det import RegexDetector
from app.pipeline import Pipeline
from app.policy import Policy, PolicyError

POLICY_YAML = """
default_profile: student
profiles:
  student:
    default_action: mask
    actions:
      EMAIL: tokenize
      PHONE: redact
      CREDIT_CARD: block
  developer:
    default_action: allow
    actions:
      EMAIL: tokenize
"""


@pytest.fixture
def policy(tmp_path) -> Policy:
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_YAML, encoding="utf-8")
    return Policy.load(path)


def _pipeline(policy: Policy, keywords: list[str] | None = None) -> Pipeline:
    detectors = [RegexDetector()]
    if keywords:
        detectors.append(KeywordDetector(keywords))
    return Pipeline(policy, detectors)


def test_clean_text_unchanged(policy: Policy) -> None:
    p = _pipeline(policy)
    result, _vault = p.process("nothing sensitive here")
    assert result.protected_text == "nothing sensitive here"
    assert result.blocked is False
    assert result.findings == []
    assert result.profile == "student"


def test_email_tokenized_then_restored(policy: Policy) -> None:
    p = _pipeline(policy)
    result, vault = p.process("mail alice@example.com end")
    assert result.protected_text == "mail <EMAIL_1> end"
    assert p.restore(vault, result.protected_text) == "mail alice@example.com end"


def test_luhn_valid_card_blocked_for_student(policy: Policy) -> None:
    p = _pipeline(policy)
    result, _vault = p.process("card 4111 1111 1111 1111 ok", profile="student")
    assert result.blocked is True
    assert result.protected_text == ""
    assert result.blocked_types == ["CREDIT_CARD"]
    assert result.profile == "student"


def test_keyword_detection_when_keywords_given(policy: Policy) -> None:
    p = _pipeline(policy, keywords=["project falcon"])
    result, _vault = p.process("deploy project falcon now")
    types = [t for t, *_ in result.findings]
    assert "KEYWORD" in types


def test_analyze_merges_and_does_not_modify_text(policy: Policy) -> None:
    p = _pipeline(policy)
    text = "mail alice@example.com call 9876543210"
    detections = p.analyze(text)
    spans = [(d.start, d.end) for d in detections]
    assert spans == sorted(spans)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2
    assert [text[s:e] for s, e in spans] == ["alice@example.com", "9876543210"]


def test_findings_contain_no_raw_values(policy: Policy) -> None:
    p = _pipeline(policy)
    result, _vault = p.process("mail alice@example.com")
    for type_name, start, end, confidence in result.findings:
        assert type_name in ("EMAIL", "PHONE", "CREDIT_CARD", "PAN", "IP_ADDRESS", "API_KEY", "KEYWORD")
        assert isinstance(start, int) and isinstance(end, int)
        assert "alice@example.com" not in type_name


def test_fresh_vault_per_call(policy: Policy) -> None:
    p = _pipeline(policy)
    r1, _ = p.process("mail alice@example.com")
    r2, _ = p.process("mail alice@example.com")
    assert "<EMAIL_1>" in r1.protected_text
    assert "<EMAIL_1>" in r2.protected_text


def test_unknown_profile_raises(policy: Policy) -> None:
    p = _pipeline(policy)
    with pytest.raises(PolicyError):
        p.process("mail alice@example.com", profile="ghost")
