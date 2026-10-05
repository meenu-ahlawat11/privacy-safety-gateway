"""Tests for Policy. Synthetic data only."""

import pytest

from app.detectors.base import Detection
from app.policy import Policy, PolicyError
from app.protect import protect
from app.vault import Vault

VALID_YAML = """
default_profile: student
profiles:
  student:
    default_action: mask
    actions:
      EMAIL: mask
      PHONE: redact
  developer:
    default_action: allow
    actions:
      EMAIL: tokenize
"""


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "policy.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_valid_load(tmp_path) -> None:
    policy = Policy.load(_write(tmp_path, VALID_YAML))
    assert set(policy.profile_names()) == {"student", "developer"}


def test_action_lookup_per_type(tmp_path) -> None:
    policy = Policy.load(_write(tmp_path, VALID_YAML))
    action_for = policy.action_for("student")
    assert action_for("EMAIL") == "mask"
    assert action_for("PHONE") == "redact"


def test_default_action_fallback(tmp_path) -> None:
    policy = Policy.load(_write(tmp_path, VALID_YAML))
    action_for = policy.action_for("student")
    assert action_for("IP_ADDRESS") == "mask"


def test_default_profile_used_when_none(tmp_path) -> None:
    policy = Policy.load(_write(tmp_path, VALID_YAML))
    action_for = policy.action_for(None)
    assert action_for("PHONE") == "redact"  # student's action


def test_unknown_profile_error(tmp_path) -> None:
    policy = Policy.load(_write(tmp_path, VALID_YAML))
    with pytest.raises(PolicyError):
        policy.action_for("ghost")


def test_invalid_action_error(tmp_path) -> None:
    bad = VALID_YAML.replace("redact", "explode")
    with pytest.raises(PolicyError):
        Policy.load(_write(tmp_path, bad))


def test_missing_file_error(tmp_path) -> None:
    with pytest.raises(PolicyError):
        Policy.load(tmp_path / "nope.yaml")


def test_invalid_yaml_error(tmp_path) -> None:
    with pytest.raises(PolicyError):
        Policy.load(_write(tmp_path, "profiles: [unclosed\n  bad: : :"))


def test_missing_profiles_error(tmp_path) -> None:
    with pytest.raises(PolicyError):
        Policy.load(_write(tmp_path, "default_profile: x\n"))


def test_missing_default_action_error(tmp_path) -> None:
    bad = """
profiles:
  student:
    actions:
      EMAIL: mask
"""
    with pytest.raises(PolicyError):
        Policy.load(_write(tmp_path, bad))


def test_unknown_default_profile_error(tmp_path) -> None:
    bad = VALID_YAML.replace("default_profile: student", "default_profile: ghost")
    with pytest.raises(PolicyError):
        Policy.load(_write(tmp_path, bad))


def test_real_policy_file_loads() -> None:
    policy = Policy.load("policies/policy.yaml")
    assert "student" in policy.profile_names()
    action_for = policy.action_for("student")
    assert action_for("EMAIL") == "mask"
    assert action_for("CREDIT_CARD") == "block"


def test_integration_action_for_student_into_protect() -> None:
    policy = Policy.load("policies/policy.yaml")
    text = "email alice@example.com phone 9876543210"
    detections = [
        Detection(start=6, end=23, type="EMAIL", confidence=0.9, source="test"),
        Detection(start=30, end=40, type="PHONE", confidence=0.9, source="test"),
    ]
    result = protect(text, detections, policy.action_for("student"), Vault())
    assert result.blocked is False
    assert result.text == "email al***@example.com phone [REDACTED_PHONE]"
    assert result.applied == [("EMAIL", "mask"), ("PHONE", "redact")]
