"""Tests for AuditLog. Synthetic data only."""

import sqlite3

import pytest

from app.audit import AuditLog


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "audit.db")


def _sample(**overrides):
    kwargs = {
        "profile": "student",
        "finding_counts": {"EMAIL": 2, "PHONE": 1},
        "actions": {"EMAIL": "mask", "PHONE": "redact"},
        "injection_score": 0.1,
        "injection_verdict": "allow",
        "decision": "allowed",
        "overrides": 0,
        "prompt_length": 42,
    }
    kwargs.update(overrides)
    return kwargs


def test_table_created(db_path: str) -> None:
    AuditLog(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
        ).fetchone()
    assert row is not None


def test_record_returns_increasing_ids(db_path: str) -> None:
    log = AuditLog(db_path)
    first = log.record(**_sample())
    second = log.record(**_sample())
    third = log.record(**_sample())
    assert first < second < third


def test_recent_newest_first_and_limit(db_path: str) -> None:
    log = AuditLog(db_path)
    for i in range(5):
        log.record(**_sample(prompt_length=i))
    recent = log.recent(limit=3)
    assert len(recent) == 3
    assert [r["prompt_length"] for r in recent] == [4, 3, 2]


def test_summary_totals_and_per_type(db_path: str) -> None:
    log = AuditLog(db_path)
    log.record(**_sample(finding_counts={"EMAIL": 2, "PHONE": 1}, decision="modified"))
    log.record(**_sample(finding_counts={"EMAIL": 1}, decision="allowed"))
    summary = log.summary()
    assert summary["total_events"] == 2
    assert summary["counts_per_type"] == {"EMAIL": 3, "PHONE": 1}
    assert summary["decisions"] == {"modified": 1, "allowed": 1}


def test_blocked_per_day(db_path: str) -> None:
    log = AuditLog(db_path)
    log.record(**_sample(decision="blocked"))
    log.record(**_sample(decision="blocked"))
    log.record(**_sample(decision="allowed"))
    summary = log.summary()
    assert sum(summary["blocked_per_day"].values()) == 2
    for day, count in summary["blocked_per_day"].items():
        assert count == 2
        assert len(day) == 10  # YYYY-MM-DD


def test_invalid_decision_raises(db_path: str) -> None:
    log = AuditLog(db_path)
    with pytest.raises(ValueError):
        log.record(**_sample(decision="explode"))


def test_data_persists_across_instances(db_path: str) -> None:
    AuditLog(db_path).record(**_sample())
    fresh = AuditLog(db_path)
    assert fresh.summary()["total_events"] == 1
    assert len(fresh.recent()) == 1


def test_no_raw_values_in_db_bytes(db_path: str) -> None:
    log = AuditLog(db_path)
    log.record(**_sample(finding_counts={"EMAIL": 1}, actions={"EMAIL": "mask"}))
    raw = open(db_path, "rb").read()
    assert b"alice@example.com" not in raw
    assert b"4111111111111111" not in raw
    assert b"lorem ipsum secret prompt" not in raw
