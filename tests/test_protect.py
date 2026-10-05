"""Tests for protect(). Synthetic data only."""

import pytest

from app.detectors.base import Detection
from app.protect import protect
from app.vault import Vault


def _d(start: int, end: int, type: str, confidence: float = 0.9) -> Detection:
    return Detection(start=start, end=end, type=type, confidence=confidence, source="test")


def _action_for(mapping: dict[str, str]):
    return lambda t: mapping[t]


def test_mask_email() -> None:
    text = "mail alice@example.com end"
    d = [_d(5, 22, "EMAIL")]
    r = protect(text, d, _action_for({"EMAIL": "mask"}), Vault())
    assert r.text == "mail al***@example.com end"
    assert r.applied == [("EMAIL", "mask")]
    assert r.blocked is False


def test_mask_email_short_local_part() -> None:
    text = "mail a@example.com end"
    d = [_d(5, 19, "EMAIL")]
    r = protect(text, d, _action_for({"EMAIL": "mask"}), Vault())
    assert r.text == "mail a***@example.com end"


def test_mask_phone() -> None:
    text = "call +91 98765 43210 now"
    d = [_d(5, 20, "PHONE")]
    r = protect(text, d, _action_for({"PHONE": "mask"}), Vault())
    assert r.text == "call +** ***** *3210 now"


def test_mask_credit_card() -> None:
    text = "card 4111 1111 1111 1111 ok"
    d = [_d(5, 24, "CREDIT_CARD")]
    r = protect(text, d, _action_for({"CREDIT_CARD": "mask"}), Vault())
    assert r.text == "card **** **** **** 1111 ok"


def test_mask_other_type_all_stars() -> None:
    text = "pan ABCDE1234F ok"
    d = [_d(4, 14, "PAN")]
    r = protect(text, d, _action_for({"PAN": "mask"}), Vault())
    assert r.text == "pan ********** ok"


def test_redact() -> None:
    text = "mail alice@example.com end"
    d = [_d(5, 22, "EMAIL")]
    r = protect(text, d, _action_for({"EMAIL": "redact"}), Vault())
    assert r.text == "mail [REDACTED_EMAIL] end"


def test_allow_keeps_text() -> None:
    text = "mail alice@example.com end"
    d = [_d(5, 22, "EMAIL")]
    r = protect(text, d, _action_for({"EMAIL": "allow"}), Vault())
    assert r.text == text
    assert r.applied == [("EMAIL", "allow")]


def test_tokenize() -> None:
    text = "mail alice@example.com end"
    d = [_d(5, 22, "EMAIL")]
    v = Vault()
    r = protect(text, d, _action_for({"EMAIL": "tokenize"}), v)
    assert r.text == "mail <EMAIL_1> end"
    assert v.restore(r.text) == text


def test_block_returns_empty_text() -> None:
    text = "call 9876543210 now"
    d = [_d(5, 15, "PHONE")]
    r = protect(text, d, _action_for({"PHONE": "block"}), Vault())
    assert r.blocked is True
    assert r.text == ""
    assert r.blocked_types == ["PHONE"]
    assert r.applied == [("PHONE", "block")]


def test_multiple_detections_one_text() -> None:
    text = "mail alice@example.com or call 9876543210"
    d = [_d(5, 22, "EMAIL"), _d(31, 41, "PHONE")]
    r = protect(text, d, _action_for({"EMAIL": "redact", "PHONE": "mask"}), Vault())
    assert r.text == "mail [REDACTED_EMAIL] or call ******3210"
    assert r.applied == [("EMAIL", "redact"), ("PHONE", "mask")]


def test_text_before_between_after_preserved() -> None:
    text = "abXcdYef"
    d = [_d(2, 3, "PAN"), _d(5, 6, "PAN")]
    r = protect(text, d, _action_for({"PAN": "redact"}), Vault())
    assert r.text == "ab[REDACTED_PAN]cd[REDACTED_PAN]ef"


def test_overlapping_detections_merged_before_applying() -> None:
    text = "aaa@bbb.com see"
    # two overlapping detections over "aaa@bbb.com"
    d = [_d(0, 11, "EMAIL", 0.99), _d(0, 5, "PAN", 0.85)]
    r = protect(text, d, _action_for({"EMAIL": "redact", "PAN": "mask"}), Vault())
    assert r.text == "[REDACTED_EMAIL] see"
    assert r.applied == [("EMAIL", "redact")]


def test_unknown_action_raises() -> None:
    text = "mail alice@example.com"
    d = [_d(5, 22, "EMAIL")]
    with pytest.raises(ValueError):
        protect(text, d, _action_for({"EMAIL": "explode"}), Vault())


def test_full_tokenize_then_restore_round_trip() -> None:
    text = "email alice@example.com call 9876543210"
    d = [_d(6, 23, "EMAIL"), _d(29, 39, "PHONE")]
    v = Vault()
    r = protect(text, d, _action_for({"EMAIL": "tokenize", "PHONE": "tokenize"}), v)
    assert r.text == "email <EMAIL_1> call <PHONE_1>"
    assert v.restore(r.text) == text


def test_applied_never_contains_raw_values() -> None:
    text = "mail alice@example.com call 9876543210"
    d = [_d(5, 22, "EMAIL"), _d(28, 38, "PHONE")]
    r = protect(text, d, _action_for({"EMAIL": "mask", "PHONE": "tokenize"}), Vault())
    for type_name, action in r.applied:
        assert type_name in ("EMAIL", "PHONE")
        assert action in ("mask", "tokenize")
        assert "alice@example.com" not in type_name
        assert "9876543210" not in action
