"""Tests for RegexDetector. Synthetic data only."""

import pytest

from app.detectors.regex_det import RegexDetector
from app.detectors.validators import verhoeff_check_digit

_detector = RegexDetector()


def _make_aadhaar() -> str:
    prefix = "23456789012"  # 11 digits, starts with 2-9
    return prefix + str(verhoeff_check_digit(prefix))


AADHAAR = _make_aadhaar()
AADHAAR_SPACED = f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}"


def _bad_aadhaar() -> str:
    """12 digits, starts 2-9, but fails Verhoeff."""
    i = 5
    replacement = "0" if AADHAAR[i] != "0" else "1"
    return AADHAAR[:i] + replacement + AADHAAR[i + 1 :]


@pytest.mark.parametrize(
    "text,expected_type,expected_span,expected_confidence",
    [
        ("Contact me at alice@example.com today", "EMAIL", "alice@example.com", 0.85),
        ("Call +91 9876543210 now", "PHONE", "+91 9876543210", 0.85),
        ("Call 9876543210 now", "PHONE", "9876543210", 0.85),
        ("Call 919876543210 now", "PHONE", "919876543210", 0.85),
        ("Card 4111 1111 1111 1111 ok", "CREDIT_CARD", "4111 1111 1111 1111", 0.99),
        ("Card 4111111111111111 ok", "CREDIT_CARD", "4111111111111111", 0.99),
        (f"Aadhaar {AADHAAR_SPACED} ok", "AADHAAR", AADHAAR_SPACED, 0.99),
        (f"Aadhaar {AADHAAR} ok", "AADHAAR", AADHAAR, 0.99),
        ("PAN ABCDE1234F ok", "PAN", "ABCDE1234F", 0.85),
        ("IP 192.168.0.1 ok", "IP_ADDRESS", "192.168.0.1", 0.85),
        ("key AKIAIOSFODNN7EXAMPLE ok", "API_KEY", "AKIAIOSFODNN7EXAMPLE", 0.85),
        ("key sk-abcdefghijklmnopqrstuvwxyz ok", "API_KEY", "sk-abcdefghijklmnopqrstuvwxyz", 0.85),
        ("key ghp_" + "a" * 36, "API_KEY", "ghp_" + "a" * 36, 0.85),
    ],
)
def test_positive_detection(
    text: str, expected_type: str, expected_span: str, expected_confidence: float
) -> None:
    detections = [d for d in _detector.detect(text) if d.type == expected_type]
    assert len(detections) == 1
    detection = detections[0]
    assert text[detection.start : detection.end] == expected_span
    assert detection.confidence == expected_confidence
    assert detection.source == "regex"


@pytest.mark.parametrize(
    "text,absent_type",
    [
        ("Card 4111111111111112 ok", "CREDIT_CARD"),  # Luhn-failing 16 digits
        (f"Aadhaar {_bad_aadhaar()} ok", "AADHAAR"),  # Verhoeff-failing 12 digits
        ("Phone 5876543210 ok", "PHONE"),  # starts with 5
        ("IP 999.1.1.1 ok", "IP_ADDRESS"),  # octet out of range
        ("Phone 98765432101 ok", "PHONE"),  # inside a longer digit run
        ("PAN abcde1234f ok", "PAN"),  # lowercase
        ("Card 4111-1111-1111-1112 ok", "CREDIT_CARD"),  # Luhn-failing w/ hyphens
        ("key AKIA123", "API_KEY"),  # too short
    ],
)
def test_near_miss_does_not_match(text: str, absent_type: str) -> None:
    assert all(d.type != absent_type for d in _detector.detect(text))


def test_multiple_types_in_one_prompt() -> None:
    text = (
        f"Email alice@example.com, phone +91 9876543210, card 4111 1111 1111 1111, "
        f"aadhaar {AADHAAR_SPACED}, pan ABCDE1234F, ip 10.0.0.1, key sk-{'a' * 20}"
    )
    detections = _detector.detect(text)
    assert [d.type for d in detections] == [
        "EMAIL",
        "PHONE",
        "CREDIT_CARD",
        "AADHAAR",
        "PAN",
        "IP_ADDRESS",
        "API_KEY",
    ]
    assert [text[d.start : d.end] for d in detections] == [
        "alice@example.com",
        "+91 9876543210",
        "4111 1111 1111 1111",
        AADHAAR_SPACED,
        "ABCDE1234F",
        "10.0.0.1",
        f"sk-{'a' * 20}",
    ]
    confidences = {d.type: d.confidence for d in detections}
    assert confidences["CREDIT_CARD"] == 0.99
    assert confidences["AADHAAR"] == 0.99
    assert all(d.source == "regex" for d in detections)
