"""Regex-based detector for common sensitive data patterns."""

import re

from app.detectors.base import Detection, Detector
from app.detectors.validators import luhn_valid, verhoeff_valid

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+91[\s-]?|91[\s-]?)?[6-9]\d{9}(?!\d)"
)

_CREDIT_CARD_RE = re.compile(r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)")

_AADHAAR_RE = re.compile(r"(?<!\d)[2-9](?:[ \-]?\d){11}(?!\d)")

_PAN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])")

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP_RE = re.compile(
    r"(?<![\d.])" + _OCTET + r"(?:\." + _OCTET + r"){3}" + r"(?![\d.])"
)

_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36})"
    r"(?![A-Za-z0-9_-])"
)


class RegexDetector(Detector):
    """Detects sensitive data using regular expressions."""

    def detect(self, text: str) -> list[Detection]:
        detections: list[Detection] = []

        for match in _EMAIL_RE.finditer(text):
            detections.append(self._make(match, "EMAIL", 0.85))

        for match in _PHONE_RE.finditer(text):
            detections.append(self._make(match, "PHONE", 0.85))

        for match in _CREDIT_CARD_RE.finditer(text):
            if luhn_valid(match.group(0)):
                detections.append(self._make(match, "CREDIT_CARD", 0.99))

        for match in _AADHAAR_RE.finditer(text):
            if verhoeff_valid(match.group(0)):
                detections.append(self._make(match, "AADHAAR", 0.99))

        for match in _PAN_RE.finditer(text):
            detections.append(self._make(match, "PAN", 0.85))

        for match in _IP_RE.finditer(text):
            detections.append(self._make(match, "IP_ADDRESS", 0.85))

        for match in _API_KEY_RE.finditer(text):
            detections.append(self._make(match, "API_KEY", 0.85))

        detections.sort(key=lambda d: d.start)
        return detections

    def _make(self, match: re.Match[str], type: str, confidence: float) -> Detection:
        return Detection(
            start=match.start(),
            end=match.end(),
            type=type,
            confidence=confidence,
            source="regex",
        )
