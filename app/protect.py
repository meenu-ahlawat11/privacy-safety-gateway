"""Apply privacy actions to detected spans."""

from dataclasses import dataclass, field
from typing import Callable

from app.detectors.base import Detection
from app.merge import merge_detections
from app.vault import Vault


@dataclass(frozen=True)
class ProtectResult:
    """Outcome of protect().

    applied holds only (detection_type, action) pairs — never raw
    sensitive values.
    """

    text: str
    blocked: bool
    blocked_types: list[str] = field(default_factory=list)
    applied: list[tuple[str, str]] = field(default_factory=list)


def _mask_value(detection_type: str, value: str) -> str:
    if detection_type == "EMAIL":
        local, sep, domain = value.partition("@")
        if not sep:
            return "*" * len(value)
        keep = local[:2] if len(local) >= 2 else local[:1]
        return f"{keep}***@{domain}"
    if detection_type in ("PHONE", "CREDIT_CARD", "AADHAAR"):
        digit_positions = [i for i, ch in enumerate(value) if ch.isdigit()]
        keep_last = set(digit_positions[-4:])
        chars = []
        for i, ch in enumerate(value):
            if ch.isdigit() and i not in keep_last:
                chars.append("*")
            else:
                chars.append(ch)
        return "".join(chars)
    return "*" * len(value)


def protect(
    text: str,
    detections: list[Detection],
    action_for: Callable[[str], str],
    vault: Vault,
) -> ProtectResult:
    """Apply actions to detections in text.

    Overlapping detections are merged first (merge_detections keeps the
    highest-confidence/longest/earliest winner per group), so replaced
    spans never conflict.

    Right-to-left technique: replacements are applied from the end of
    the string back to the start. Every replacement only changes the
    region at or after its own start offset, so offsets of detections
    still to be processed (all to the left) remain valid — no offset
    recalculation is needed.
    """
    merged = merge_detections(detections)

    actions: list[tuple[Detection, str]] = []
    for d in merged:
        action = action_for(d.type)
        if action not in ("allow", "mask", "redact", "tokenize", "block"):
            raise ValueError(f"unknown action: {action!r}")
        actions.append((d, action))

    blocked = [d for d, action in actions if action == "block"]
    if blocked:
        return ProtectResult(
            text="",
            blocked=True,
            blocked_types=[d.type for d in blocked],
            applied=[(d.type, "block") for d in blocked],
        )

    result = text
    applied: list[tuple[str, str]] = []
    for d, action in sorted(actions, key=lambda pair: pair[0].start, reverse=True):
        value = result[d.start : d.end]
        if action == "allow":
            replacement = value
        elif action == "redact":
            replacement = f"[REDACTED_{d.type}]"
        elif action == "tokenize":
            replacement = vault.tokenize(d.type, value)
        else:  # mask
            replacement = _mask_value(d.type, value)
        result = result[: d.start] + replacement + result[d.end :]
        applied.append((d.type, action))

    applied.reverse()
    return ProtectResult(text=result, blocked=False, blocked_types=[], applied=applied)
