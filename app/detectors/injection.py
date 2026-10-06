"""Regex-based prompt-injection scanner.

Scoring: each match contributes a weight in [0, 1]. Weights are combined
with the probabilistic-OR rule, computed by hand:

    score = 1 - product(1 - w_i)   over all match weights

so a single 0.8 match scores 0.8, two 0.5 matches score 0.75, etc.
The result is rounded to 3 decimals and never exceeds 1.0.

Thresholds: score >= block_threshold -> "block";
score >= warn_threshold -> "warn"; otherwise "allow".

Limitations: this scanner catches only selected, syntactic classes of
prompt injection (the categories encoded in the rules below). It has no
semantic understanding, so paraphrased, multilingual, or novel attack
phrasings will be missed, and some benign text may still score as warn.
"""

import re
from dataclasses import dataclass, field

_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"


@dataclass(frozen=True)
class InjectionMatch:
    """A single injection match. Stores no matched text."""

    category: str
    start: int
    end: int
    weight: float


@dataclass
class InjectionResult:
    """Aggregate scan outcome."""

    score: float
    verdict: str
    matches: list[InjectionMatch] = field(default_factory=list)


def _normalize(text: str) -> tuple[str, list[int]]:
    """Strip zero-width characters; return normalized text and an index map.

    The index map has one entry per normalized character plus a final entry,
    mapping each normalized offset back to the offset in the ORIGINAL text.
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for orig_index, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        normalized_chars.append(ch)
        index_map.append(orig_index)
    index_map.append(len(text))
    return "".join(normalized_chars), index_map


_RULES: list[tuple[str, re.Pattern[str], float]] = [
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all|any|the)?\s*"
            r"(?:previous|prior|above)\s+(?:instructions?|rules?)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:reveal|show|print|repeat)\s+(?:your|the)\s+"
            r"(?:system\s+prompt|hidden\s+instructions?)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"(?:you\s+are\s+now\s+dan|developer\s+mode\s+enabled|"
            r"act\s+as\s+an?\s+unrestricted\s+(?:ai|assistant|model))",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:bypass|disable)\s+(?:your|the)\s+(?:safety\s+)?"
            r"(?:filters?|restrictions?)|without\s+any\s+restrictions",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(
            r"(?:send|post|forward)\s+the\s+(?:data|conversation|api\s+key)\s+"
            r"to\s+(?:a\s+)?(?:url|email|server|webhook)|"
            r"include\s+(?:the\s+)?(?:secrets?|api\s+key|credentials?)\s+"
            r"in\s+(?:the\s+)?reply",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "fake_role_marker",
        re.compile(r"^(?:system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
        0.3,
    ),
]


class InjectionScanner:
    """Scans text for prompt-injection patterns and scores them."""

    def __init__(self, warn_threshold: float = 0.4, block_threshold: float = 0.8) -> None:
        self.warn_threshold = warn_threshold
        self.block_threshold = block_threshold

    def scan(self, text: str) -> InjectionResult:
        normalized, index_map = _normalize(text)
        matches: list[InjectionMatch] = []
        for category, pattern, weight in _RULES:
            for m in pattern.finditer(normalized):
                start = index_map[m.start()]
                end = index_map[m.end()]
                matches.append(
                    InjectionMatch(category=category, start=start, end=end, weight=weight)
                )
        matches.sort(key=lambda m: (m.start, m.end))

        # Hand-written score combination: 1 - product(1 - w).
        complement = 1.0
        for m in matches:
            complement *= 1.0 - m.weight
        score = round(1.0 - complement, 3)
        if score >= self.block_threshold:
            verdict = "block"
        elif score >= self.warn_threshold:
            verdict = "warn"
        else:
            verdict = "allow"
        return InjectionResult(score=score, verdict=verdict, matches=matches)
