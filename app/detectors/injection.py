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
            r"(?:ignore|disregard|forget|override|do\s+not\s+follow|"
            r"stop\s+following|set\s+aside|replace|supersede|discard|"
            r"drop|throw\s+away)\s+"
            r"(?:[\w,'-]+\s+){0,6}?"
            r"(?:instructions?|rules?|constraints?|guidelines?|guardrails?|"
            r"system\s+message|system\s+prompt|safety\s+rules?|task\s+list|"
            r"directives?|setup|programming|prompt)\b"
            r"(?!\s+(?:section|chapter|paragraph|part|document|report)\b)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "instruction_override",
        re.compile(r"do\s+exactly\s+what\s+I\s+say", re.IGNORECASE),
        0.8,
    ),
    (
        "instruction_override",
        re.compile(
            r"(?:system\s+update|new\s+priority|priority\s+message|admin\s+notice)"
            r"[^.?\n]{0,60}?"
            r"(?:your|the)\s+(?:earlier|previous|prior|old)\s+"
            r"(?:guidelines?|instructions?|rules?)"
            r"[^.?\n]{0,60}?"
            r"(?:deprecated|replaced?|void|obsolete|overridden)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:reveal|show|print|repeat|output|leak|dump|list|display|"
            r"tell\s+me)\s+"
            r"(?:\w+\s+){0,4}?"
            r"(?:system\s+prompt|hidden\s+(?:prompt|instructions?)|"
            r"(?:internal|full|exact)\s+instructions?|configuration|"
            r"instructions?\s+you\s+were\s+given)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:reveal|show|output|leak|dump|list|display|tell\s+me)\s+"
            r"(?:\w+\s+){0,4}?(?:the\s+)?(?:text|everything)\s+above",
            re.IGNORECASE,
        ),
        0.8,
    ),
    # Bare "repeat/print the text above" is a weaker signal: warn only.
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:repeat|print)\s+(?:\w+\s+){0,4}?(?:the\s+)?"
            r"(?:text|everything)\s+above",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"what\s+(?:were|are|was|is)[^.?\n]{0,50}?"
            r"(?:system\s+prompt|hidden\s+configuration)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"what\s+(?:were|are)\s+(?:the\s+|your\s+)?(?:exact\s+)?"
            r"(?:instructions?|rules?|configuration)\b"
            r"(?![^.?\n]{0,60}?(?:system\s+prompt|hidden\s+configuration))"
            r"(?=[^.?\n]{0,40}?\b(?:your|you|hidden|secret|system)\b)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:summari[sz]e|repeat|copy)[\s,]+(?:[\w,'-]+\s+){0,8}?"
            r"(?:word\s+for\s+word[\s,]+|verbatim[\s,]+)?"
            r"(?:[\w,'-]+\s+){0,8}?"
            r"(?:text|block|everything)\s+(?:immediately\s+)?"
            r"(?:above|before)(?:\s+where)?\s+(?:my|this|the)\s+"
            r"(?:messages?|conversations?)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:reveal|print|show)\s+your\s+(?:internal|hidden|secret)\s+"
            r"(?:setup|configuration|text|prompt)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"(?:act\s+as|adopt|role[\s-]?play\s+as|simulate|switch\s+into|"
            r"pretend\s+to\s+be)\s+(?:an?\s+)?"
            r"(?:dan|stan|evil\s+twin|developer\s+mode|jailbroken\s+model|"
            r"unrestricted\s+(?:ai|assistant|model)|"
            r"ai\s+with\s+no\s+(?:rules?|guidelines?|content\s+policy|"
            r"restrictions?|ethics))\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"(?:you\s+are\s+now\s+dan|developer\s+mode\s+enabled)",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"(?:pretend|imagine|act|role[\s-]?play|respond)\s+"
            r"(?:you\s+are|your\s+name\s+is|as)\b"
            r"[^.?\n]{0,40}?"
            r"(?:no\s+restrictions?|never\s+refuses?|always\s+does\s+the\s+opposite|"
            r"ignores?\s+every\s+(?:ethical\s+)?rules?|"
            r"no\s+(?:rules?|ethics|limits?))",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"you\s+are\s+now\s+['\"]?\w[^.?\n]{0,40}?"
            r"(?:rogue|unrestricted|opposite|no\s+(?:rules?|ethics|limits?)|"
            r"ignores?\s+every)",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:bypass|disable|turn\s+off|ignore|circumvent)\s+"
            r"(?:your\s+|the\s+|our\s+)?"
            r"(?:safety\s+(?:layer|checks?|filters?)|content\s+filters?|"
            r"safeguards?|guardrails?|restrictions?)|"
            r"without\s+(?:any\s+)?(?:restrictions?|guidelines?)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:your\s+|the\s+)?"
            r"(?:safety\s+filters?|content\s+moderation|content\s+policy|"
            r"guardrails?|restrictions?)\s+"
            r"(?:are|is|have\s+been|has\s+been)\s+(?:now\s+)?"
            r"(?:disabled|off|removed|lifted|suspended)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:your\s+|the\s+)?"
            r"(?:content\s+moderation|safety\s+filters?|restrictions?)\s+"
            r"(?:doesn't|does\s+not|don't)\s+exist",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:temporarily\s+)?bypass\s+your\s+"
            r"(?:content\s+policy|safety(?!\s+filters?)|filters?|guidelines?)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(
            r"(?:leak|send|forward|post)\s+(?:the\s+|your\s+|my\s+)?"
            r"(?:data|conversation|secrets?|api\s*key|credentials?)"
            r"(?:\s+to\s+(?:a\s+|an\s+|the\s+|my\s+)?"
            r"(?:url|email|server|webhook))?",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(r"\bexfiltrate\b", re.IGNORECASE),
        0.5,
    ),
    (
        "encoding_evasion",
        re.compile(
            r"(?:encode|encrypt|disguise|hide)\s+(?:your\s+)?"
            r"(?:answer|response|output)\s+(?:\w+\s+){0,6}?"
            r"(?:slips?|gets?)\s+past\s+(?:the\s+)?(?:safety|filters?|checks?)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(
            r"(?:send|forward|mail|email|upload|post|transmit)\s+"
            r"(?:\w+\s+){0,12}?"
            r"(?:everything|all|any|the|our|my)\s+"
            r"(?:\w+\s+){0,12}?"
            r"(?:sensitive|private|confidential|chat\s+history|conversation|"
            r"details|data)\s+"
            r"(?:\w+\s+){0,12}?"
            r"(?:to|at)\s+(?:https?://\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(
            r"compose\s+an?\s+email\s+containing\s+(?:the\s+|all\s+|everything\s+)?"
            r"(?:\w+\s+){0,4}?"
            r"(?:chat\s+history|conversation|details|data)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "data_exfiltration",
        re.compile(
            r"include\s+(?:the\s+)?(?:secrets?|api\s*key|credentials?)\s+"
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
