"""End-to-end pipeline: detect -> merge -> policy -> protect -> audit fields."""

from dataclasses import dataclass, field

from app.detectors.base import Detection, Detector
from app.detectors.keyword_trie import KeywordDetector
from app.detectors.regex_det import RegexDetector
from app.merge import merge_detections
from app.policy import Policy
from app.protect import protect
from app.vault import Vault


@dataclass(frozen=True)
class PipelineResult:
    """Result of Pipeline.process. Contains no raw sensitive values."""

    protected_text: str
    blocked: bool
    blocked_types: list[str] = field(default_factory=list)
    findings: list[tuple[str, int, int, float]] = field(default_factory=list)
    applied: list[tuple[str, str]] = field(default_factory=list)
    profile: str = ""


class Pipeline:
    def __init__(self, policy: Policy, detectors: list[Detector]) -> None:
        self._policy = policy
        self._detectors = detectors

    @classmethod
    def default(cls, keywords: list[str] | None = None) -> "Pipeline":
        detectors: list[Detector] = [RegexDetector()]
        if keywords:
            detectors.append(KeywordDetector(keywords))
        return cls(Policy.load(), detectors)

    def analyze(self, text: str, profile: str | None = None) -> list[Detection]:
        """Run all detectors and merge overlaps. Dry-run: text unchanged."""
        detections: list[Detection] = []
        for detector in self._detectors:
            detections.extend(detector.detect(text))
        return merge_detections(detections)

    def _resolve_profile(self, profile: str | None) -> str:
        if profile is not None:
            return profile
        # action_for(None) has already validated that a default exists.
        return self._policy._default_profile

    def process(self, text: str, profile: str | None = None) -> tuple[PipelineResult, Vault]:
        """Full pipeline. Each call uses a fresh Vault (tokens restart at 1)."""
        vault = Vault()
        detections = self.analyze(text, profile)
        protect_result = protect(
            text, detections, self._policy.action_for(profile), vault
        )
        findings = [(d.type, d.start, d.end, d.confidence) for d in detections]
        result = PipelineResult(
            protected_text=protect_result.text,
            blocked=protect_result.blocked,
            blocked_types=protect_result.blocked_types,
            findings=findings,
            applied=protect_result.applied,
            profile=self._resolve_profile(profile),
        )
        return result, vault

    def restore(self, vault: Vault, text: str) -> str:
        return vault.restore(text)
