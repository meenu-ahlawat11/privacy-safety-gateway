"""Shared base types for detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A single detected span in text.

    start: inclusive character offset.
    end: exclusive character offset.
    type: category of the detection (e.g. "credit_card").
    confidence: score between 0.0 and 1.0.
    source: name of the detector/source that produced it.
    """

    start: int
    end: int
    type: str
    confidence: float
    source: str


class Detector(ABC):
    """Abstract base class for all detectors."""

    @abstractmethod
    def detect(self, text: str) -> list[Detection]:
        """Return all detections found in the given text."""
        ...
