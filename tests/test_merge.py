"""Tests for merge_detections. Synthetic data only."""

from app.detectors.base import Detection
from app.merge import merge_detections


def _d(start: int, end: int, confidence: float = 0.85, type: str = "TEST") -> Detection:
    return Detection(start=start, end=end, type=type, confidence=confidence, source="test")


def test_empty_list() -> None:
    assert merge_detections([]) == []


def test_single_detection() -> None:
    d = _d(3, 7)
    assert merge_detections([d]) == [d]


def test_non_overlapping_all_kept() -> None:
    a, b, c = _d(0, 5), _d(10, 15), _d(20, 25)
    assert merge_detections([a, b, c]) == [a, b, c]


def test_touching_spans_both_kept() -> None:
    a, b = _d(0, 5), _d(5, 10)
    assert merge_detections([a, b]) == [a, b]


def test_overlap_higher_confidence_wins() -> None:
    low = _d(0, 10, confidence=0.85)
    high = _d(5, 8, confidence=0.99)
    assert merge_detections([low, high]) == [high]


def test_equal_confidence_longer_span_wins() -> None:
    short = _d(2, 6, confidence=0.85)
    long = _d(0, 10, confidence=0.85)
    assert merge_detections([short, long]) == [long]


def test_chain_a_b_c_one_winner() -> None:
    a = _d(0, 10, confidence=0.85)
    b = _d(5, 15, confidence=0.99)
    c = _d(12, 20, confidence=0.85)  # overlaps B but not A
    assert merge_detections([a, b, c]) == [b]


def test_fully_nested_span() -> None:
    outer = _d(0, 20, confidence=0.85)
    inner = _d(5, 10, confidence=0.99)
    assert merge_detections([outer, inner]) == [inner]


def test_unsorted_input() -> None:
    a, b, c = _d(20, 25), _d(0, 5), _d(10, 15)
    assert merge_detections([a, b, c]) == [b, c, a]


def test_input_list_not_mutated() -> None:
    a, b, c = _d(10, 15), _d(0, 5), _d(5, 7)
    original = [a, b, c]
    snapshot = list(original)
    merge_detections(original)
    assert original == snapshot
