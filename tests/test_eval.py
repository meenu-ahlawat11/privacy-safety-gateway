"""Tests for the evaluation helpers in tests/eval.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detectors.base import Detection
from tests.eval import (
    evaluate,
    find_errors,
    load_dataset,
    pii_counts,
    precision_recall_f1,
)

DATASET = Path(__file__).resolve().parent / "dataset.json"


def test_precision_recall_f1_known_values() -> None:
    p, r, f1 = precision_recall_f1(8, 2, 2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)

    p, r, f1 = precision_recall_f1(5, 0, 5)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)


def test_precision_recall_f1_zero_division() -> None:
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 3, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 0, 4) == (0.0, 0.0, 0.0)


def test_pii_counts_known_tp_fp_fn() -> None:
    text = "email bob@example.com then 4111111111111111 end"
    labels = [
        {"type": "EMAIL", "value": "bob@example.com"},
        {"type": "PHONE", "value": "9876543210"},
    ]
    detections = [
        Detection(6, 21, "EMAIL", 0.9, "test"),
        Detection(27, 43, "CREDIT_CARD", 0.99, "test"),
        Detection(6, 21, "API_KEY", 0.8, "test"),
    ]
    counts = pii_counts(text, labels, detections)
    assert counts["EMAIL"] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts["PHONE"] == {"tp": 0, "fp": 0, "fn": 1}
    assert counts["CREDIT_CARD"] == {"tp": 0, "fp": 1, "fn": 0}
    assert counts["API_KEY"] == {"tp": 0, "fp": 1, "fn": 0}


def test_find_errors_reports_only_types_and_ids() -> None:
    text = "email bob@example.com end"
    labels = [
        {"type": "EMAIL", "value": "bob@example.com"},
        {"type": "PHONE", "value": "9876543210"},
    ]
    detections = [Detection(0, 5, "EMAIL", 0.9, "test")]
    errors = find_errors(labels, text, detections)
    assert errors["missed"] == ["EMAIL", "PHONE"]
    assert errors["false_positives"] == ["EMAIL"]
    dumped = json.dumps(errors)
    assert "bob@example.com" not in dumped
    assert "9876543210" not in dumped


def test_dataset_schema_and_label_values() -> None:
    data = load_dataset(DATASET)
    assert len(data) >= 120
    valid_categories = {"pii", "injection", "clean", "hard_negative"}
    for item in data:
        assert set(item) == {"id", "text", "category", "labels"}
        assert item["category"] in valid_categories
        assert isinstance(item["id"], int)
        assert isinstance(item["text"], str) and item["text"]
        assert isinstance(item["labels"], list)
        for label in item["labels"]:
            assert set(label) == {"type", "value"}
        if item["category"] == "pii":
            assert item["labels"], item
            for label in item["labels"]:
                assert label["value"] in item["text"]
        elif item["category"] == "injection":
            assert item["labels"] == [{"type": "INJECTION", "value": ""}]
        else:
            assert item["labels"] == []


def test_results_contain_no_raw_values() -> None:
    dataset = load_dataset(DATASET)
    results = evaluate(dataset)
    dumped = json.dumps(results)
    for item in dataset:
        assert item["text"] not in dumped
        for label in item["labels"]:
            if label["value"]:
                assert label["value"] not in dumped
