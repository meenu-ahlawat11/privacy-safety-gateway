"""Evaluate PII detectors and the InjectionScanner against tests/dataset.json.

Outputs contain only counts, types, rates, latencies and prompt ids.
Raw prompt text and raw detected values are never printed or saved.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.detectors.base import Detection
from app.detectors.injection import InjectionScanner
from app.detectors.keyword_trie import KeywordDetector
from app.detectors.regex_det import RegexDetector
from app.merge import merge_detections

DATASET_PATH = Path("tests/dataset.json")
CSV_OUT = Path("benchmarks/eval_results.csv")
JSON_OUT = Path("benchmarks/eval_results.json")
POSITIVE_VERDICTS = {"warn", "block"}
CATEGORIES = ("pii", "injection", "clean", "hard_negative")


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


KEYWORD_TERMS: list[str] = sorted(
    {
        label["value"]
        for item in load_dataset()
        for label in item["labels"]
        if label["type"] == "KEYWORD"
    }
)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, f1); any zero division yields 0.0."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def pii_counts(
    text: str, labels: list[dict[str, str]], detections: list[Detection]
) -> dict[str, dict[str, int]]:
    """Match detections to labels by (type, exact value). Returns per-type TP/FP/FN."""
    label_multiset = Counter((lab["type"], lab["value"]) for lab in labels)
    detection_multiset = Counter((d.type, text[d.start : d.end]) for d in detections)
    det_per_type: Counter[str] = Counter(d.type for d in detections)
    lab_per_type: Counter[str] = Counter(lab["type"] for lab in labels)
    tp_per_type: Counter[str] = Counter()
    for key in label_multiset:
        matched = min(label_multiset[key], detection_multiset.get(key, 0))
        if matched:
            tp_per_type[key[0]] += matched
    out: dict[str, dict[str, int]] = {}
    for t in set(det_per_type) | set(lab_per_type):
        tp = tp_per_type.get(t, 0)
        out[t] = {
            "tp": tp,
            "fp": det_per_type.get(t, 0) - tp,
            "fn": lab_per_type.get(t, 0) - tp,
        }
    return out


def find_errors(
    labels: list[dict[str, str]], text: str, detections: list[Detection]
) -> dict[str, list[str]]:
    """Return {'missed': [types...], 'false_positives': [types...]} (values never shown)."""
    label_multiset = Counter((lab["type"], lab["value"]) for lab in labels)
    detection_multiset = Counter((d.type, text[d.start : d.end]) for d in detections)
    missed: list[str] = []
    for key, count in label_multiset.items():
        gap = count - min(count, detection_multiset.get(key, 0))
        missed.extend([key[0]] * gap)
    extra: list[str] = []
    for key, count in detection_multiset.items():
        gap = count - min(count, label_multiset.get(key, 0))
        extra.extend([key[0]] * gap)
    return {"missed": missed, "false_positives": extra}


def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    regex = RegexDetector()
    has_keywords = any(
        lab["type"] == "KEYWORD" for item in dataset for lab in item["labels"]
    )
    keywords = KeywordDetector(KEYWORD_TERMS) if has_keywords else None
    scanner = InjectionScanner()

    pii_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"missed": [], "false_positives": []}
    )
    latencies: list[float] = []

    inj_tp = inj_fp = inj_fn = inj_tn = 0
    n_clean = flagged_clean = 0
    n_hard = flagged_hard = 0
    cat_counts: Counter[str] = Counter()

    for item in dataset:
        text = item["text"]
        cat_counts[item["category"]] += 1
        start = time.perf_counter()
        detections = regex.detect(text)
        if keywords is not None:
            detections = detections + keywords.detect(text)
        detections = merge_detections(detections)
        result = scanner.scan(text)
        latencies.append((time.perf_counter() - start) * 1000.0)

        if item["category"] == "pii":
            for det_type, counts in pii_counts(text, item["labels"], detections).items():
                for key in ("tp", "fp", "fn"):
                    pii_totals[det_type][key] += counts[key]
            item_errors = find_errors(item["labels"], text, detections)
            for det_type in item_errors["missed"]:
                errors[det_type]["missed"].append(item["id"])
            for det_type in item_errors["false_positives"]:
                errors[det_type]["false_positives"].append(item["id"])

        predicted_injection = result.verdict in POSITIVE_VERDICTS
        truth_injection = item["category"] == "injection"
        if predicted_injection and truth_injection:
            inj_tp += 1
        elif predicted_injection and not truth_injection:
            inj_fp += 1
        elif not predicted_injection and truth_injection:
            inj_fn += 1
        else:
            inj_tn += 1
        if item["category"] == "clean":
            n_clean += 1
            flagged_clean += int(predicted_injection)
        elif item["category"] == "hard_negative":
            n_hard += 1
            flagged_hard += int(predicted_injection)

    per_type: dict[str, dict[str, Any]] = {}
    sum_tp = sum_fp = sum_fn = 0
    for det_type in sorted(pii_totals):
        c = pii_totals[det_type]
        p, r, f1 = precision_recall_f1(c["tp"], c["fp"], c["fn"])
        per_type[det_type] = {**c, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        sum_tp += c["tp"]
        sum_fp += c["fp"]
        sum_fn += c["fn"]
    op, orec, of1 = precision_recall_f1(sum_tp, sum_fp, sum_fn)

    ip, irec, if1 = precision_recall_f1(inj_tp, inj_fp, inj_fn)
    sorted_lat = sorted(latencies)
    p95_index = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    return {
        "counts": {cat: cat_counts.get(cat, 0) for cat in CATEGORIES},
        "pii": {
            "overall": {
                "tp": sum_tp,
                "fp": sum_fp,
                "fn": sum_fn,
                "precision": round(op, 4),
                "recall": round(orec, 4),
                "f1": round(of1, 4),
            },
            "per_type": per_type,
        },
        "injection": {
            "tp": inj_tp,
            "fp": inj_fp,
            "fn": inj_fn,
            "tn": inj_tn,
            "precision": round(ip, 4),
            "recall": round(irec, 4),
            "f1": round(if1, 4),
            "fpr_clean": round(flagged_clean / n_clean, 4) if n_clean else 0.0,
            "fpr_hard_negative": round(flagged_hard / n_hard, 4) if n_hard else 0.0,
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95": round(sorted_lat[p95_index], 3) if latencies else 0.0,
        },
        "errors": {t: dict(errors[t]) for t in sorted(errors)},
    }


def print_table(results: dict[str, Any]) -> None:
    print("PII per-type (tp/fp/fn, precision, recall, f1):")
    for det_type, m in results["pii"]["per_type"].items():
        print(
            f"  {det_type:<12} tp={m['tp']:<3} fp={m['fp']:<3} fn={m['fn']:<3} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        )
    o = results["pii"]["overall"]
    print(
        f"  OVERALL      tp={o['tp']:<3} fp={o['fp']:<3} fn={o['fn']:<3} "
        f"P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}"
    )
    i = results["injection"]
    print(
        f"Injection: tp={i['tp']} fp={i['fp']} fn={i['fn']} tn={i['tn']} "
        f"P={i['precision']:.3f} R={i['recall']:.3f} F1={i['f1']:.3f} "
        f"FPR(clean)={i['fpr_clean']:.3f} FPR(hard_negative)={i['fpr_hard_negative']:.3f}"
    )
    print(
        f"Latency per prompt: mean={results['latency_ms']['mean']:.3f} ms, "
        f"p95={results['latency_ms']['p95']:.3f} ms"
    )
    if results["errors"]:
        print("Errors (prompt ids only, by type):")
        for det_type, info in results["errors"].items():
            print(
                f"  {det_type}: missed ids={info['missed']} "
                f"false-positive ids={info['false_positives']}"
            )


def save_results(results: dict[str, Any]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    rows: list[list[Any]] = []
    for det_type, m in results["pii"]["per_type"].items():
        rows.append(["pii", det_type, m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"]])
    o = results["pii"]["overall"]
    rows.append(["pii", "overall", o["tp"], o["fp"], o["fn"], o["precision"], o["recall"], o["f1"]])
    i = results["injection"]
    rows.append(["injection", "overall", i["tp"], i["fp"], i["fn"], i["precision"], i["recall"], i["f1"]])
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "type", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writerows(rows)
        writer.writerow(["fpr", "clean", "", "", "", "", "", i["fpr_clean"]])
        writer.writerow(["fpr", "hard_negative", "", "", "", "", "", i["fpr_hard_negative"]])
        writer.writerow(["latency_ms", "mean", "", "", "", results["latency_ms"]["mean"], "", ""])
        writer.writerow(["latency_ms", "p95", "", "", "", results["latency_ms"]["p95"], "", ""])


def main() -> None:
    dataset = load_dataset()
    results = evaluate(dataset)
    print_table(results)
    save_results(results)
    print(f"Wrote {CSV_OUT} and {JSON_OUT}")


if __name__ == "__main__":
    main()
