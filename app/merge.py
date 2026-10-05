"""Interval merging for overlapping detections."""

from app.detectors.base import Detection


def merge_detections(detections: list[Detection]) -> list[Detection]:
    """Merge overlapping detections, keeping one winner per overlapping group.

    Algorithm (sort-and-sweep):
      1. Sort detections by start offset.
      2. Sweep left to right, grouping detections where each detection's
         start is strictly less than the running maximum end of the current
         group. Since `end` is exclusive, spans that merely touch (one ends
         exactly where the next starts) do NOT overlap and end the group.
      3. From each group pick exactly one winner: highest confidence, then
         longest span, then earliest start.
      4. Return the winners sorted by start.

    Time complexity: O(n log n) for the sort plus O(n) for the sweep and
    winner selection, i.e. O(n log n) overall. Space: O(n).

    The input list is not mutated.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.start)

    groups: list[list[Detection]] = []
    current: list[Detection] = [ordered[0]]
    current_max_end = ordered[0].end

    for detection in ordered[1:]:
        if detection.start < current_max_end:
            current.append(detection)
            current_max_end = max(current_max_end, detection.end)
        else:
            groups.append(current)
            current = [detection]
            current_max_end = detection.end
    groups.append(current)

    winners = [
        max(group, key=lambda d: (d.confidence, d.end - d.start, -d.start))
        for group in groups
    ]
    winners.sort(key=lambda d: d.start)
    return winners
