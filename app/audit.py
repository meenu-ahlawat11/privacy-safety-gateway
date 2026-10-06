"""SQLite-backed audit trail for the gateway.

Privacy rule: this log NEVER accepts or stores raw prompt text, raw
sensitive values, or anything that could leak them. The public API only
accepts counts per type, action names, scores, verdicts, decisions,
override counts, and prompt lengths.
"""

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_VALID_DECISIONS = {"allowed", "modified", "blocked"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    profile TEXT,
    finding_counts TEXT,
    actions TEXT,
    injection_score REAL,
    injection_verdict TEXT,
    decision TEXT,
    overrides INTEGER,
    prompt_length INTEGER
)
"""

_COLUMNS = (
    "id, timestamp, profile, finding_counts, actions, injection_score, "
    "injection_verdict, decision, overrides, prompt_length"
)


class AuditLog:
    """Append-only audit log stored in a local SQLite database."""

    def __init__(self, db_path: str = "data/audit.db") -> None:
        self._db_path = str(db_path)
        parent = Path(self._db_path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_TABLE)

    def record(
        self,
        profile: str,
        finding_counts: dict[str, int],
        actions: dict[str, str],
        injection_score: float,
        injection_verdict: str,
        decision: str,
        overrides: int = 0,
        prompt_length: int = 0,
    ) -> int:
        """Insert an event and return its row id."""
        if decision not in _VALID_DECISIONS:
            raise ValueError(f"invalid decision: {decision!r}")
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO audit_events (timestamp, profile, finding_counts, "
                "actions, injection_score, injection_verdict, decision, "
                "overrides, prompt_length) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    profile,
                    json.dumps(finding_counts),
                    json.dumps(actions),
                    injection_score,
                    injection_verdict,
                    decision,
                    overrides,
                    prompt_length,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return events newest first, up to limit."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM audit_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        """Aggregate totals, per-type sums, decisions, blocked per day."""
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            finding_rows = conn.execute("SELECT finding_counts FROM audit_events").fetchall()
            decision_rows = conn.execute(
                "SELECT decision, COUNT(*) FROM audit_events GROUP BY decision"
            ).fetchall()
            blocked_rows = conn.execute(
                "SELECT substr(timestamp, 1, 10), COUNT(*) FROM audit_events "
                "WHERE decision = 'blocked' GROUP BY substr(timestamp, 1, 10)"
            ).fetchall()

        counts_per_type: Counter[str] = Counter()
        for (blob,) in finding_rows:
            for type_name, count in json.loads(blob).items():
                counts_per_type[type_name] += count

        return {
            "total_events": total,
            "counts_per_type": dict(counts_per_type),
            "decisions": {decision: count for decision, count in decision_rows},
            "blocked_per_day": {day: count for day, count in blocked_rows},
        }

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": row[0],
            "timestamp": row[1],
            "profile": row[2],
            "finding_counts": json.loads(row[3]),
            "actions": json.loads(row[4]),
            "injection_score": row[5],
            "injection_verdict": row[6],
            "decision": row[7],
            "overrides": row[8],
            "prompt_length": row[9],
        }
