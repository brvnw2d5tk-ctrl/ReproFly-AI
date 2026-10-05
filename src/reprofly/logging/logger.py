"""Append-only experiment logger backed by SQLite.

Design goals (Phase 0):
  * The ONLY public mutating API is append_*; there is no update/delete.
  * Append-only is additionally enforced INSIDE SQLite by BEFORE UPDATE /
    BEFORE DELETE triggers that RAISE(ABORT). Even code that bypasses the
    Python API cannot silently rewrite history.
  * WAL mode + busy_timeout for safe concurrent access from the MCP server
    and the test suite.
  * No LLM/model calls exist in Phase 0, so no LLM log rows are fabricated.

Tables: runs, agent_calls, tool_calls, experiments, protocol_events, errors.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    mode            TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS agent_calls (
    call_id    TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    timestamp  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    agent_role TEXT NOT NULL,
    purpose    TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    agent_role      TEXT NOT NULL,
    tool            TEXT NOT NULL,
    arguments       TEXT NOT NULL,          -- JSON
    result_status   TEXT NOT NULL,          -- OK | DENIED | ERROR | NOT_IMPLEMENTED
    duration_ms     REAL,
    protocol_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    name          TEXT NOT NULL,
    description   TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS protocol_events (
    event_id   TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    timestamp  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    event_type TEXT NOT NULL,               -- e.g. BUDGET_DECREASED, VIOLATION_LOGGED
    detail     TEXT NOT NULL                -- JSON
);

CREATE TABLE IF NOT EXISTS errors (
    error_id   TEXT PRIMARY KEY,
    run_id     TEXT,
    timestamp  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    source     TEXT NOT NULL,               -- module/function name
    error_type TEXT NOT NULL,
    message    TEXT NOT NULL,
    context    TEXT                          -- JSON, no secrets ever
);

-- ---- append-only enforcement: reject any UPDATE / DELETE ----------------
CREATE TRIGGER IF NOT EXISTS runs_no_update     BEFORE UPDATE ON runs     BEGIN SELECT RAISE(ABORT,'append-only: runs'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete     BEFORE DELETE ON runs     BEGIN SELECT RAISE(ABORT,'append-only: runs'); END;
CREATE TRIGGER IF NOT EXISTS agent_calls_no_update BEFORE UPDATE ON agent_calls BEGIN SELECT RAISE(ABORT,'append-only: agent_calls'); END;
CREATE TRIGGER IF NOT EXISTS agent_calls_no_delete BEFORE DELETE ON agent_calls BEGIN SELECT RAISE(ABORT,'append-only: agent_calls'); END;
CREATE TRIGGER IF NOT EXISTS tool_calls_no_update  BEFORE UPDATE ON tool_calls  BEGIN SELECT RAISE(ABORT,'append-only: tool_calls'); END;
CREATE TRIGGER IF NOT EXISTS tool_calls_no_delete  BEFORE DELETE ON tool_calls  BEGIN SELECT RAISE(ABORT,'append-only: tool_calls'); END;
CREATE TRIGGER IF NOT EXISTS experiments_no_update BEFORE UPDATE ON experiments BEGIN SELECT RAISE(ABORT,'append-only: experiments'); END;
CREATE TRIGGER IF NOT EXISTS experiments_no_delete BEFORE DELETE ON experiments BEGIN SELECT RAISE(ABORT,'append-only: experiments'); END;
CREATE TRIGGER IF NOT EXISTS protocol_events_no_update BEFORE UPDATE ON protocol_events BEGIN SELECT RAISE(ABORT,'append-only: protocol_events'); END;
CREATE TRIGGER IF NOT EXISTS protocol_events_no_delete BEFORE DELETE ON protocol_events BEGIN SELECT RAISE(ABORT,'append-only: protocol_events'); END;
CREATE TRIGGER IF NOT EXISTS errors_no_update  BEFORE UPDATE ON errors  BEGIN SELECT RAISE(ABORT,'append-only: errors'); END;
CREATE TRIGGER IF NOT EXISTS errors_no_delete  BEFORE DELETE ON errors  BEGIN SELECT RAISE(ABORT,'append-only: errors'); END;
"""

# Tables covered by append-only triggers (kept in sync with _SCHEMA).
APPEND_ONLY_TABLES = (
    "runs", "agent_calls", "tool_calls",
    "experiments", "protocol_events", "errors",
)


class ExperimentLogger:
    """Thread-safe append-only logger. One instance per process."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _append(self, sql: str, params: tuple) -> str:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    # -- append API ----------------------------------------------------------

    def start_run(
        self,
        experiment_id: str,
        mode: str,
        protocol_version: str,
        dataset_version: str,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self._append(
            "INSERT INTO runs (run_id, experiment_id, mode, protocol_version, dataset_version) "
            "VALUES (?,?,?,?,?)",
            (run_id, experiment_id, mode, protocol_version, dataset_version),
        )
        return run_id

    def finish_run(self, run_id: str) -> None:
        # append-only: 'finished_at' is set via a dedicated UPDATE-free path —
        # we INSERT a protocol event instead of mutating the run row.
        self.log_protocol_event(
            run_id, "RUN_FINISHED", {"run_id": run_id}
        )

    def log_agent_call(
        self, run_id: str, agent_role: str, purpose: str | None = None
    ) -> str:
        call_id = f"ac_{uuid.uuid4().hex[:12]}"
        self._append(
            "INSERT INTO agent_calls (call_id, run_id, agent_role, purpose) VALUES (?,?,?,?)",
            (call_id, run_id, agent_role, purpose),
        )
        return call_id

    def log_tool_call(
        self,
        run_id: str,
        agent_role: str,
        tool: str,
        arguments: dict[str, Any],
        result_status: str,
        duration_ms: float,
        protocol_version: str,
    ) -> str:
        call_id = f"tc_{uuid.uuid4().hex[:12]}"
        self._append(
            "INSERT INTO tool_calls (call_id, run_id, agent_role, tool, arguments, "
            "result_status, duration_ms, protocol_version) VALUES (?,?,?,?,?,?,?,?)",
            (
                call_id,
                run_id,
                agent_role,
                tool,
                json.dumps(arguments, ensure_ascii=False, default=str),
                result_status,
                float(duration_ms),
                protocol_version,
            ),
        )
        return call_id

    def log_experiment(
        self, experiment_id: str, run_id: str, name: str, description: str = ""
    ) -> str:
        self._append(
            "INSERT INTO experiments (experiment_id, run_id, name, description) VALUES (?,?,?,?)",
            (experiment_id, run_id, name, description),
        )
        return experiment_id

    def log_protocol_event(
        self, run_id: str, event_type: str, detail: dict[str, Any]
    ) -> str:
        event_id = f"pe_{uuid.uuid4().hex[:12]}"
        self._append(
            "INSERT INTO protocol_events (event_id, run_id, event_type, detail) VALUES (?,?,?,?)",
            (event_id, run_id, event_type, json.dumps(detail, ensure_ascii=False, default=str)),
        )
        return event_id

    def log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        error_id = f"er_{uuid.uuid4().hex[:12]}"
        self._append(
            "INSERT INTO errors (error_id, run_id, source, error_type, message, context) "
            "VALUES (?,?,?,?,?,?)",
            (
                error_id,
                run_id,
                source,
                error_type,
                message,
                json.dumps(context or {}, ensure_ascii=False, default=str),
            ),
        )
        return error_id

    # -- read-only queries (for tests / audits) ------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Arbitrary SELECT for auditing. Mutating statements are rejected."""
        low = sql.strip().lower()
        if not low.startswith("select"):
            raise PermissionError("ExperimentLogger.query() only permits SELECT")
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def tool_calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY timestamp", (run_id,)
        )

    def protocol_events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM protocol_events WHERE run_id = ? ORDER BY timestamp", (run_id,)
        )


__all__ = ["ExperimentLogger", "APPEND_ONLY_TABLES"]
