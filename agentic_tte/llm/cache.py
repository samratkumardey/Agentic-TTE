"""SQLite response cache keyed by (model, digest, messages, schema, options).

EmulatRx used a JSON-file cache in ``LLMZoo``; SQLite is safe for concurrent run workers.
Including the model digest means a re-pulled model with new weights never reuses old answers.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
