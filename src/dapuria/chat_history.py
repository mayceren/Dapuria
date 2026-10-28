"""Local, scoped conversation storage; model/index data never enters this store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_KEYS = ("messages", "phase", "pending", "latest_result", "inventory", "excluded_ingredients")


class ChatHistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    owner_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner_id, id)
                )
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "pinned" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, owner_id: str, conversation_id: str, state: dict[str, Any]) -> None:
        messages = state.get("messages", [])
        if not messages:
            return
        snapshot = {key: state.get(key) for key in STATE_KEYS}
        if snapshot["inventory"] is None:
            snapshot["inventory"] = self._legacy_inventory(snapshot)
        if snapshot["excluded_ingredients"] is None:
            snapshot["excluded_ingredients"] = []
        payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        first_query = next((message["content"] for message in messages if message["role"] == "user"), "Percakapan")
        title = " ".join(first_query.split())[:72]
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO conversations (owner_id, id, title, created_at, updated_at, snapshot)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, id) DO UPDATE SET
                    updated_at = excluded.updated_at, snapshot = excluded.snapshot
            """, (owner_id, conversation_id, title, now, now, payload))

    def list(self, owner_id: str, search: str = "") -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT id, title, created_at, updated_at, pinned FROM conversations
                WHERE owner_id = ? ORDER BY pinned DESC, updated_at DESC, id
            """, (owner_id,)).fetchall()
        query = search.strip().casefold()
        return [{**dict(row), "pinned": bool(row["pinned"])}
                for row in rows if query in row["title"].casefold()]

    def load(self, owner_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot FROM conversations WHERE owner_id = ? AND id = ?",
                (owner_id, conversation_id),
            ).fetchone()
        if row is None:
            return None
        snapshot = json.loads(row["snapshot"])
        if isinstance(snapshot, dict) and "inventory" not in snapshot:
            snapshot["inventory"] = self._legacy_inventory(snapshot)
        if isinstance(snapshot, dict):
            snapshot.setdefault("excluded_ingredients", [])
            if snapshot.get("phase") == "review_ingredients":
                snapshot["phase"] = "await_query"
                snapshot["latest_result"] = None
        if (
            not isinstance(snapshot, dict)
            or not all(key in snapshot for key in STATE_KEYS)
            or not isinstance(snapshot.get("messages"), list)
            or snapshot.get("phase") not in {"await_query", "await_confirmation"}
            or not isinstance(snapshot.get("inventory"), list)
            or not all(isinstance(item, str) for item in snapshot["inventory"])
            or not isinstance(snapshot.get("excluded_ingredients"), list)
            or not all(isinstance(item, str) for item in snapshot["excluded_ingredients"])
        ):
            raise ValueError("Format riwayat percakapan tidak didukung.")
        if snapshot["phase"] == "await_confirmation" and not isinstance(snapshot.get("pending"), dict):
            raise ValueError("Data klarifikasi pada riwayat tidak lengkap.")
        return snapshot

    @staticmethod
    def _legacy_inventory(snapshot: dict[str, Any]) -> list[str]:
        result = snapshot.get("latest_result") or (snapshot.get("pending") or {}).get("initial_result") or {}
        return list(result.get("user_ingredients", []))

    def rename(self, owner_id: str, conversation_id: str, title: str) -> None:
        title = " ".join(title.split())[:72]
        if not title:
            raise ValueError("Judul percakapan tidak boleh kosong.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ? WHERE owner_id = ? AND id = ?",
                (title, owner_id, conversation_id),
            )

    def delete(self, owner_id: str, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE owner_id = ? AND id = ?",
                (owner_id, conversation_id),
            )

    def set_pinned(self, owner_id: str, conversation_id: str, pinned: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET pinned = ? WHERE owner_id = ? AND id = ?",
                (int(pinned), owner_id, conversation_id),
            )
