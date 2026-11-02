"""Per-session view of IndexedDB history, with acknowledged, retryable writes."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from .history_format import STATE_KEYS, validate_record, validate_snapshot


class BrowserHistoryStore:
    def __init__(self):
        self.client_id = str(uuid4())
        self.ready = False
        self.refreshed = False
        self.error: str | None = None
        self.notice: str | None = None
        self.redirects: dict[str, str] = {}
        self._records: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._request: dict | None = None
        self._refresh = False

    @property
    def syncing(self) -> bool:
        return bool(self._pending or self._request or not self.ready)

    def request(self) -> dict | None:
        if self.error:
            return None
        if self._request is None:
            if not self.ready or (self._refresh and not self._pending):
                self._request = {"id": str(uuid4()), "kind": "read"}
                self._refresh = False
            elif self._pending:
                self._request = {"id": str(uuid4()), "kind": "apply", "operations": deepcopy(self._pending)}
        return deepcopy(self._request)

    def retry(self) -> None:
        self.error = None
        self._request = None

    def refresh(self) -> None:
        self._refresh = True

    def receive(self, response: dict | None) -> bool:
        if not response or not self._request or response.get("id") != self._request["id"]:
            return False
        if response.get("error"):
            if response["error"] == "conflict" and isinstance(response.get("records"), list):
                self._keep_conflicting_edits(response["records"])
                return True
            self.error = (
                "Riwayat belum tersimpan di browser. Coba simpan lagi."
                if self.ready else "Riwayat browser belum dapat dibuka. Izinkan penyimpanan situs, lalu coba lagi."
            )
            return True
        records = [validate_record(record, tombstones=True) for record in response.get("records", [])]
        self.refreshed = self.refreshed or (self.ready and self._request["kind"] == "read")
        applied = {operation["id"] for operation in self._request.get("operations", [])}
        remaining = [operation for operation in self._pending if operation["id"] not in applied]
        self._records = {record["id"]: record for record in records}
        self._pending = remaining
        for operation in remaining:
            self._project(operation)
        self.ready = True
        self.error = None
        self._request = None
        return True

    def _keep_conflicting_edits(self, values: list[dict]) -> None:
        records = [validate_record(record, tombstones=True) for record in values]
        changed = {operation["conversation_id"] for operation in self._pending if operation["kind"] == "put"}
        copies = [deepcopy(self._records[key]) for key in changed if not self._records[key].get("deleted")]
        self._records = {record["id"]: record for record in records}
        self._pending = []
        self._request = None
        self.error = None
        self.ready = True
        for record in copies:
            original_id = record["id"]
            record["id"] = str(uuid4())
            self.redirects[original_id] = record["id"]
            record["title"] = record["title"][:61] + " (salinan)"
            self._put(record)
        self.notice = (
            "Ada perubahan dari tab lain. Perubahan percakapanmu disimpan sebagai salinan agar keduanya tetap ada."
            if copies else "Riwayat diperbarui dari tab lain. Pilih ulang percakapan yang ingin dihapus."
        )

    def _project(self, operation: dict) -> None:
        record = deepcopy(operation.get("record", {"id": operation["conversation_id"], "deleted": True}))
        record["revision"] = operation["expected_revision"] + 1
        self._records[record["id"]] = record

    def _put(self, value: dict) -> None:
        record = validate_record(value)
        current = self._records.get(record["id"], {})
        operation = {"id": str(uuid4()), "kind": "put", "conversation_id": record["id"],
                     "expected_revision": current.get("revision", 0), "record": record}
        self._pending.append(operation)
        self._project(operation)

    def save(self, owner_id: str, conversation_id: str, state: dict) -> None:
        if not state.get("messages"):
            return
        snapshot = validate_snapshot({key: state.get(key) for key in STATE_KEYS})
        current = self._records.get(conversation_id)
        if current and not current.get("deleted") and current["snapshot"] == snapshot:
            return
        now = datetime.now(timezone.utc).isoformat()
        first_query = next((message["content"] for message in snapshot["messages"] if message["role"] == "user"), "Percakapan")
        record = current if current and not current.get("deleted") else {
            "id": conversation_id, "title": " ".join(first_query.split())[:72] or "Percakapan",
            "created_at": now, "pinned": False,
        }
        self._put({**record, "snapshot": snapshot, "updated_at": now})

    def list(self, owner_id: str, search: str = "") -> list[dict]:
        query = search.strip().casefold()
        records = [record for record in self._records.values()
                   if not record.get("deleted") and query in record["title"].casefold()]
        records.sort(key=lambda record: (record["updated_at"], record["id"]), reverse=True)
        records.sort(key=lambda record: not record["pinned"])
        return [{key: record[key] for key in ("id", "title", "created_at", "updated_at", "pinned")} for record in records]

    def load(self, owner_id: str, conversation_id: str) -> dict | None:
        record = self._records.get(conversation_id)
        return deepcopy(record["snapshot"]) if record and not record.get("deleted") else None

    def rename(self, owner_id: str, conversation_id: str, title: str) -> None:
        title = " ".join(title.split())[:72]
        if not title:
            raise ValueError("Judul percakapan tidak boleh kosong.")
        self._put({**self._records[conversation_id], "title": title})

    def set_pinned(self, owner_id: str, conversation_id: str, pinned: bool) -> None:
        self._put({**self._records[conversation_id], "pinned": pinned})

    def delete(self, owner_id: str, conversation_id: str) -> None:
        current = self._records.get(conversation_id)
        if current and not current.get("deleted"):
            operation = {"id": str(uuid4()), "kind": "delete", "conversation_id": conversation_id,
                         "expected_revision": current.get("revision", 0)}
            self._pending.append(operation)
            self._project(operation)

