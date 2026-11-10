"""Validated conversation records shared by browser and SQLite stores."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone

STATE_KEYS = ("messages", "phase", "pending", "latest_result", "inventory", "excluded_ingredients")
VALID_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


class HistoryFormatError(ValueError):
    """Stored JSON does not follow the conversation schema."""


def _string_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_result(result) -> None:
    if result is None:
        return
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise HistoryFormatError("Data rekomendasi dalam riwayat tidak valid.")
    for field in ("user_ingredients", "excluded_ingredients"):
        if field in result and not _string_list(result[field]):
            raise ValueError("Daftar bahan dalam riwayat tidak valid.")
    if result["status"] != "ok":
        return
    if not _string_list(result.get("user_ingredients")):
        raise HistoryFormatError("Bahan pengguna dalam rekomendasi tidak lengkap.")
    if not isinstance(result.get("results"), list):
        raise HistoryFormatError("Daftar resep dalam riwayat tidak valid.")
    for recipe in result["results"]:
        if not isinstance(recipe, dict):
            raise HistoryFormatError("Data resep dalam riwayat tidak valid.")
        if not all(field in recipe for field in ("id", "title", "ingredient_names", "matched_user_items", "matched_recipe_items", "missing_items")):
            raise HistoryFormatError("Data resep dalam riwayat tidak lengkap.")
        for field in ("title", "steps_text"):
            if field in recipe and not isinstance(recipe[field], str):
                raise ValueError("Judul atau langkah resep dalam riwayat tidak valid.")
        for field in ("ingredient_names", "matched_recipe_items", "matched_user_items", "missing_items"):
            if field in recipe and not _string_list(recipe[field]):
                raise ValueError("Daftar bahan resep dalam riwayat tidak valid.")


def validate_snapshot(value: dict) -> dict:
    if not isinstance(value, dict):
        raise HistoryFormatError("Format riwayat percakapan tidak didukung.")
    snapshot = deepcopy(value)
    if snapshot.get("inventory") is None:
        initial = (snapshot.get("pending") or {}).get("initial_result") if isinstance(snapshot.get("pending"), dict) else None
        result = snapshot.get("latest_result") or initial or {}
        snapshot["inventory"] = result.get("user_ingredients", []) if isinstance(result, dict) else []
    snapshot.setdefault("excluded_ingredients", [])
    if snapshot.get("phase") == "review_ingredients":
        snapshot["phase"], snapshot["latest_result"] = "await_query", None
    if (
        not all(key in snapshot for key in STATE_KEYS)
        or not isinstance(snapshot.get("phase"), str)
        or snapshot["phase"] not in {"await_query", "await_confirmation"}
        or not isinstance(snapshot.get("messages"), list)
        or not _string_list(snapshot.get("inventory"))
        or not _string_list(snapshot.get("excluded_ingredients"))
    ):
        raise ValueError("Format riwayat percakapan tidak didukung.")
    for message in snapshot["messages"]:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise ValueError("Pesan dalam riwayat tidak valid.")
        _validate_result(message.get("result"))
    _validate_result(snapshot["latest_result"])
    pending = snapshot["pending"]
    if snapshot["phase"] == "await_confirmation" and not isinstance(pending, dict):
        raise ValueError("Data klarifikasi pada riwayat tidak lengkap.")
    if pending is not None:
        if not isinstance(pending, dict):
            raise ValueError("Data klarifikasi pada riwayat tidak valid.")
        for field in ("base_inventory", "retained_ingredients", "candidate_staples", "additional_ingredients", "excluded_ingredients"):
            if field in pending and not _string_list(pending[field]):
                raise ValueError("Bahan klarifikasi dalam riwayat tidak valid.")
        if "original_query" in pending and not isinstance(pending["original_query"], str):
            raise ValueError("Pertanyaan dalam riwayat tidak valid.")
        _validate_result(pending.get("initial_result"))
        if snapshot["phase"] == "await_confirmation":
            if pending.get("kind") == "soy_sauce_type":
                if "kecap" not in snapshot["inventory"]:
                    raise HistoryFormatError("Bahan yang perlu diperjelas dalam riwayat tidak ditemukan.")
            elif (
                not pending.get("candidate_staples") or not isinstance(pending.get("initial_result"), dict)
                or pending["initial_result"].get("status") != "ok"
            ):
                raise HistoryFormatError("Data klarifikasi pada riwayat tidak lengkap.")
    # Reject non-JSON objects and non-finite numbers before either backend saves.
    json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    return {key: snapshot[key] for key in STATE_KEYS}


def validate_record(value: dict, *, tombstones: bool = False) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not VALID_ID.fullmatch(value["id"]):
        raise ValueError("Identitas percakapan tidak valid.")
    revision = value.get("revision", 0)
    if type(revision) is not int or revision < 0:
        raise ValueError("Versi percakapan tidak valid.")
    if tombstones and value.get("deleted") is True:
        return {"id": value["id"], "revision": revision, "deleted": True}
    if not isinstance(value.get("title"), str) or not value["title"].strip() or type(value.get("pinned")) is not bool:
        raise ValueError("Judul atau pin percakapan tidak valid.")
    record = {"id": value["id"], "title": " ".join(value["title"].split())[:72],
              "pinned": value["pinned"], "snapshot": validate_snapshot(value.get("snapshot")), "revision": revision}
    for field in ("created_at", "updated_at"):
        try:
            timestamp = datetime.fromisoformat(value[field].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError
            record[field] = timestamp.astimezone(timezone.utc).isoformat()
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Waktu percakapan tidak valid.") from exc
    return record


