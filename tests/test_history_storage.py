"""Acknowledged browser writes and history validation, independent of model inference."""

from copy import deepcopy

import pytest

from dapuria.browser_history import BrowserHistoryStore
from dapuria.chat_history import ChatHistoryStore
from dapuria.history_format import validate_record


@pytest.fixture
def snapshot():
    result = {"status": "ok", "user_ingredients": ["jamur", "tomat"], "results": [{
        "id": "recipe-1", "title": "Jamur Kukus", "ingredient_names": ["jamur", "tomat", "garam"],
        "matched_recipe_items": ["jamur", "tomat"], "matched_user_items": ["jamur", "tomat"],
        "missing_items": ["garam"], "steps_text": "Kukus hingga matang.",
    }]}
    return {
        "messages": [{"role": "user", "content": "jamur, tomat"},
                     {"role": "assistant", "content": "Ada garam juga?", "result": result}],
        "phase": "await_confirmation", "latest_result": None,
        "pending": {"original_query": "jamur, tomat", "candidate_staples": ["garam"],
                    "base_inventory": ["jamur", "tomat"], "initial_result": result, "additional_ingredients": []},
        "inventory": ["jamur", "tomat"], "excluded_ingredients": ["telur"],
    }


@pytest.fixture
def record(snapshot):
    return {"id": "chat-1", "title": "Jamur dan tomat", "pinned": True,
            "created_at": "2026-11-02T00:00:00+00:00", "updated_at": "2026-11-02T01:00:00+00:00",
            "snapshot": snapshot}


def browser_with_records(records=()):
    store = BrowserHistoryStore()
    request = store.request()
    assert store.receive({"id": request["id"], "records": list(records)})
    return store


def acknowledge(store, durable):
    request = store.request()
    if request["kind"] == "apply":
        for operation in request["operations"]:
            assert durable.get(operation["conversation_id"], {}).get("revision", 0) == operation["expected_revision"]
            record = deepcopy(operation.get("record", {"id": operation["conversation_id"], "deleted": True}))
            record["revision"] = operation["expected_revision"] + 1
            durable[record["id"]] = record
    store.receive({"id": request["id"], "records": deepcopy(list(durable.values()))})


def test_browser_restore_keeps_full_chat_recipes_and_pending_confirmation(record):
    durable = {}
    first = browser_with_records()
    first.save("local", record["id"], record["snapshot"])
    first.rename("local", record["id"], record["title"])
    first.set_pinned("local", record["id"], record["pinned"])
    acknowledge(first, durable)
    restored = browser_with_records(durable.values())
    assert restored.list("local") == first.list("local")
    assert restored.load("local", "chat-1") == record["snapshot"]
    assert restored.list("local")[0]["pinned"]
    assert restored.request() is None


def test_writes_are_stable_until_acknowledged_and_late_responses_do_not_drop_new_edits(snapshot):
    store = browser_with_records()
    store.save("local", "chat-1", snapshot)
    request = store.request()
    assert request == store.request()
    store.rename("local", "chat-1", "Judul baru")
    assert not store.receive({"id": "stale-request", "records": []})
    assert store.syncing
    committed = deepcopy(request["operations"][0]["record"])
    committed["revision"] = 1
    store.receive({"id": request["id"], "records": [committed]})
    assert store.load("local", "chat-1") == snapshot
    assert store.list("local")[0]["title"] == "Judul baru"
    assert store.request()["operations"][0]["expected_revision"] == 1


def test_failed_browser_write_is_retryable_without_losing_the_chat(snapshot):
    store = browser_with_records()
    store.save("local", "chat-1", snapshot)
    request = store.request()
    store.receive({"id": request["id"], "error": "quota"})
    assert store.error
    assert store.request() is None
    assert store.load("local", "chat-1") == snapshot
    store.retry()
    retry = store.request()
    assert retry["id"] != request["id"]
    assert retry["operations"] == request["operations"]
    acknowledge(store, {})
    assert not store.syncing
    assert store.error is None


def test_tab_conflict_preserves_both_versions_as_separate_chats(record):
    original = validate_record({**record, "revision": 1})
    store = browser_with_records([original])
    store.rename("local", "chat-1", "Versi tab ini")
    remote = {**original, "title": "Versi tab lain", "revision": 2}
    store.receive({"id": store.request()["id"], "error": "conflict", "records": [remote]})
    assert {item["title"] for item in store.list("local")} == {"Versi tab lain", "Versi tab ini (salinan)"}
    assert store.redirects["chat-1"] != "chat-1"
    assert store.notice
    acknowledge(store, {"chat-1": remote})
    assert not store.syncing


def test_external_refresh_marks_the_active_snapshot_for_reload(record):
    store = browser_with_records([record])
    assert not store.refreshed
    changed = deepcopy(record)
    changed["snapshot"]["messages"].append({"role": "user", "content": "Pesan dari tab lain"})
    changed["revision"] = 2
    store.refresh()
    assert store.request()["kind"] == "read"
    store.receive({"id": store.request()["id"], "records": [changed]})
    assert store.refreshed
    assert store.load("local", record["id"]) == changed["snapshot"]


@pytest.mark.parametrize("backend", ["sqlite", "browser"])
def test_history_backends_preserve_titles_pins_and_isolate_returned_values(backend, tmp_path, snapshot):
    store = ChatHistoryStore(tmp_path / "history.sqlite3") if backend == "sqlite" else browser_with_records()
    store.save("local", "chat-1", snapshot)
    store.rename("local", "chat-1", "Judul pribadi")
    store.set_pinned("local", "chat-1", True)
    store.save("local", "chat-1", snapshot)
    item = store.list("local")[0]
    assert (item["title"], item["pinned"]) == ("Judul pribadi", True)
    loaded = store.load("local", "chat-1")
    loaded["inventory"].append("micin")
    assert store.load("local", "chat-1") == snapshot
    store.delete("local", "chat-1")
    assert not store.list("local")
    assert store.load("local", "chat-1") is None


@pytest.mark.parametrize("field,value", [
    ("title", []), ("pinned", "false"), ("id", "../../other"), ("updated_at", "invalid"),
    ("created_at", "2026-11-02T00:00:00"),
])
def test_invalid_record_metadata_is_rejected(field, value, record):
    with pytest.raises(ValueError):
        validate_record({**record, field: value})


@pytest.mark.parametrize("invalid", [
    {"messages": [{"role": [], "content": "invalid"}]}, {"phase": []},
    {"inventory": "jamur"}, {"pending": None},
    {"pending": {"candidate_staples": [1]}},
    {"latest_result": {"status": "ok", "results": [{"ingredient_names": 1}]}},
])
def test_invalid_snapshots_do_not_overwrite_existing_history(invalid, record, tmp_path):
    bad = {**record["snapshot"], **invalid}
    sqlite = ChatHistoryStore(tmp_path / "history.sqlite3")
    browser = browser_with_records()
    for store in (sqlite, browser):
        store.save("local", record["id"], record["snapshot"])
        with pytest.raises(ValueError):
            store.save("local", record["id"], bad)
        assert len(store.list("local")) == 1
        assert store.load("local", record["id"]) == record["snapshot"]
