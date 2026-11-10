"""Soy sauce type selection happens before retrieval and survives chat restore."""

from copy import deepcopy
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dapuria.chat_history import ChatHistoryStore
from dapuria.ingredient_clarification import (
    expand_soy_sauce_reply,
    update_soy_sauce_inventory,
)
from dapuria.preferences import update_preferences
from dapuria.preprocessing import ingredient_matches


@pytest.mark.parametrize(("reply", "types"), [
    ("manis", ["kecap manis"]),
    ("yang asin dongg", ["kecap asin"]),
    ("iya yang manis", ["kecap manis"]),
    ("jenis asin", ["kecap asin"]),
    ("kecapnya manis", ["kecap manis"]),
    ("ada kecap asin", ["kecap asin"]),
    ("dua-duanya", ["kecap manis", "kecap asin"]),
    ("keduanya ada", ["kecap manis", "kecap asin"]),
    ("manis dan asin", ["kecap manis", "kecap asin"]),
    ("kecap ikan", ["kecap ikan"]),
    ("kecap inggris", ["kecap inggris"]),
    ("yang kecap jamur", ["kecap jamur"]),
])
def test_explicit_type_replaces_generic_soy_sauce_and_preserves_other_foods(reply, types):
    expanded = expand_soy_sauce_reply(reply)
    assert update_soy_sauce_inventory(["ayam", "kecap", "garam"], expanded) == ["ayam", "garam", *types]


@pytest.mark.parametrize("reply", [
    "iya", "ada dongg", "lainnya", "jenis lain", "saya lupa", "manis atau asin",
    "mungkin manis", "kecap manis?", "apakah kecap asin", "kayaknya kecap manis", "kecap lainnya",
])
def test_unclear_answers_do_not_guess_a_type_or_add_filler(reply):
    assert update_soy_sauce_inventory(["ayam", "kecap"], expand_soy_sauce_reply(reply)) == ["ayam", "kecap"]


def test_contextual_expansion_does_not_rename_other_ingredients_or_ignore_denials():
    text = "kecap manis, ikan asin, jagung manis"
    assert expand_soy_sauce_reply(text) == text
    update = update_preferences([], expand_soy_sauce_reply("nggak punya yang asin, tapi punya manis"))
    assert update.excluded == ["kecap asin"]
    assert update_soy_sauce_inventory(["ayam", "kecap"], update.inventory_text) == ["ayam", "kecap manis"]
    assert not ingredient_matches("kecap manis", "kecap asin")
    assert not ingredient_matches("kecap asin", "kecap manis")


@pytest.fixture
def soy_sauce_chat(monkeypatch, tmp_path):
    searches = []
    candidates = [{
        "id": name, "title": title, "sbert_score": 0.9,
        "ingredient_names": ["ayam", "kecap " + name, "garam", "bawang putih"],
        "steps_text": "Masak hingga matang.",
    } for name, title in (("manis", "Ayam Kecap Manis"), ("asin", "Sup Ayam Kecap Asin"),
                          ("ikan", "Ayam Panggang Kecap Ikan"))]

    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            searches.append(list(ingredients))
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    monkeypatch.setattr("dapuria.llama_service.LlamaService.available", lambda self: False)
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("DAPURIA_HISTORY_BACKEND", "sqlite")
    st.cache_resource.clear()
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py")).run()
    yield app, searches, root, tmp_path / "history.sqlite3"
    st.cache_resource.clear()


@pytest.mark.parametrize("reply", ["manis", "asin", "kecap ikan", "dua-duanya"])
def test_chat_asks_before_searching_and_uses_the_selected_type(soy_sauce_chat, reply):
    app, searches, _, _ = soy_sauce_chat
    app.chat_input[0].set_value("ayam, kecap, garam").run(timeout=20)
    assert not app.exception
    assert not searches
    assert app.session_state.phase == "await_confirmation"
    assert app.session_state.pending == {"kind": "soy_sauce_type"}
    assert "jenis kecap lainnya" in app.session_state.messages[-1]["content"]
    assert not any(button.label == "Ada, tambahkan" for button in app.button)

    app.chat_input[0].set_value(reply).run(timeout=20)
    assert not app.exception
    assert app.session_state.pending is None
    assert app.session_state.latest_result["status"] == "ok"
    assert len(searches) == 1
    assert "kecap" not in searches[0]
    assert searches[0] == app.session_state.inventory
    assert not any(item in searches[0] for item in ("manis", "asin", "dua-duanya"))
    if reply in {"manis", "asin"}:
        recipes = app.session_state.latest_result["results"]
        own_recipe = next(recipe for recipe in recipes if recipe["id"] == reply)
        other_recipe = next(recipe for recipe in recipes if recipe["id"] == {"manis": "asin", "asin": "manis"}[reply])
        assert "kecap " + reply in own_recipe["matched_recipe_items"]
        assert "kecap " + other_recipe["id"] in other_recipe["missing_items"]


def test_type_question_survives_restore_and_unclear_replies_keep_it_pending(soy_sauce_chat):
    app, searches, root, history_path = soy_sauce_chat
    app.chat_input[0].set_value("ayam, kecap").run(timeout=20)
    chat_id = app.session_state.conversation_id
    store = ChatHistoryStore(history_path)
    assert store.load("local", chat_id)["pending"] == {"kind": "soy_sauce_type"}
    restored = AppTest.from_file(str(root / "app.py"))
    restored.query_params["chat"] = chat_id
    restored.run()
    assert not restored.exception
    restored.chat_input[0].set_value("ada dongg").run(timeout=20)
    assert not restored.exception
    assert not searches
    assert restored.session_state.inventory == ["ayam", "kecap"]
    assert restored.session_state.pending == {"kind": "soy_sauce_type"}
    restored.chat_input[0].set_value("manis?").run(timeout=20)
    assert not restored.exception
    assert not searches
    assert restored.session_state.inventory == ["ayam", "kecap"]
    restored.button(key="soy_sauce_Kecap asin").click().run(timeout=20)
    assert not restored.exception
    assert searches == [["ayam", "kecap asin"]]


@pytest.mark.parametrize("query", ["ayam, kecap manis", "ayam, kecap asin", "ayam, kecap ikan", "ayam tanpa kecap"])
def test_named_types_and_excluded_soy_sauce_do_not_prompt_for_a_type(soy_sauce_chat, query):
    app, searches, _, _ = soy_sauce_chat
    app.chat_input[0].set_value(query).run(timeout=20)
    assert not app.exception
    assert searches
    assert (app.session_state.pending or {}).get("kind") != "soy_sauce_type"
    assert "jenis kecap lainnya" not in app.session_state.messages[-1]["content"]


def test_unspecified_soy_sauce_added_during_staple_confirmation_gets_its_own_question(soy_sauce_chat):
    app, searches, _, _ = soy_sauce_chat
    app.chat_input[0].set_value("ayam, garam").run(timeout=20)
    assert app.session_state.phase == "await_confirmation"
    count = len(searches)
    app.chat_input[0].set_value("ada kecap juga").run(timeout=20)
    assert not app.exception
    assert app.session_state.pending == {"kind": "soy_sauce_type"}
    assert len(searches) == count
    app.chat_input[0].set_value("manis").run(timeout=20)
    assert not app.exception
    assert searches[-1] == ["ayam", "garam", "kecap manis"]


def test_user_can_remove_unspecified_soy_sauce_and_continue_searching(soy_sauce_chat):
    app, searches, _, _ = soy_sauce_chat
    app.chat_input[0].set_value("ayam, kecap").run(timeout=20)
    app.chat_input[0].set_value("ga punya kecap, tanpa kecap saja").run(timeout=20)
    assert not app.exception
    assert searches == [["ayam"]]
    assert app.session_state.inventory == ["ayam"]
    assert app.session_state.pending is None
