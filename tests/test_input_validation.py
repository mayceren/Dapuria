import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dapuria.chat_history import ChatHistoryStore
from dapuria.config import Settings
from dapuria.llama_service import LlamaService
from dapuria.pantry import update_inventory
from dapuria.preferences import update_preferences
from dapuria.preprocessing import (
    parse_ingredient,
    parse_user_ingredients,
    parse_user_input,
)
from dapuria.recipe_service import RecipeService
from dapuria.validation import validate_initial


@pytest.mark.parametrize("text", [
    "panci", "wajan", "spatula", "pisau", "talenan", "kompor", "blender", "saringan", "baskom",
    "toples", "gelas", "piring", "sendok", "garpu", "mangkuk", "pake panci", "pakai wajan",
    "menggunakan blender", "masak pake kompor", "ada gelas", "punya pisau",
    "aku punya panci dan spatula", "goreng menggunakan wajan", "ada gelas, pake panci",
    "halo", "apa kabar?", "ada laptop dan mouse", "aku punya meja dan kursi",
    "tolong nyalakan kompor", "hari ini cuacanya panas",
])
def test_non_food_messages_do_not_produce_ingredients(text):
    assert parse_user_ingredients(text) == []
    assert parse_user_input(text).unrecognized == []
    assert update_inventory([], text) == []


@pytest.mark.parametrize(("text", "expected"), [
    ("aku punya telur, panci, tomat, dan spatula", ["telur", "tomat"]),
    ("ada ayam, pakai wajan, bawang putih", ["ayam", "bawang putih"]),
    ("masak pake telur dan tomat", ["telur", "tomat"]),
    ("goreng menggunakan telur dan tahu", ["telur", "tahu"]),
    ("ayam menggunakan wajan dan tomat", ["ayam", "tomat"]),
    ("2 butir telur, kecap asin, tomat, seledri", ["telur", "kecap asin", "tomat", "seledri"]),
    ("1 gelas susu", ["susu"]), ("2 sendok makan gula", ["gula"]), ("1 mangkuk nasi", ["nasi"]),
    ("0,5 kg dada ayam, 2 siung baput", ["dada ayam", "bawang putih"]),
    ("seledri, daun bawang, jamur tiram, saus tiram, tepung maizena",
     ["seledri", "daun bawang", "jamur tiram", "saus tiram", "tepung maizena"]),
    ("daun katuk, kecombrang, andaliman, terubuk", ["daun katuk", "kecombrang", "andaliman", "terubuk"]),
])
def test_foods_quantities_and_conversational_prefixes_remain_supported(text, expected):
    assert parse_user_ingredients(text) == expected


def test_unrecognized_names_remain_separate_from_inventory():
    parsed = parse_user_input("aku punya telur dan buah antahberantah")
    assert parsed.ingredients == ["telur"]
    assert parsed.unrecognized == ["buah antahberantah"]
    assert update_inventory(["tomat"], "buah antahberantah") == ["tomat"]
    assert parse_user_ingredients("putih") == []


def test_corpus_vocabulary_accepts_additional_food_names_without_changing_recipe_normalization(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ingredient_names"])
        writer.writeheader()
        writer.writerow({"ingredient_names": json.dumps(["buah nama lokal baru", "pakai wajan"])})
    monkeypatch.setattr("dapuria.preprocessing.Settings", lambda: SimpleNamespace(corpus_path=corpus))
    assert parse_user_ingredients("buah nama lokal baru") == ["buah nama lokal baru"]
    assert parse_user_ingredients("pakai wajan") == []
    assert parse_ingredient("bawang putih dihaluskan")["name"] == "bawang putih"
    assert parse_ingredient("toples air")["name"] == "air"


@pytest.mark.parametrize("text", ["aku punya panci", "pakai wajan", "ada laptop dan mouse", "halo"])
def test_irrelevant_messages_preserve_existing_food_inventory(text):
    assert update_inventory(["telur", "tomat"], text) == ["telur", "tomat"]


def test_inventory_edits_and_food_restrictions_remain_validated():
    inventory = ["telur", "tomat"]
    assert update_inventory(inventory, "tambah tahu dan panci") == ["telur", "tomat", "tahu"]
    assert update_inventory(inventory, "hapus telur") == ["tomat"]
    assert update_inventory(inventory, "ganti telur dengan tahu") == ["tomat", "tahu"]
    assert update_inventory(inventory, "ganti telur dengan panci") == inventory
    assert update_inventory(inventory, "ganti telur dengan buah antahberantah") == inventory
    assert update_preferences([], "jangan pakai telur dan panci").excluded == ["telur"]
    invalid = update_preferences(["telur"], "hapus batasan panci")
    assert invalid.excluded == ["telur"]
    assert not invalid.has_directive
    assert update_preferences(["telur"], "hapus batasan").excluded == []
    assert validate_initial(["pake panci"]) == (False, "no_ingredients")
    assert validate_initial(["garam"]) == (False, "need_main_ingredient")


@pytest.mark.parametrize("reply", ["pakai panci", "aku punya panci", "halo", "ada laptop dan mouse", "telur"])
def test_irrelevant_confirmation_does_not_rely_on_ollama_to_guess_ownership(monkeypatch, reply):
    llama = LlamaService()
    calls = []
    monkeypatch.setattr(llama, "available", lambda: True)
    monkeypatch.setattr(llama, "_chat", lambda *args, **kwargs: calls.append(reply) or '{"confirmed_ingredients":["bawang putih"]}')
    parsed = llama.interpret_clarification_reply(reply, ["bawang putih"])
    assert parsed.confirmation.status == "unclear"
    assert parsed.confirmation.ingredients == []
    assert calls == []
    assert llama.interpret_confirmation("iya", ["panci"]).ingredients == []


def test_service_does_not_retrieve_for_invalid_or_unrecognized_input(monkeypatch):
    calls = []

    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            calls.append(query)
            return []

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    service = RecipeService(Settings())
    for message in ("ada gelas, pake panci", "halo", "ada laptop", "buah antahberantah"):
        result = service.handle_user_input(message)
        assert result["status"] == "no_ingredients"
        assert result["user_ingredients"] == []
        assert "bahan utama" not in result["message"]
        assert service.rerank(message) == ([], [])
    assert calls == []


@pytest.fixture
def input_chat(monkeypatch, tmp_path):
    searches = []
    initializations = []
    candidates = [
        {"id": "egg", "title": "Telur Tomat", "ingredient_names": ["telur", "tomat", "bawang putih", "kecap asin"],
         "steps_text": "Masak hingga matang.", "sbert_score": 0.9},
        {"id": "tofu", "title": "Tahu Kukus", "ingredient_names": ["tahu", "telur", "tomat", "bawang putih"],
         "steps_text": "Kukus hingga matang.", "sbert_score": 0.8},
    ]

    class Retriever:
        def __init__(self, settings):
            initializations.append(True)

        def retrieve(self, query, ingredients):
            searches.append(list(ingredients))
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    monkeypatch.setattr(LlamaService, "available", lambda self: False)
    monkeypatch.setenv("DAPURIA_HISTORY_BACKEND", "sqlite")
    path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(path))
    st.cache_resource.clear()
    try:
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "app.py")).run()
        yield app, searches, initializations, ChatHistoryStore(path)
    finally:
        st.cache_resource.clear()


@pytest.mark.parametrize("message", ["ada gelas, pake panci", "halo", "ada laptop dan mouse", "buah antahberantah"])
def test_chat_rejects_non_food_before_loading_services_or_persisting_inventory(input_chat, message):
    app, searches, initializations, store = input_chat
    app.chat_input[0].set_value(message).run(timeout=20)
    assert not app.exception
    assert searches == initializations == []
    assert app.session_state.inventory == []
    assert app.session_state.pending is None
    assert app.session_state.latest_result is None
    assert "bahan utama" not in app.session_state.messages[-1]["content"]
    saved = store.load("local", app.session_state.conversation_id)
    assert saved["inventory"] == []
    assert "result" not in saved["messages"][-1]


def test_irrelevant_chat_preserves_pending_confirmation_and_existing_inventory(input_chat):
    app, searches, _, store = input_chat
    app.chat_input[0].set_value("telur, tomat").run(timeout=20)
    assert not app.exception
    pending = deepcopy(app.session_state.pending)
    count = len(searches)
    app.chat_input[0].set_value("aku punya panci").run(timeout=20)
    assert not app.exception
    assert app.session_state.inventory == ["telur", "tomat"]
    assert app.session_state.phase == "await_confirmation"
    assert app.session_state.pending == pending
    assert len(searches) == count
    app.chat_input[0].set_value("ada dongg").run(timeout=20)
    assert not app.exception
    assert app.session_state.inventory == ["telur", "tomat", *pending["candidate_staples"]]
    saved = store.load("local", app.session_state.conversation_id)
    assert saved["inventory"] == app.session_state.inventory
    assert saved["latest_result"]["user_ingredients"] == app.session_state.inventory
    app.button(key="continue_chat").click().run()
    count = len(searches)
    app.chat_input[0].set_value("ada laptop dan mouse").run(timeout=20)
    assert not app.exception
    assert len(searches) == count
    assert app.session_state.pending is None
    assert app.session_state.inventory == saved["inventory"]


def test_mixed_chat_and_type_clarification_keep_only_food(input_chat):
    app, searches, _, _ = input_chat
    app.chat_input[0].set_value("aku punya telur, panci, tomat, dan spatula, kecap").run(timeout=20)
    assert not app.exception
    assert app.session_state.inventory == ["telur", "tomat", "kecap"]
    pending = deepcopy(app.session_state.pending)
    app.chat_input[0].set_value("menggunakan blender").run(timeout=20)
    assert not app.exception
    assert app.session_state.pending == pending
    assert searches == []
    app.chat_input[0].set_value("asin").run(timeout=20)
    assert not app.exception
    assert searches == [["telur", "tomat", "kecap asin"]]


def test_invalid_replacement_cannot_remove_existing_food_or_trigger_search(input_chat):
    app, searches, _, _ = input_chat
    app.chat_input[0].set_value("telur, tomat").run(timeout=20)
    pending = deepcopy(app.session_state.pending)
    count = len(searches)
    app.chat_input[0].set_value("ganti telur dengan panci").run(timeout=20)
    assert not app.exception
    assert app.session_state.inventory == ["telur", "tomat"]
    assert app.session_state.pending == pending
    assert len(searches) == count


def test_legacy_invalid_inventory_is_not_used_in_a_new_recommendation(input_chat):
    app, searches, _, store = input_chat
    app.chat_input[0].set_value("telur, tomat").run(timeout=20)
    pending = deepcopy(app.session_state.pending)
    pending["base_inventory"] += ["pake panci", "laptop"]
    pending["initial_result"]["user_ingredients"] += ["pake panci", "laptop"]
    app.session_state.pending = pending
    app.session_state.inventory = ["telur", "tomat", "pake panci", "laptop"]
    app.chat_input[0].set_value("tidak ada").run(timeout=20)
    assert not app.exception
    assert searches[-1] == ["telur", "tomat"]
    saved = store.load("local", app.session_state.conversation_id)
    assert saved["inventory"] == ["telur", "tomat"]
    assert saved["latest_result"]["user_ingredients"] == ["telur", "tomat"]
