"""Conversation fillers, unavailable ingredients, and seasoning aliases."""

from copy import deepcopy
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dapuria.llama_service import LlamaService
from dapuria.pantry import update_inventory
from dapuria.preferences import ingredient_is_excluded, update_preferences
from dapuria.preprocessing import ingredient_matches, parse_user_ingredients
from dapuria.recipe_data import normalize_recipe


@pytest.fixture
def llama(monkeypatch):
    monkeypatch.setattr(LlamaService, "available", lambda self: False)
    return LlamaService()


@pytest.mark.parametrize("reply", ["ada dongg", "ada donggg!", "ada dongg ya", "iya dongg", "ada kok"])
def test_emphatic_replies_confirm_the_pending_ingredient_without_adding_fillers(llama, reply):
    parsed = llama.interpret_clarification_reply(reply, ["bawang putih"])
    assert parsed.confirmation.status == "confirmed"
    assert parsed.confirmation.ingredients == ["bawang putih"]
    assert parsed.additional_ingredients == []
    assert parse_user_ingredients(reply) == []
    assert update_inventory(["jamur"], reply) == ["jamur"]


def test_questions_and_unrelated_ingredients_do_not_confirm_the_pending_ingredient(llama):
    reply = llama.interpret_clarification_reply("ada, bawang bombay juga ada?", ["bawang merah"])
    assert reply.confirmation.status == "unclear"
    assert reply.additional_ingredients == []
    reply = llama.interpret_clarification_reply("ada bawang bombay", ["bawang merah"])
    assert reply.confirmation.status == "unclear"
    assert reply.additional_ingredients == ["bawang bombay"]


@pytest.mark.parametrize(("message", "excluded", "remaining"), [
    ("aduh saya ga punya tempe,telur toge. gimana dong?",
     ["tempe", "telur", "tauge"], ["susu", "jamur"]),
    ("aku nggak punya tempe, telur dan toge",
     ["tempe", "telur", "tauge"], ["susu", "jamur"]),
    ("aku ngga punya telur", ["telur"], ["susu", "jamur", "tempe", "tauge"]),
    ("aku belum punya telur, tapi punya tomat",
     ["telur"], ["susu", "jamur", "tempe", "tauge", "tomat"]),
    ("aku tidak punya tepung terigu", ["tepung terigu"], ["susu", "jamur", "tempe", "telur", "tauge"]),
    ("saya ga punya telur ayam", ["telur ayam"], ["susu", "jamur", "tempe", "telur", "tauge"]),
])
def test_named_unavailability_excludes_recipes_and_removes_only_the_named_stock(message, excluded, remaining):
    update = update_preferences([], message)
    assert update.has_directive
    assert update.excluded == excluded
    assert update_inventory(["susu", "jamur", "tempe", "telur", "tauge"], update.inventory_text) == remaining
    assert "gimana" not in update.inventory_text
    assert all(ingredient_is_excluded(name, update.excluded) for name in excluded)


@pytest.mark.parametrize("reply", ["tidak ada", "nggak punya", "ga ada", "tidak ada dongg"])
def test_generic_denials_still_answer_the_question_instead_of_starting_a_new_search(llama, reply):
    update = update_preferences([], reply)
    assert not update.has_directive
    assert update.excluded == []
    assert llama.interpret_clarification_reply(reply, ["bawang putih"]).confirmation.status == "declined"


def test_exclusion_lists_keep_negation_scope_and_support_existing_controls():
    update = update_preferences(["udang"], "ga punya telur dan toge tapi punya tomat")
    assert update.excluded == ["udang", "telur", "tauge"]
    assert update_inventory(["jamur", "telur"], update.inventory_text) == ["jamur", "tomat"]
    assert update_preferences(update.excluded, "boleh pakai taoge").excluded == ["udang", "telur"]
    assert update_preferences(update.excluded, "hapus batasan").excluded == []
    assert update_preferences([], "jangan pakai tempe dan telur").excluded == ["tempe", "telur"]


@pytest.mark.parametrize("alias", [
    "micin", "MSG", "sasa", "Ajinomoto", "vetsin", "micin sasa", "sasa msg", "penyedap rasa sasa",
])
def test_msg_aliases_share_one_ingredient_identity(alias):
    assert parse_user_ingredients(alias) == ["msg"]
    assert ingredient_matches("micin", alias)
    assert ingredient_matches(alias, "ajinomoto")
    assert ingredient_is_excluded(alias, ["micin"])


@pytest.mark.parametrize("product", [
    "santan sasa", "sasa santan cair", "tepung serbaguna sasa", "bumbu sasa soto ayam",
    "penyedap non msg", "kaldu bubuk non msg",
])
def test_other_brand_products_and_non_msg_seasonings_do_not_count_as_msg(product):
    assert not ingredient_matches("micin", product)
    assert not ingredient_matches(product, "micin")
    assert not ingredient_is_excluded(product, ["micin"])


def test_aliases_are_deduplicated_and_sprout_exclusions_match_corpus_spellings():
    recipe = normalize_recipe({"id": "aliases", "ingredient_names": ["micin", "sasa", "msg", "ajinomoto"]})
    assert recipe["ingredient_names"] == ["msg"]
    assert parse_user_ingredients("telur toge") == ["telur", "tauge"]
    assert parse_user_ingredients("telur ayam, jamur tiram, tepung terigu") == [
        "telur ayam", "jamur tiram", "tepung terigu",
    ]
    assert all(ingredient_is_excluded(name, ["toge"]) for name in ["toge", "taoge", "tauge", "taoge kacang hijau"])


def test_continuing_an_old_inventory_cleans_fillers_and_canonicalizes_aliases():
    assert update_inventory(["jamur", "dongg", "micin", "sasa", "toge"], "ada dongg") == [
        "jamur", "msg", "tauge",
    ]


@pytest.mark.parametrize("completed", [False, True])
def test_chat_replaces_recommendations_after_named_unavailability(monkeypatch, tmp_path, completed):
    candidates = [{
        "id": str(index), "title": title, "ingredient_names": names,
        "sbert_score": 0.95 - index * 0.03, "steps_text": "Masak hingga matang.",
    } for index, (title, names) in enumerate([
        ("Telur Ceplok Tomat", ["telur", "tomat", "gula", "bawang putih"]),
        ("Sup Toge Jamur", ["toge", "jamur", "gula", "bawang putih"]),
        ("Tumis Tempe", ["tempe", "jamur", "gula", "bawang putih"]),
        ("Tahu Siram Saus Jamur", ["tahu", "jamur", "susu", "bawang putih"]),
        ("Jamur Kukus", ["jamur", "tomat", "susu", "bawang putih", "sasa"]),
    ])]
    searches = []

    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            searches.append(list(ingredients))
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    monkeypatch.setattr(LlamaService, "available", lambda self: False)
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("DAPURIA_HISTORY_BACKEND", "sqlite")
    st.cache_resource.clear()
    try:
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "app.py")).run()
        app.chat_input[0].set_value("susu, jamur,gula,tomat").run(timeout=20)
        assert not app.exception
        assert app.session_state.phase == "await_confirmation"
        asked = app.session_state.pending["candidate_staples"]
        # A continued chat may still contain the filler saved by the old parser.
        app.session_state.inventory = [*app.session_state.inventory, "dongg"]
        pending = deepcopy(app.session_state.pending)
        pending["base_inventory"].append("dongg")
        app.session_state.pending = pending
        if completed:
            app.chat_input[0].set_value("ada dongg").run(timeout=20)
            assert not app.exception
            assert app.session_state.phase == "await_query"
            assert app.session_state.inventory == ["susu", "jamur", "gula", "tomat", *asked]
            app.button(key="continue_chat").click().run()

        before = [item for item in app.session_state.inventory if item != "dongg"]
        count = len(searches)
        app.chat_input[0].set_value("aduh saya ga punya tempe,telur toge. gimana dong?").run(timeout=20)

        assert not app.exception
        assert len(searches) == count + 1
        assert app.session_state.phase == "await_query"
        assert app.session_state.pending is None
        assert app.session_state.inventory == before
        assert app.session_state.excluded_ingredients == ["tempe", "telur", "tauge"]
        result = app.session_state.latest_result
        assert result["status"] == "ok"
        assert {recipe["title"] for recipe in result["results"]} == {"Tahu Siram Saus Jamur", "Jamur Kukus"}
        assert not any(
            ingredient_is_excluded(name, app.session_state.excluded_ingredients)
            for recipe in result["results"] for name in recipe["ingredient_names"]
        )
        assert "tanpa **tempe, telur, tauge**" in app.session_state.messages[-1]["content"]
    finally:
        st.cache_resource.clear()
