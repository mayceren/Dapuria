import csv
import json
import random
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dapuria.config import Settings
from dapuria.preprocessing import (
    ingredient_matches,
    parse_ingredient,
    parse_user_ingredients,
)
from dapuria.recipe_data import normalize_recipe
from dapuria.recipe_service import RecipeService
from dapuria.reranking import coverage_stats, harmonic_efficiency
from dapuria.retrieval import HybridRetriever
from dapuria.selection import select_recommendations
from dapuria.validation import is_relevant_enough, relevance_threshold

EGG_QUERY = "2 butir telur, kecap asin, tomat, seledri"
EGG_INVENTORY = ["telur", "kecap asin", "tomat", "seledri"]


def scored_recipe(identifier, names, inventory, final_score=0.5):
    recipe = normalize_recipe({"id": identifier, "title": identifier, "ingredient_names": names,
                               "steps_text": "Masak hingga matang."})
    stats = coverage_stats(inventory, recipe["ingredient_names"])
    return {**recipe, **stats, "final_score": final_score,
            "ingredient_efficiency": harmonic_efficiency(stats["coverage_recipe"], stats["coverage_user"])}


def test_query_parsing_and_coverage_use_food_identity_without_quantity():
    assert parse_user_ingredients(EGG_QUERY) == EGG_INVENTORY
    stats = coverage_stats(EGG_INVENTORY, ["telur ayam", "kecap asin", "tomat", "air", "garam"])
    assert stats["coverage_user"] == 0.75
    assert stats["coverage_recipe"] == 0.6
    assert stats["matched_user_items"] == ["kecap asin", "telur", "tomat"]
    assert stats["missing_items"] == ["air", "garam"]
    assert harmonic_efficiency(stats["coverage_recipe"], stats["coverage_user"]) == pytest.approx(2 / 3)


@pytest.mark.parametrize(("first", "second", "expected"), [
    ("kecap asin", "kecap manis", False), ("kecap manis", "kecap asin", False),
    ("ayam", "telur ayam", False), ("telur ayam", "ayam", False),
    ("telur", "telur ayam", True), ("tomat", "saus tomat", False),
    ("saus tomat", "tomat", False), ("beras", "tepung beras", False),
    ("bawang putih", "bawang putih bubuk", False), ("baput", "bawang putih", True),
    ("teri", "terigu", False), ("garam", "garam halus", True),
])
def test_source_foods_products_aliases_and_word_boundaries(first, second, expected):
    assert ingredient_matches(first, second) is expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("toples air", "air"), ("1 toples air", "air"), ("panci air", "air"),
    ("panci presto air", "air"), ("wadah mie keriting", "mie keriting"),
    ("baskom biji tekwan", "tekwan"), ("wajan", ""), ("saringan kukusan", ""),
    ("toples ukuran sedang untuk wadah", ""), ("wadah tahan panas (untuk mengukus", ""),
    ("loyang untuk mendudukkan kentang", ""), ("🍝bumbu🍝", ""),
    ("bumbu halus", ""), ("bumbu soto instan", "bumbu soto instan"),
    ("minyak untuk mengoles wadah", "minyak untuk mengoles wadah"),
])
def test_food_containers_equipment_and_section_headers(raw, expected):
    assert parse_ingredient(raw)["name"] == expected


@pytest.mark.parametrize("scores", list(product((0.0, 1.0), repeat=3)))
def test_balanced_candidate_beats_low_use_and_excessive_missing_ingredients(scores):
    low = scored_recipe("A", ["telur", "garam"], EGG_INVENTORY, scores[0])
    balanced = scored_recipe("B", ["telur", "kecap asin", "tomat", "garam", "air"], EGG_INVENTORY, scores[1])
    costly = scored_recipe("C", ["telur", "kecap asin", "tomat"] + [f"bahan {index}" for index in range(12)],
                           EGG_INVENTORY, scores[2])
    assert select_recommendations([low, costly, balanced], 3)[0]["id"] == "B"


@pytest.mark.parametrize("inventory", [
    ["ayam", "bawang putih", "kecap manis", "tomat"],
    ["tahu", "tempe", "bawang putih", "kecap manis"],
    ["tempe", "wortel", "bawang merah", "seledri"],
    ["nasi", "telur", "wortel", "bawang putih"],
    ["bayam", "jagung", "bawang merah", "garam"],
])
def test_balancing_generalizes_across_ingredient_combinations(inventory):
    low = scored_recipe("A", inventory[:2] + ["tambahan"], inventory)
    balanced = scored_recipe("B", inventory[:3] + ["tambahan satu", "tambahan dua"], inventory)
    costly = scored_recipe("C", inventory + [f"tambahan {index}" for index in range(15)], inventory)
    assert select_recommendations([costly, low, balanced], 3)[0]["id"] == "B"


def test_matching_more_ingredients_is_not_an_absolute_priority():
    ready = scored_recipe("ready", EGG_INVENTORY[:2], EGG_INVENTORY)
    costly = scored_recipe("costly", EGG_INVENTORY + [f"tambahan {index}" for index in range(30)], EGG_INVENTORY)
    assert select_recommendations([costly, ready], 2)[0]["id"] == "ready"


def test_wlc_contributes_and_identical_quality_is_deterministic():
    first = scored_recipe("a", EGG_INVENTORY[:3] + ["air"], EGG_INVENTORY, 0.3)
    second = scored_recipe("b", EGG_INVENTORY[:3] + ["air"], EGG_INVENTORY, 0.8)
    assert select_recommendations([first, second], 2)[0]["id"] == "b"
    second["final_score"] = first["final_score"]
    original = deepcopy([first, second])
    assert [recipe["id"] for recipe in select_recommendations([second, first], 2)] == ["a", "b"]
    assert [first, second] == original


@pytest.fixture
def egg_corpus_candidates():
    titles = {"telur ayam asin", "scrambled egg with tomatoes", "telur dadar tepung", "tim telur",
              "telur ceplok kuah tomat"}
    with (Path(__file__).resolve().parents[1] / "data/corpus/recipes.csv").open(encoding="utf-8-sig", newline="") as stream:
        return [{"id": f"resep_{index}", "title": row["title_clean"],
                 "ingredient_names": json.loads(row["ingredient_names"]), "steps_text": row["steps_text"],
                 "sbert_score": 0.99 if row["title_clean"].casefold() == "telur ayam asin" else (
                     0.8 if row["title_clean"].casefold() == "scrambled egg with tomatoes" else 0.7
                 )} for index, row in enumerate(csv.DictReader(stream)) if row["title_clean"].casefold() in titles]


def corpus_service(monkeypatch, candidates):
    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    return RecipeService(Settings())


@pytest.mark.parametrize("extra", [[], ["air"]])
def test_actual_corpus_examples_favor_ingredient_combination_without_title_rules(monkeypatch, egg_corpus_candidates, extra):
    service = corpus_service(monkeypatch, egg_corpus_candidates)
    inventory = EGG_INVENTORY + extra
    result = service.handle_user_input(", ".join(inventory))
    assert result["status"] == "ok"
    assert result["user_ingredients"] == inventory
    assert result["results"][0]["title"] == "Scrambled Egg with Tomatoes"
    assert len(result["results"][0]["matched_user_items"]) == 3 + len(extra)
    ranked, _ = service.rerank(", ".join(inventory), top_n=70)
    salted = next(recipe for recipe in ranked if recipe["title"] == "Telur ayam asin")
    assert salted["ingredient_names"] == ["telur ayam", "garam", "air"]
    assert salted["source_ingredient_names"] == ["telur ayam", "garam", "toples air"]
    assert salted["matched_user_items"] == sorted(["telur"] + extra)
    assert is_relevant_enough(salted, inventory) is bool(extra)
    assert relevance_threshold(inventory) == 2
    assert all(recipe["final_score"] == pytest.approx(
        0.9 * recipe["sbert_score_norm"] + 0.1 * recipe["ingredient_efficiency"]
    ) for recipe in ranked)


def test_partial_matches_remain_available_and_exclusions_still_apply(monkeypatch, egg_corpus_candidates):
    partial = [recipe for recipe in egg_corpus_candidates if recipe["title"] in {"Tim Telur", "Telur ayam asin"}]
    service = corpus_service(monkeypatch, partial)
    result = service.handle_user_input(EGG_QUERY)
    assert [recipe["title"] for recipe in result["results"]] == ["Tim Telur"]
    service = corpus_service(monkeypatch, egg_corpus_candidates)
    filtered = service.handle_user_input(EGG_QUERY, excluded_ingredients=["bawang putih"])
    assert filtered["status"] == "ok"
    assert all("bawang putih" not in recipe["ingredient_names"] for recipe in filtered["results"])


def test_keyword_expansion_keeps_high_coverage_candidates_outside_the_semantic_pool():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.settings = Settings(top_k_sbert=50, max_keyword_extra=20)
    semantic = [scored_recipe(f"semantic-{index:02}", EGG_INVENTORY[:2] + ["air"], EGG_INVENTORY)
                for index in range(50)]
    for recipe in semantic:
        recipe.update(sbert_score=0.8, retrieval_source="semantic")
    extra = [scored_recipe(f"keyword-{index:02}", EGG_INVENTORY[:2] + ["garam"], EGG_INVENTORY)
             for index in range(30)]
    high = scored_recipe("high-coverage", EGG_INVENTORY, EGG_INVENTORY)
    documents = [(recipe["id"], recipe, "", np.array([0.1, 0.9])) for recipe in semantic + extra + [high]]
    retriever._semantic_candidates = lambda text: (np.array([1.0, 0.0]), deepcopy(semantic))
    retriever._get_all_documents = lambda: documents
    first = retriever.retrieve(EGG_QUERY, EGG_INVENTORY)
    assert len(first) == len({recipe["id"] for recipe in first}) == 70
    found = next(recipe for recipe in first if recipe["id"] == "high-coverage")
    assert found["retrieval_source"] == "keyword"
    assert found["keyword_matched_count"] == 4
    random.Random(42).shuffle(documents)
    assert retriever.retrieve(EGG_QUERY, EGG_INVENTORY) == first


def test_new_chat_resets_inventory_and_unconfirmed_staples_are_not_added(monkeypatch, tmp_path, egg_corpus_candidates):
    corpus_service(monkeypatch, egg_corpus_candidates)
    monkeypatch.setattr("dapuria.llama_service.LlamaService.available", lambda self: False)
    monkeypatch.setenv("DAPURIA_HISTORY_BACKEND", "sqlite")
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(tmp_path / "history.sqlite3"))
    st.cache_resource.clear()
    try:
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "app.py")).run()
        app.chat_input[0].set_value(EGG_QUERY).run(timeout=20)
        assert not app.exception
        assert app.session_state.inventory == EGG_INVENTORY
        assert app.session_state.pending
        app.chat_input[0].set_value("tidak ada").run(timeout=20)
        assert not app.exception
        assert app.session_state.inventory == EGG_INVENTORY
        first_chat = app.session_state.conversation_id
        app.button(key="continue_chat").click().run()
        app.chat_input[0].set_value("ada air").run(timeout=20)
        assert not app.exception
        assert app.session_state.inventory == EGG_INVENTORY + ["air"]
        next(button for button in app.button if button.label == "Mulai percakapan baru").click().run()
        assert app.session_state.inventory == []
        assert app.session_state.pending is None
        assert app.session_state.conversation_id != first_chat
        app.chat_input[0].set_value("ayam, bawang putih, kecap manis, tomat").run(timeout=20)
        assert not app.exception
        assert app.session_state.inventory == ["ayam", "bawang putih", "kecap manis", "tomat"]
    finally:
        st.cache_resource.clear()


def test_service_ranking_is_repeatable_without_llama(monkeypatch, egg_corpus_candidates):
    def unexpected_llama_call(*args, **kwargs):
        raise AssertionError("Ranking must not call Llama3")

    monkeypatch.setattr("dapuria.llama_service.LlamaService._chat", unexpected_llama_call)
    monkeypatch.setattr("dapuria.llama_service.LlamaService.available", unexpected_llama_call)
    service = corpus_service(monkeypatch, egg_corpus_candidates)
    expected = service.handle_user_input(EGG_QUERY)
    for seed in range(3):
        random.Random(seed).shuffle(egg_corpus_candidates)
        assert service.handle_user_input(EGG_QUERY) == expected
