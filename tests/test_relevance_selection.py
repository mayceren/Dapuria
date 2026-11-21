import csv
import json
import random
from copy import deepcopy
from itertools import permutations, product
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
from dapuria.reranking import coverage_stats, harmonic_efficiency, weighted_linear_score
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


@pytest.mark.parametrize(("better_use", "better_missing", "worse_use", "worse_missing"), [
    (4, 5, 4, 6), (4, 5, 3, 6), (4, 5, 3, 5), (4, 0, 4, 1),
])
@pytest.mark.parametrize("semantic_scores", list(product((0.0, 1.0), repeat=2)))
def test_dominance_precedes_wlc_regardless_of_semantic_advantage(
    better_use, better_missing, worse_use, worse_missing, semantic_scores,
):
    inventory = EGG_INVENTORY + ["bawang putih"]
    better = scored_recipe("better", inventory[:better_use] + [f"tambahan {i}" for i in range(better_missing)], inventory)
    worse = scored_recipe("worse", inventory[:worse_use] + [f"tambahan {i}" for i in range(worse_missing)], inventory)
    for recipe, semantic in zip((better, worse), semantic_scores):
        recipe["final_score"] = weighted_linear_score(semantic, recipe["ingredient_efficiency"], 0.9, 0.1)
    assert [row["id"] for row in select_recommendations([worse, better], 2)] == ["better", "worse"]


def test_four_matches_with_twelve_missing_do_not_automatically_beat_three_with_one_missing():
    inventory = EGG_INVENTORY + ["bawang putih"]
    costly = scored_recipe("costly", inventory[:4] + [f"tambahan {i}" for i in range(12)], inventory)
    practical = scored_recipe("practical", inventory[:3] + ["garam"], inventory)
    assert [row["id"] for row in select_recommendations([costly, practical], 2)] == ["practical", "costly"]
    assert practical["coverage_user"] == 0.6
    assert practical["coverage_recipe"] == 0.75


@pytest.mark.parametrize(("semantic_scores", "expected"), [((1.0, 0.0), "practical"), ((0.0, 1.0), "more-use")])
def test_wlc_still_influences_candidates_with_different_tradeoffs(semantic_scores, expected):
    inventory = EGG_INVENTORY + ["bawang putih"]
    practical = scored_recipe("practical", inventory[:3] + ["air", "garam"], inventory)
    more_use = scored_recipe("more-use", inventory[:4] + [f"tambahan {i}" for i in range(4)], inventory)
    for recipe, semantic in zip((practical, more_use), semantic_scores):
        recipe["final_score"] = weighted_linear_score(semantic, recipe["ingredient_efficiency"], 0.9, 0.1)
    assert select_recommendations([practical, more_use], 2)[0]["id"] == expected


def test_dominance_and_balanced_score_do_not_create_a_comparison_cycle():
    inventory = EGG_INVENTORY + ["bawang putih"]
    better = scored_recipe("better", inventory[:4] + [f"tambahan {i}" for i in range(5)], inventory)
    worse = scored_recipe("worse", inventory[:4] + [f"tambahan {i}" for i in range(6)], inventory)
    practical = scored_recipe("practical", inventory[:3] + ["garam"], inventory)
    for recipe, semantic in zip((better, worse, practical), (0.0, 1.0, 0.6)):
        recipe["final_score"] = weighted_linear_score(semantic, recipe["ingredient_efficiency"], 0.9, 0.1)
    for order in permutations((better, worse, practical)):
        assert [row["id"] for row in select_recommendations(list(order), 3)] == ["practical", "better", "worse"]


def test_diversity_removes_blockers_before_choosing_the_next_candidate():
    inventory = EGG_INVENTORY + ["bawang putih"]
    candidates = []
    for identifier, title, matched, missing in [
        ("first", "Telur Kecap Bombay", 5, 0),
        ("duplicate", "Telur bumbu kecap", 4, 1),
        ("balanced", "Sup Telur", 4, 2),
        ("ready", "Telur Panggang", 3, 0),
        ("next", "Semur Telur", 2, 0),
        ("last", "Pepes Telur", 2, 1),
    ]:
        recipe = scored_recipe(identifier, inventory[:matched] + [f"tambahan {i}" for i in range(missing)], inventory)
        recipe["title"] = title
        candidates.append(recipe)
    original = deepcopy(candidates)
    for seed in range(8):
        random.Random(seed).shuffle(candidates)
        assert [row["id"] for row in select_recommendations(candidates, 5)] == ["first", "balanced", "ready", "next", "last"]
    assert sorted(candidates, key=lambda row: row["id"]) == sorted(original, key=lambda row: row["id"])
    assert select_recommendations(original[:2], 5) == [original[0]]


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_full_order_respects_every_dominance_relation_and_is_independent_of_retrieval_order(seed):
    inventory = EGG_INVENTORY + ["bawang putih"]
    generator = random.Random(seed)
    candidates = []
    for index in range(30):
        matched, missing = generator.randint(2, 5), generator.randint(0, 12)
        recipe = scored_recipe(str(index), inventory[:matched] + [f"tambahan {i}" for i in range(missing)], inventory)
        recipe["final_score"] = weighted_linear_score(generator.random(), recipe["ingredient_efficiency"], 0.9, 0.1)
        candidates.append(recipe)
    original = deepcopy(candidates)
    ordered = select_recommendations(candidates, len(candidates))
    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1:]:
            assert not (
                len(later["matched_user_items"]) >= len(earlier["matched_user_items"])
                and len(later["missing_items"]) <= len(earlier["missing_items"])
                and (len(later["matched_user_items"]) > len(earlier["matched_user_items"])
                     or len(later["missing_items"]) < len(earlier["missing_items"]))
            )
    assert select_recommendations(candidates, 5) == ordered[:5]
    generator.shuffle(candidates)
    assert select_recommendations(candidates, len(candidates)) == ordered
    assert sorted(candidates, key=lambda row: row["id"]) == sorted(original, key=lambda row: row["id"])


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
