"""Regressions for ingredient identity, practical selection, and recipe provenance."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from dapuria.config import Settings
from dapuria.preprocessing import (
    ingredient_matches,
    parse_ingredient,
    parse_user_ingredients,
)
from dapuria.recipe_data import normalize_recipe, recipe_title
from dapuria.recipe_details import RecipeDetails
from dapuria.recipe_service import RecipeService
from dapuria.reranking import coverage_stats
from dapuria.selection import select_recommendations

QUERY = "dada ayam 250 gr, kecap, garam, saus tiram, bawang putih, bawang bombay"
INVENTORY = ["dada ayam", "kecap", "garam", "saus tiram", "bawang putih", "bawang bombay"]


@pytest.mark.parametrize("text", [
    QUERY, QUERY.replace("dada ayam 250 gr", "250 gr dada ayam"),
    QUERY.replace("dada ayam 250 gr", "dada ayam 0,25 kg"),
])
def test_quantities_do_not_change_the_inventory(text):
    assert parse_user_ingredients(text) == INVENTORY


@pytest.mark.parametrize(("raw", "expected"), [
    ("dada ayam iris tipis/kecil2 sesuai selera", "dada ayam"),
    ("ayam potong", "ayam"), ("3 siung baput", "bawang putih"),
    ("5 siung bamer", "bawang merah"), ("bawang bombai", "bawang bombay"),
    ("buah naga", "buah naga"), ("bawang goreng", "bawang goreng"),
    ("dada ayam kampung (rebus", "dada ayam kampung"), ("mil air", "air"),
    ("pcs saus tiram", "saus tiram"), ("0,5 kg ayam", "ayam"),
    ("1", ""), ("1/2", ""), ("250 gr", ""),
])
def test_ingredient_names_exclude_amounts_and_preparation(raw, expected):
    assert parse_ingredient(raw)["name"] == expected


@pytest.mark.parametrize(("available", "required", "matches"), [
    ("dada ayam", "dada ayam iris tipis/kecil2 sesuai selera", True),
    ("dada ayam", "ayam potong", True), ("bawang putih", "baput", True),
    ("bawang putih", "bamer", False), ("bawang putih", "baputang", False),
    ("ayam", "kaldu ayam", False), ("kaldu ayam", "ayam", False),
    ("udang", "terasi udang", False), ("kaldu", "bubuk kaldu ayam", True),
    ("teri", "terigu", False),
    ("bawang putih", "bawang putih bubuk", False),
    ("bawang putih bubuk", "bawang putih", False),
    ("bawang putih", "bawang putih powder", False),
    ("bawang putih powder", "bawang putih", False),
    ("bawang putih", "bawang putih (bubuk)", False),
    ("bawang putih (powder)", "bawang putih", False),
    ("bawang putih", "garlic powder", False),
    ("garlic powder", "bawang putih", False),
    ("garlic powder", "bawang putih bubuk", True),
    ("bawang putih bubuk", "bubuk bawang putih", True),
    ("bubuk bawang putih", "bawang putih bubuk", True),
    ("powder bawang putih", "bawang putih bubuk", True),
    ("bawang putih bubuk", "bawang putih (bubuk desaku)", True),
    ("bawang putih", "bawang putih dihaluskan", True),
    ("gula", "gula", True), ("gula", "gula pasir", True),
    ("gula pasir", "gula", True), ("gula", "gula pasir rose brand", True),
    ("gula", "gula merah", False), ("gula merah", "gula", False),
    ("gula", "gula aren", False), ("gula aren", "gula", False),
    ("gula", "gula jawa", False), ("gula", "gula palem", False),
    ("gula", "gula halus", False), ("gula", "gula bubuk", False),
])
def test_matching_preserves_alias_and_product_boundaries(available, required, matches):
    assert ingredient_matches(available, required) is matches


def test_negation_and_conversation_inventory_are_preserved():
    assert parse_user_ingredients("aku punya nasi tapi telur habis") == ["nasi"]
    assert parse_user_ingredients("aku tidak punya nasi dan telur") == []
    assert parse_user_ingredients("aku punya telor sama telur dan baput") == ["telur", "bawang putih"]


@pytest.mark.parametrize(("source", "expected"), [
    ("826) Ayam Kecap Bombay", "Ayam Kecap Bombay"),
    ("12. Ayam Goreng", "Ayam Goreng"),
    ("3 Bahan Ayam", "3 Bahan Ayam"), ("1.5 kg Ayam", "1.5 kg Ayam"),
])
def test_only_source_list_numbering_is_removed_from_titles(source, expected):
    assert recipe_title(source) == expected


def test_normalization_is_idempotent_and_keeps_original_recipe_identity():
    source = {"id": "original", "title": "826) Ayam Kecap Bombay", "ingredient_names": [
        "ayam potong", "baput", "bawang putih", "1", "garam dan lada bubuk",
    ]}
    original = deepcopy(source)
    recipe = normalize_recipe(source)
    assert source == original
    assert recipe["id"] == "original"
    assert recipe["ingredient_names"] == ["ayam", "bawang putih", "garam", "lada bubuk"]
    assert recipe["source_title"] == original["title"]
    assert recipe["source_ingredient_names"] == original["ingredient_names"]
    assert normalize_recipe(recipe) == recipe


@pytest.fixture
def corpus_candidates():
    corpus = Path(__file__).resolve().parents[1] / "data/corpus/recipes.csv"
    selected = {217, 4162, 4559, 4815}
    with corpus.open(encoding="utf-8-sig", newline="") as handle:
        return [{
            "id": f"recipe-{index}", "title": row["title_clean"],
            "ingredient_names": json.loads(row["ingredient_names"]),
            "steps_text": row["steps_text"], "sbert_score": 0.99 - order * 0.04,
        } for order, (index, row) in enumerate(
            pair for pair in enumerate(csv.DictReader(handle)) if pair[0] in selected
        )]


def service_with_candidates(monkeypatch, candidates):
    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            assert ingredients == INVENTORY
            assert query == ", ".join(INVENTORY)
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    return RecipeService(Settings())


def test_real_corpus_examples_have_correct_ingredients_and_counts(monkeypatch, corpus_candidates):
    service = service_with_candidates(monkeypatch, corpus_candidates)
    ranked, parsed = service.rerank(QUERY, top_n=70)
    assert parsed == INVENTORY
    recipes = {recipe["id"]: recipe for recipe in ranked}
    alias_recipe = recipes["recipe-217"]
    assert alias_recipe["matched_recipe_items"] == ["dada ayam", "bawang putih", "garam", "kecap"]
    assert alias_recipe["missing_items"] == ["bawang merah", "tomat", "penyedap rasa", "gula"]
    bombay = recipes["recipe-4815"]
    assert bombay["title"] == "Ayam Kecap Bombay"
    assert bombay["num_ingredients"] == 7
    assert len(bombay["matched_recipe_items"]) == 6
    assert bombay["missing_items"] == ["air"]
    for recipe in ranked:
        assert recipe["final_score"] == pytest.approx(
            0.9 * recipe["sbert_score_norm"] + 0.1 * recipe["ingredient_efficiency"]
        )


def test_selection_prefers_readiness_and_does_not_recommend_only_matching_seasonings(
    monkeypatch, corpus_candidates,
):
    # Give the impractical sate the highest semantic score deliberately.
    for candidate in corpus_candidates:
        candidate["sbert_score"] = 1.0 if candidate["id"] == "recipe-4162" else 0.7
    corpus_candidates.append({
        "id": "seasonings-only", "title": "Sambal Kecap", "sbert_score": 0.9,
        "ingredient_names": ["kecap", "garam", "bawang putih"], "steps_text": "Campurkan.",
    })
    service = service_with_candidates(monkeypatch, corpus_candidates)
    ranked, _ = service.rerank(QUERY, top_n=70)
    assert ranked[0]["id"] == "recipe-4162"
    result = service.handle_user_input(QUERY)
    assert result["status"] == "ok"
    assert result["results"][0]["id"] == "recipe-4815"
    assert [recipe["id"] for recipe in result["results"]] == ["recipe-4815", "recipe-4162"]


def test_excluded_ingredients_are_filtered_before_practical_selection(monkeypatch, corpus_candidates):
    service = service_with_candidates(monkeypatch, corpus_candidates)
    result = service.handle_user_input(QUERY, excluded_ingredients=["kemiri"])
    assert result["status"] == "ok"
    assert all("kemiri" not in recipe["ingredient_names"] for recipe in result["results"])


def test_exclusions_still_check_original_ingredient_text_after_cleaning(monkeypatch):
    candidate = {
        "id": "with-shrimp", "title": "Ayam Campur", "sbert_score": 0.9,
        "ingredient_names": ["ayam potong campur udang", "garam", "kecap"],
        "steps_text": "Masak.",
    }
    service = service_with_candidates(monkeypatch, [candidate])
    assert service.handle_user_input(QUERY, excluded_ingredients=["udang"])["status"] == "no_match"


def test_diversity_keeps_distinct_dishes_and_selection_does_not_mutate_scores():
    titles = ["Ayam Kecap Bombay", "Ayam bumbu kecap", "Ayam Kecap", "Sate Ayam Kecap",
              "Ayam Saus Tiram", "Semur Ayam", "Ayam Panggang"]
    candidates = [{
        "id": str(i), "title": title, "missing_items": ["air"] * i,
        "ingredient_efficiency": 0.8, "final_score": 0.95 - i * 0.01,
    } for i, title in enumerate(titles)]
    original = deepcopy(candidates)
    result = select_recommendations(candidates, 5)
    assert [recipe["title"] for recipe in result] == [titles[i] for i in (0, 3, 4, 5, 6)]
    assert candidates == original
    assert select_recommendations(candidates, 0) == []


def test_sugar_and_powder_matching_preserves_inventory_and_original_recipe_labels():
    inventory = parse_user_ingredients("ayam, bawang putih, gula")
    recipe = normalize_recipe({"ingredient_names": [
        "ayam", "bawang putih bubuk", "gula", "gula pasir", "gula merah", "gula aren",
    ]})
    stats = coverage_stats(inventory, recipe["ingredient_names"])
    assert inventory == ["ayam", "bawang putih", "gula"]
    assert recipe["ingredient_names"] == [
        "ayam", "bawang putih bubuk", "gula", "gula pasir", "gula merah", "gula aren",
    ]
    assert stats["matched_recipe_items"] == ["ayam", "gula", "gula pasir"]
    assert stats["missing_items"] == ["bawang putih bubuk", "gula merah", "gula aren"]
    assert parse_user_ingredients("baking powder") == ["baking powder"]


def test_corrected_ingredient_counts_put_the_fewest_missing_ingredients_first(monkeypatch):
    inventory = ["ayam", "bawang putih", "gula"]
    candidates = [{
        "id": name, "title": title, "ingredient_names": names, "sbert_score": score,
        "steps_text": "Masak hingga matang.",
    } for name, title, names, score in [
        ("powder", "Sup Ayam", ["ayam", "bawang putih bubuk", "gula pasir"], 0.7),
        ("palm", "Semur Ayam", ["ayam", "bawang putih", "gula merah"], 0.8),
        ("many", "Ayam Panggang", ["ayam", "bawang putih", "bawang putih bubuk", "gula aren", "garam"], 0.99),
        ("ready", "Ayam Kukus", ["ayam", "bawang putih", "gula"], 0.1),
    ]]

    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            assert ingredients == inventory
            return deepcopy(candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    service = RecipeService(Settings())
    ranked, _ = service.rerank(", ".join(inventory), top_n=4)
    assert ranked[0]["id"] == "many"
    result = service.handle_user_input(", ".join(inventory))
    assert result["status"] == "ok"
    recipes = result["results"]
    assert recipes[0]["id"] == "ready"
    assert [len(recipe["missing_items"]) for recipe in recipes] == [0, 1, 1, 3]
    assert next(recipe for recipe in recipes if recipe["id"] == "powder")["missing_items"] == ["bawang putih bubuk"]
    assert next(recipe for recipe in recipes if recipe["id"] == "palm")["missing_items"] == ["gula merah"]
    assert all(recipe["final_score"] == pytest.approx(
        0.9 * recipe["sbert_score_norm"] + 0.1 * recipe["ingredient_efficiency"]
    ) for recipe in recipes)


def test_full_steps_can_still_be_restored_after_cleaning_metadata(tmp_path):
    corpus = tmp_path / "recipes.csv"
    source = {"title": "826) Ayam Kecap Bombay", "ingredient_names": ["ayam potong", "baput", "1"]}
    steps = "Siapkan bahan. " * 200
    with corpus.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title_clean", "ingredient_names", "steps_text"])
        writer.writeheader()
        writer.writerow({"title_clean": source["title"],
                         "ingredient_names": json.dumps(source["ingredient_names"]), "steps_text": steps})
    recipe = normalize_recipe({**source, "steps_text": steps[:2000]})
    restored = RecipeDetails(corpus).restore_steps(recipe)
    assert restored["steps_text"] == steps
    assert restored["title"] == "Ayam Kecap Bombay"


def test_chat_flow_keeps_added_ingredients_and_renders_clean_recommendations(
    monkeypatch, tmp_path, corpus_candidates,
):
    from streamlit.testing.v1 import AppTest

    class Retriever:
        def __init__(self, settings):
            pass

        def retrieve(self, query, ingredients):
            return deepcopy(corpus_candidates)

    monkeypatch.setattr("dapuria.recipe_service.HybridRetriever", Retriever)
    monkeypatch.setattr("dapuria.llama_service.LlamaService.available", lambda self: False)
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(tmp_path / "history.sqlite3"))
    monkeypatch.setenv("DAPURIA_HISTORY_BACKEND", "sqlite")
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py")).run()
    app.chat_input[0].set_value("dada ayam 250 gr, kecap, garam, saus tiram").run(timeout=20)
    assert not app.exception
    assert app.session_state.phase == "await_confirmation"
    assert "jenis kecap lainnya" in app.session_state.messages[-1]["content"]
    app.chat_input[0].set_value("kecap manis, ada bawang putih dan bawang bombay").run(timeout=20)
    assert not app.exception
    assert set(app.session_state.inventory) == set(INVENTORY) - {"kecap"} | {"kecap manis"}
    result = app.session_state.latest_result
    assert result["results"][0]["title"] == "Ayam Kecap Bombay"
    assert result["results"][0]["missing_items"] == ["air"]
    html = "\n".join(element.value for element in app.markdown)
    assert "6 dari 7" in html
    assert "826) Ayam Kecap Bombay" not in html
    assert "<li>1</li>" not in html
