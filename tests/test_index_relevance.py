import os
import shutil
from dataclasses import replace

import pytest

from dapuria.config import Settings
from dapuria.recipe_service import RecipeService
from dapuria.reranking import coverage_stats
from dapuria.selection import select_recommendations
from dapuria.validation import is_relevant_enough

pytestmark = pytest.mark.skipif(os.getenv("DAPURIA_RUN_INDEX_TESTS") != "1",
                                reason="Set DAPURIA_RUN_INDEX_TESTS=1 to test the bundled Chroma index and cached SBERT model")


@pytest.fixture(scope="module")
def indexed_service(tmp_path_factory):
    settings = Settings()
    index = tmp_path_factory.mktemp("dapuria-index-relevance") / "index"
    shutil.copytree(settings.chroma_dir, index)
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("HF_HUB_OFFLINE", "1")
        environment.setenv("TRANSFORMERS_OFFLINE", "1")
        environment.setenv("ANONYMIZED_TELEMETRY", "False")
        yield RecipeService(replace(settings, chroma_dir=index))


@pytest.mark.parametrize("query", [
    "2 butir telur, kecap asin, tomat, seledri",
    "2 butir telur, kecap asin, tomat, seledri, air",
    "ayam, bawang putih, kecap manis, tomat",
    "tahu, tempe, bawang putih, kecap manis",
])
def test_real_hybrid_pool_and_combination_relevance(indexed_service, query):
    ranked, inventory = indexed_service.rerank(query, top_n=70)
    assert len(ranked) <= 70
    assert len(ranked) == len({recipe["id"] for recipe in ranked})
    assert {recipe["retrieval_source"] for recipe in ranked} == {"semantic", "keyword"}
    result = indexed_service.handle_user_input(query)
    assert result["status"] == "ok"
    assert len(result["results"][0]["matched_user_items"]) >= 3
    assert all(is_relevant_enough(recipe, inventory) for recipe in result["results"])
    for index, earlier in enumerate(result["results"]):
        stats = coverage_stats(inventory, earlier["ingredient_names"])
        assert {key: earlier[key] for key in stats} == stats
        for later in result["results"][index + 1:]:
            assert not (
                len(later["matched_user_items"]) >= len(earlier["matched_user_items"])
                and len(later["missing_items"]) <= len(earlier["missing_items"])
                and (len(later["matched_user_items"]) > len(earlier["matched_user_items"])
                     or len(later["missing_items"]) < len(earlier["missing_items"]))
            )
    if inventory[0] == "telur":
        scrambled = next(recipe for recipe in ranked if recipe["title"] == "Scrambled Egg with Tomatoes")
        salted = next(recipe for recipe in ranked if recipe["title"] == "Telur ayam asin")
        assert len(scrambled["matched_user_items"]) == (4 if "air" in inventory else 3)
        assert result["results"][0]["id"] != salted["id"]
        assert salted["ingredient_names"] == ["telur ayam", "garam", "air"]
    if inventory[0] == "ayam":
        assert "ayam" in result["results"][0]["matched_user_items"]


def test_real_egg_pair_and_full_pool_respect_dominance(indexed_service):
    query = "2 butir telur, kecap asin, tomat, seledri, bawang putih"
    ranked, inventory = indexed_service.rerank(query, top_n=70)
    recipes = {recipe["id"]: recipe for recipe in ranked}
    kecap, dadar = recipes["resep_135"], recipes["resep_2235"]
    assert kecap["title"] == "Telur Kecap"
    assert dadar["title"] == "Telur Dadar Ala Saya"
    assert kecap["num_ingredients"] == 10 and dadar["num_ingredients"] == 9
    assert len(kecap["matched_user_items"]) == len(dadar["matched_user_items"]) == 4
    assert len(kecap["missing_items"]) == 6 and len(dadar["missing_items"]) == 5
    assert kecap["sbert_score"] > dadar["sbert_score"]
    assert kecap["final_score"] > dadar["final_score"]
    assert select_recommendations([kecap, dadar], 2) == [dadar, kecap]
    valid = [recipe for recipe in ranked if is_relevant_enough(recipe, inventory)]
    expected = select_recommendations(valid, 5)
    result = indexed_service.handle_user_input(query)
    assert result["status"] == "ok"
    assert [recipe["id"] for recipe in result["results"]] == [recipe["id"] for recipe in expected]
    reversed_order = select_recommendations(list(reversed(valid)), 5)
    assert expected == reversed_order
    if kecap in expected and dadar in expected:
        assert expected.index(dadar) < expected.index(kecap)
