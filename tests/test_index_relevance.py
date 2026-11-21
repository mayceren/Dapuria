import os
import shutil
from dataclasses import replace

import pytest

from dapuria.config import Settings
from dapuria.recipe_service import RecipeService
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
    if inventory[0] == "telur":
        scrambled = next(recipe for recipe in ranked if recipe["title"] == "Scrambled Egg with Tomatoes")
        salted = next(recipe for recipe in ranked if recipe["title"] == "Telur ayam asin")
        assert len(scrambled["matched_user_items"]) == (4 if "air" in inventory else 3)
        assert result["results"][0]["id"] != salted["id"]
        assert salted["ingredient_names"] == ["telur ayam", "garam", "air"]
    if inventory[0] == "ayam":
        assert "ayam" in result["results"][0]["matched_user_items"]
