from __future__ import annotations

from collections import Counter
from typing import Any

from .config import Settings
from .preferences import ingredient_is_excluded, update_preferences
from .preprocessing import ingredient_matches, parse_user_ingredients
from .recipe_data import normalize_recipe
from .recipe_details import RecipeDetails
from .reranking import (
    coverage_stats,
    harmonic_efficiency,
    minmax_normalize,
    weighted_linear_score,
)
from .retrieval import HybridRetriever
from .selection import select_recommendations
from .validation import is_relevant_enough, validate_initial

COMMON_STAPLES = [
    "garam",
    "gula",
    "minyak goreng",
    "bawang putih",
    "bawang merah",
    "merica",
    "kecap manis",
    "kecap asin",
    "kaldu",
    "air",
]


class RecipeService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.retriever = HybridRetriever(self.settings)
        self.details = RecipeDetails(self.settings.corpus_path)

    def healthcheck(self) -> dict[str, Any]:
        data = self.retriever.healthcheck()
        data["config"] = {
            "top_k_sbert": self.settings.top_k_sbert,
            "max_keyword_extra": self.settings.max_keyword_extra,
            "semantic_weight": self.settings.semantic_weight,
            "ingredient_weight": self.settings.ingredient_weight,
            "top_n": self.settings.top_n,
        }
        return data

    def rerank(
        self, user_text: str, top_n: int | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        top_n = top_n or self.settings.top_n
        user_ingredients = parse_user_ingredients(user_text)

        canonical_query = ", ".join(user_ingredients)

        candidates = [
            normalize_recipe(candidate)
            for candidate in self.retriever.retrieve(canonical_query, user_ingredients)
        ]

        if not candidates:
            return [], user_ingredients

        normalized = minmax_normalize([row["sbert_score"] for row in candidates])
        ranked: list[dict[str, Any]] = []
        for candidate, semantic_norm in zip(candidates, normalized):
            stats = coverage_stats(user_ingredients, candidate["ingredient_names"])
            efficiency = harmonic_efficiency(
                stats["coverage_recipe"], stats["coverage_user"]
            )
            final_score = weighted_linear_score(
                float(semantic_norm),
                efficiency,
                self.settings.semantic_weight,
                self.settings.ingredient_weight,
            )
            ranked.append(
                {
                    **candidate,
                    **stats,
                    "ingredient_efficiency": float(efficiency),
                    "sbert_score_norm": float(semantic_norm),
                    "final_score": float(final_score),
                }
            )
        ranked.sort(
            key=lambda row: (
                -row["final_score"],
                -row["sbert_score_norm"],
                str(row["id"]),
            )
        )
        return ranked[:top_n], user_ingredients

    def handle_user_input(
        self, user_text: str, top_n: int | None = None,
        excluded_ingredients: list[str] | None = None,
    ) -> dict[str, Any]:
        preferences = update_preferences(excluded_ingredients or [], user_text)
        excluded = preferences.excluded
        metadata = {"excluded_ingredients": excluded} if excluded else {}
        user_ingredients = [item for item in parse_user_ingredients(preferences.inventory_text)
                            if not ingredient_is_excluded(item, excluded)]
        valid, status = validate_initial(
            user_ingredients, self.settings.min_ingredient_count
        )
        if not valid and status == "no_ingredients":
            return {
                **metadata,
                "status": status,
                "user_ingredients": [],
                "message": "Belum ada bahan yang terdeteksi. Masukkan bahan yang kamu miliki, dipisahkan dengan koma.",
            }
        if not valid and status == "need_main_ingredient":
            return {
                **metadata,
                "status": status,
                "user_ingredients": user_ingredients,
                "message": (
                    f"Bahan yang disebutkan ({', '.join(user_ingredients)}) belum memuat bahan utama yang dikenali. "
                    "Tambahkan minimal satu bahan utama, misalnya ayam, telur, tahu, tempe, ikan, nasi, kentang, atau sayuran."
                ),
            }

        requested_top_n = top_n or self.settings.top_n

        # Ambil cukup banyak hasil sebelum final validation.
        candidate_limit = self.settings.top_k_sbert + self.settings.max_keyword_extra

        ranked_results, parsed = self.rerank(
            ", ".join(user_ingredients),
            top_n=candidate_limit,
        )

        # Validasi dilakukan pada setiap candidate,
        # bukan hanya candidate peringkat pertama.
        valid_results = [
            row for row in ranked_results
            if not any(
                ingredient_is_excluded(name, excluded)
                for name in row["ingredient_names"] + row.get("source_ingredient_names", [])
            )
            and is_relevant_enough(row, parsed)
        ]

        # Select practical, distinct dishes only after validating the whole pool.
        results = [
            self.details.restore_steps(row)
            for row in select_recommendations(valid_results, requested_top_n)
        ]

        if not results:
            return {
                **metadata,
                "status": "no_match",
                "user_ingredients": parsed,
                "message": (
                    "Sistem belum menemukan resep yang cukup cocok untuk kombinasi bahan "
                    f"({', '.join(parsed)}). "
                    + (f"Batasan bahan tetap diterapkan: {', '.join(excluded)}. " if excluded else "")
                    + "Coba tambahkan bahan lain atau gunakan kombinasi berbeda."
                ),
            }

        return {
            **metadata,
            "status": "ok",
            "user_ingredients": parsed,
            "results": results,
        }

    @staticmethod
    def build_grounded_payload(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "ok":
            return {
                "status": result.get("status"),
                "message": result.get("message", ""),
            }
        recipes = []
        for row in result["results"]:
            recipes.append(
                {
                    "recipe_id": row["id"],
                    "title": row["title"],
                    "ingredients": row["ingredient_names"],
                    "steps": row.get("steps_text", ""),
                    "matched_user_items": row["matched_user_items"],
                    "missing_items": row["missing_items"],
                }
            )
        return {
            "status": "ok",
            "user_ingredients": result["user_ingredients"],
            "recipes": recipes,
        }

    def clarification_candidates(
        self, result: dict[str, Any], max_items: int = 1
    ) -> list[str]:
        """Choose common pantry items that appear as missing in current Top-5.

        Candidate selection is deterministic. Only the highest-priority missing
        staple is asked at a time so a generic reply such as "ya" cannot be
        misinterpreted as confirmation of ingredients that were never asked.
        Llama 3 may verbalize the question, but it does not decide which
        ingredient is eligible.
        """
        if result.get("status") != "ok":
            return []
        user_items = result.get("user_ingredients", [])
        counts: Counter[str] = Counter()
        for recipe in result["results"]:
            for missing in recipe.get("missing_items", []):
                for staple in COMMON_STAPLES:
                    if ingredient_is_excluded(staple, result.get("excluded_ingredients", [])):
                        continue
                    if ingredient_matches(staple, missing) and not any(
                        ingredient_matches(staple, u) for u in user_items
                    ):
                        counts[staple] += 1
        return [name for name, _count in counts.most_common(max_items)]
