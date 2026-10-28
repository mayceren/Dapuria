from __future__ import annotations

from typing import Any
import numpy as np

from .preprocessing import ingredient_matches


def coverage_stats(
    user_ingredients: list[str], recipe_ingredients: list[str]
) -> dict[str, Any]:
    if not recipe_ingredients:
        return {
            "coverage_recipe": 0.0,
            "coverage_user": 0.0,
            "matched_recipe_items": [],
            "matched_user_items": [],
            "missing_items": [],
        }

    matched_recipe_items: list[str] = []
    matched_user_items: set[str] = set()
    for recipe_name in recipe_ingredients:
        matched = False
        for user_name in user_ingredients:
            if ingredient_matches(user_name, recipe_name):
                matched = True
                matched_user_items.add(user_name)
        if matched:
            matched_recipe_items.append(recipe_name)

    missing_items = [
        name for name in recipe_ingredients if name not in matched_recipe_items
    ]
    return {
        "coverage_recipe": len(matched_recipe_items) / len(recipe_ingredients),
        "coverage_user": len(matched_user_items) / len(user_ingredients)
        if user_ingredients
        else 0.0,
        "matched_recipe_items": matched_recipe_items,
        "matched_user_items": sorted(matched_user_items),
        "missing_items": missing_items,
    }


def harmonic_efficiency(coverage_recipe: float, coverage_user: float) -> float:
    denominator = coverage_recipe + coverage_user
    return (
        0.0 if denominator == 0 else (2 * coverage_recipe * coverage_user) / denominator
    )


def minmax_normalize(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    low = float(arr.min())
    high = float(arr.max())
    # Preserve the frozen production implementation (+1e-9 denominator guard).
    return (arr - low) / (high - low + 1e-9)


def weighted_linear_score(
    semantic_norm: float,
    ingredient_efficiency: float,
    semantic_weight: float,
    ingredient_weight: float,
) -> float:
    return semantic_weight * semantic_norm + ingredient_weight * ingredient_efficiency
