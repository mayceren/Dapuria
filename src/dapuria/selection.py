"""Select practical, distinct dishes from candidates already scored by WLC."""

from __future__ import annotations

import re
from typing import Any

from .preprocessing import normalize_ingredient_name

TITLE_FILLERS = frozenset({
    "resep", "bumbu", "sederhana", "simple", "simpel", "praktis",
    "spesial", "special", "rumahan", "enak", "mudah",
})
DISH_STYLES = frozenset({
    "sate", "sup", "sop", "soto", "bakso", "nugget", "katsu", "kari",
    "gulai", "rendang", "tongseng", "semur", "bakar", "panggang",
    "goreng", "tumis", "kukus", "pepes", "opor", "teriyaki",
})


def _title_words(recipe: dict[str, Any]) -> set[str]:
    title = normalize_ingredient_name(recipe.get("title", ""))
    return set(re.findall(r"\w+", title)) - TITLE_FILLERS


def _same_dish(first: set[str], second: set[str]) -> bool:
    if not first or not second:
        return False
    if first == second:
        return True
    if (first & DISH_STYLES) != (second & DISH_STYLES):
        return False
    common = first & second
    # Group short-title variations, but preserve distinct cooking styles.
    return len(common) >= 2 and len(common) / min(len(first), len(second)) >= 2 / 3


def _selection_key(recipe: dict[str, Any]) -> tuple:
    return (
        -selection_score(recipe),
        -recipe["coverage_user"],
        -recipe["coverage_recipe"],
        -len(recipe["matched_user_items"]),
        len(recipe["missing_items"]),
        -recipe["ingredient_efficiency"],
        -recipe["final_score"],
        str(recipe["id"]),
    )


def _dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_matched, second_matched = len(first["matched_user_items"]), len(second["matched_user_items"])
    first_missing, second_missing = len(first["missing_items"]), len(second["missing_items"])
    return (first_matched >= second_matched and first_missing <= second_missing
            and (first_matched > second_matched or first_missing < second_missing))


def select_recommendations(
    candidates: list[dict[str, Any]], limit: int,
) -> list[dict[str, Any]]:
    """Balance inventory coverage, recipe readiness, and the unchanged WLC score."""
    remaining = list(candidates)
    selected = []
    while remaining and len(selected) < limit:
        eligible = [recipe for recipe in remaining
                    if not any(_dominates(other, recipe) for other in remaining)]
        recipe = min(eligible, key=_selection_key)
        words = _title_words(recipe)
        selected.append(recipe)
        remaining = [other for other in remaining
                     if other is not recipe and not _same_dish(words, _title_words(other))]
    return selected


def selection_score(recipe: dict[str, Any]) -> float:
    return recipe["coverage_user"] * recipe["ingredient_efficiency"] * (1 + recipe["final_score"])
