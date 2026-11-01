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


def select_recommendations(
    candidates: list[dict[str, Any]], limit: int,
) -> list[dict[str, Any]]:
    """Prefer fewer missing ingredients, using efficiency and WLC to break ties.

    Callers first enforce relevance, main-ingredient availability, and exclusions.
    WLC scores remain unchanged; this is the final recommendation selection stage.
    """
    if limit < 1:
        return []
    ordered = sorted(candidates, key=lambda recipe: (
        len(recipe["missing_items"]),
        -recipe["ingredient_efficiency"],
        -recipe["final_score"],
        str(recipe["id"]),
    ))
    selected = []
    titles: list[set[str]] = []
    for recipe in ordered:
        words = _title_words(recipe)
        if any(_same_dish(words, previous) for previous in titles):
            continue
        selected.append(recipe)
        titles.append(words)
        if len(selected) == limit:
            break
    return selected
