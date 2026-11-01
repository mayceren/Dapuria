"""Normalize recipe metadata while retaining its original corpus identity."""

from __future__ import annotations

import re
from typing import Any

from .preprocessing import parse_ingredient

TITLE_NUMBER = re.compile(r"^\s*(?:\(\d+\)|\d+[.)])\s+")
INGREDIENT_CONJUNCTION = re.compile(r"\s+(?:dan|&|\+)\s+", re.IGNORECASE)
SEPARATE_SEASONINGS = frozenset({"kecap", "garam", "gula", "merica", "lada", "penyedap"})


def recipe_title(title: str) -> str:
    """Remove source list numbering, retaining meaningful numbers in dish names."""
    return TITLE_NUMBER.sub("", str(title or "")).strip()


def recipe_ingredients(names: list[str]) -> list[str]:
    """Expand explicit ingredient lists and discard empty or numeric-only items."""
    ingredients = []
    for raw in names:
        name = parse_ingredient(raw)["name"]
        parts = INGREDIENT_CONJUNCTION.split(name)
        # Some source lines join separate seasonings without punctuation.
        if len(parts) == 1 and len(name.split()) > 1 and set(name.split()) <= SEPARATE_SEASONINGS:
            parts = name.split()
        for part in parts:
            normalized = parse_ingredient(part)["name"]
            if normalized and normalized not in ingredients:
                ingredients.append(normalized)
    return ingredients


def normalize_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Return clean metadata without mutating the source or its stored recipe ID."""
    title = recipe.get("title", "")
    names = recipe.get("ingredient_names", [])
    ingredients = recipe_ingredients(names)
    return {
        **recipe,
        "source_title": recipe.get("source_title", title),
        "source_ingredient_names": recipe.get("source_ingredient_names", list(names)),
        "title": recipe_title(title),
        "ingredient_names": ingredients,
        "num_ingredients": len(ingredients),
    }
