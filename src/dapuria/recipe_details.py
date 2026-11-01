"""Restore legacy truncated instructions without changing retrieval or scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RecipeDetails:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self._steps: dict[tuple[str, tuple[str, ...]], set[str]] | None = None

    def restore_steps(self, recipe: dict[str, Any]) -> dict[str, Any]:
        indexed_steps = recipe.get("steps_text", "")
        if not isinstance(indexed_steps, str) or len(indexed_steps) != 2000:
            return recipe
        if self._steps is None:
            steps_by_recipe: dict[tuple[str, tuple[str, ...]], set[str]] = {}
            if self.corpus_path.is_file():
                with self.corpus_path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        steps = row.get("steps_text") or ""
                        if len(steps) <= 2000:
                            continue
                        try:
                            names = json.loads(row.get("ingredient_names") or "[]")
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(names, list) or not all(
                            isinstance(name, str) for name in names
                        ):
                            continue
                        key = (row.get("title_clean", ""), tuple(names))
                        steps_by_recipe.setdefault(key, set()).add(steps)
            self._steps = steps_by_recipe

        key = (
            recipe.get("source_title", recipe.get("title", "")),
            tuple(recipe.get("source_ingredient_names", recipe.get("ingredient_names", []))),
        )
        matches = [
            steps for steps in self._steps.get(key, ())
            if steps.startswith(indexed_steps)
        ]
        if len(matches) == 1:
            return {**recipe, "steps_text": matches[0]}
        return recipe
