from __future__ import annotations

from .main_ingredients import has_main_ingredient


def validate_initial(
    user_ingredients: list[str], min_count: int = 1
) -> tuple[bool, str]:
    if len(user_ingredients) < min_count:
        return False, "no_ingredients"
    if not has_main_ingredient(user_ingredients):
        return False, "need_main_ingredient"
    return True, "ok"


def relevance_threshold(user_ingredients: list[str]) -> int:
    return min(2, len(user_ingredients)) if user_ingredients else 0


def is_relevant_enough(candidate: dict | None, user_ingredients: list[str]) -> bool:
    if not candidate or not user_ingredients:
        return False
    matched_user_count = len(candidate.get("matched_user_items", []))
    return (
        matched_user_count >= relevance_threshold(user_ingredients)
        and has_main_ingredient(candidate.get("matched_user_items", []))
    )
