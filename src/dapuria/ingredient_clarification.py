"""Resolve unspecified ingredient types before recipe retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pantry import ADD, REMOVE, REPLACE, RESET, update_inventory
from .preprocessing import (
    EXPLICIT_AVAILABILITY,
    parse_user_ingredients,
    strip_reply_particles,
)


@dataclass(frozen=True)
class TypeClarification:
    ingredient: str
    choices: tuple[str, ...]
    question: str
    example: str


TYPE_CLARIFICATIONS = {
    "soy_sauce_type": TypeClarification(
        ingredient="kecap",
        choices=("kecap manis", "kecap asin"),
        question=("Kecap yang kamu punya jenis apa: **kecap manis**, **kecap asin**, "
                  "atau **jenis kecap lainnya**? Kalau jenis lain, sebutkan namanya, ya."),
        example="kecap ikan",
    ),
    "onion_type": TypeClarification(
        ingredient="bawang",
        choices=("bawang putih", "bawang merah", "bawang bombay"),
        question=("Bawang yang kamu maksud jenis apa: **bawang putih**, **bawang merah**, "
                  "**bawang bombay**, atau **jenis bawang lainnya**? "
                  "Kalau jenis lain, sebutkan namanya, ya."),
        example="daun bawang",
    ),
}
REPLY_FRAMING = re.compile(
    r"(?:(?:aku|saya|gue|gua|iya|ya|yang|jenis|punya|memiliki|ada(?:nya)?|tersedia|cuma|hanya|juga"
    r"|tidak|nggak|enggak|gak|ga|ngga|belum|habis|bukan|tanpa"
    r"|mungkin|sepertinya|kayaknya)\s*)*", re.IGNORECASE,
)


def expand_type_reply(message: str, kind: str) -> str:
    """Expand contextual short answers without renaming foods like ikan asin."""
    clarification = TYPE_CLARIFICATIONS[kind]
    ingredient = re.escape(clarification.ingredient)
    names = {name.removeprefix(clarification.ingredient + " "): name for name in clarification.choices}
    if kind == "onion_type":
        names.update({"bombai": "bawang bombay", "putih bubuk": "bawang putih bubuk",
                      "putih powder": "bawang putih bubuk"})
    all_choices = " dan ".join(clarification.choices)
    names["semuanya"] = all_choices
    if len(clarification.choices) == 2:
        names.update(dict.fromkeys(("dua-duanya", "dua duanya", "duaduanya", "keduanya"), all_choices))
    else:
        names.update(dict.fromkeys(("ketiganya", "tiga-tiganya"), all_choices))
    type_word = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b",
                           re.IGNORECASE)
    segments = re.split(
        r"([,;.!?]|\b(?:dan|sama|serta|tapi|tetapi|namun|sedangkan)\b)",
        message, flags=re.IGNORECASE,
    )
    for index in range(0, len(segments), 2):
        segment = re.sub(rf"\b{ingredient}nya\b", clarification.ingredient, segments[index], flags=re.IGNORECASE)
        framed = strip_reply_particles(segment.strip())
        if type_word.search(framed) and REPLY_FRAMING.fullmatch(type_word.sub("", framed).strip()):
            segment = type_word.sub(lambda match: names[match.group().lower()], segment)
        segment = re.sub(rf"\b(?:yang|jenis)\s+(?={ingredient}\b)", "", segment, flags=re.IGNORECASE)
        segments[index] = re.sub(rf"^(\s*)(?:iya|ya)\s+(?={ingredient}\b)", r"\1", segment, flags=re.IGNORECASE)
    return "".join(segments)


def _specific_type(item: str, ingredient: str) -> bool:
    return bool(re.fullmatch(rf"{re.escape(ingredient)}\s+(?!lain(?:nya)?\b|jenis\b).+", item)) or (
        ingredient == "bawang" and item == "daun bawang"
    )


def refine_type_inventory(inventory: list[str]) -> list[str]:
    """A named type replaces its redundant, overly broad ingredient entry."""
    resolved = {
        clarification.ingredient for clarification in TYPE_CLARIFICATIONS.values()
        if any(_specific_type(item, clarification.ingredient) for item in inventory)
    }
    return [item for item in inventory if item not in resolved]


def update_type_inventory(inventory: list[str], message: str, kind: str) -> list[str]:
    """Keep a type question pending until a type or an inventory edit is given."""
    ingredient = TYPE_CLARIFICATIONS[kind].ingredient
    # Questions and bare fillers must not become owned ingredients.
    if "?" in message or re.search(
        r"\b(?:apa|apakah|gimana|bagaimana|maksudnya|kenapa|lupa)\b", message, re.IGNORECASE,
    ) or not (
        re.search(rf"\b{re.escape(ingredient)}\b", message, re.IGNORECASE)
        or any(_specific_type(item, ingredient) for item in parse_user_ingredients(message))
        or EXPLICIT_AVAILABILITY.search(message)
        or any(pattern.match(message) for pattern in (ADD, REMOVE, REPLACE, RESET))
    ):
        return inventory
    updated = [item for item in update_inventory(inventory, message)
               if not re.fullmatch(rf"(?:jenis\s+)?{re.escape(ingredient)}\s+(?:lain(?:nya)?|jenis lain)", item)]
    return refine_type_inventory(updated)
