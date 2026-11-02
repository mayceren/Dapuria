"""Deterministic conversation inventory edits, independent of models and ranking."""

from __future__ import annotations

import re

from .preprocessing import (
    EXPLICIT_AVAILABILITY,
    UNAVAILABLE,
    UNCERTAIN,
    USER_SEPARATORS,
    ingredient_matches,
    parse_user_ingredients,
    user_ingredient_text,
)

RESET = re.compile(
    r"^(?:mulai (?:lagi|dari awal)|ulang dari awal|reset(?: bahan)?|kosongkan(?: bahan)?)\b",
    re.IGNORECASE,
)
ADD = re.compile(r"^(?:tolong\s+)?(?:tambah(?:kan|in)?|masukkan)\s+", re.IGNORECASE)
REMOVE = re.compile(r"^(?:tolong\s+)?(?:hapus(?:kan)?|buang|kurangi)\s+", re.IGNORECASE)
REPLACE = re.compile(
    r"^(?:tolong\s+)?(?:ganti|ubah)\s+(.+?)\s+(?:dengan|menjadi|jadi|ke)\s+(.+)$",
    re.IGNORECASE,
)


def _names(text: str) -> list[str]:
    text = re.sub(r"\b(?!punya\b)(\w+)nya\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ternyata|sudah|udah|habis|abis|lagi)\b", "", text, flags=re.IGNORECASE)
    return parse_user_ingredients(text)


def _remove(items: list[str], targets: list[str]) -> list[str]:
    # Removing 'ayam' removes 'dada ayam', but removing 'dada ayam' must not
    # silently remove a less specific inventory item 'ayam'. Product gates apply.
    return [item for item in items if not any(
        ingredient_matches(target, item)
        and re.search(rf"(?<!\w){re.escape(target)}(?!\w)", item)
        for target in targets
    )]


def update_inventory(current: list[str], message: str) -> list[str]:
    """Apply additions, explicit removals, replacements, and reset in text order.

    Bare lists add to the current inventory. Only an explicit reset discards it.
    Negated lists stay in removal mode until a contrast or ownership statement.
    """
    items = list(dict.fromkeys(name for item in current for name in parse_user_ingredients(item)))
    text = message.strip()
    if re.fullmatch(r"(?:cari(?:kan)?(?: resep)?(?: lagi)?|lanjut(?:kan)?|coba lagi)[.!?]*", text, re.IGNORECASE):
        return items
    text = user_ingredient_text(text)
    if RESET.match(text):
        items = []
        text = RESET.sub("", text, count=1).strip(" ,:;.!\n")
    replacement = REPLACE.fullmatch(text.rstrip(".!"))
    if replacement:
        old, new = replacement.groups()
        return list(dict.fromkeys(_remove(items, _names(old)) + _names(new)))

    for clause in re.split(r"\b(?:tapi|tetapi|namun|sedangkan)\b", text, flags=re.IGNORECASE):
        removing = False
        uncertain = False
        for segment in USER_SEPARATORS.split(clause):
            segment = segment.strip()
            if not segment:
                continue
            if UNCERTAIN.search(segment) or re.search(r"\b(?:tidak|kurang)\s+yakin\b", segment, re.IGNORECASE):
                uncertain = True
                continue
            if REMOVE.match(segment) or UNAVAILABLE.search(segment):
                removing, uncertain = True, False
            elif ADD.match(segment) or EXPLICIT_AVAILABILITY.search(segment):
                removing, uncertain = False, False
            if uncertain:
                continue
            segment = ADD.sub("", REMOVE.sub("", segment))
            if removing:
                segment = UNAVAILABLE.sub("", segment)
                items = _remove(items, _names(segment))
            else:
                items = list(dict.fromkeys(items + _names(segment)))
    return items

