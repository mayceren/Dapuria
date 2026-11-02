"""Explicit ingredient exclusions, kept separate from pantry availability."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pantry import RESET
from .preprocessing import (
    EXPLICIT_AVAILABILITY,
    NON_MSG,
    USER_SEPARATORS,
    normalize_ingredient_name,
    parse_user_ingredients,
    user_ingredient_text,
)

DIRECTIVE = re.compile(
    r"\b(?P<clear>(?:hapus(?: semua)?|reset) batasan(?: bahan)?|semua bahan boleh)\b"
    r"|\b(?P<unavailable>(?:tidak|nggak|enggak|gak|ga|ngga|belum)\s+(?:punya|memiliki|ada))\b"
    r"|\b(?P<avoid>jangan(?:\s+(?:pakai|gunakan|tambahkan))?|hindari|tanpa"
    r"|(?:tidak|nggak|enggak|gak|ga|ngga)\s+(?:mau|boleh)(?:\s+(?:pakai|gunakan))?)\b"
    r"|\b(?P<allow>boleh\s+(?:pakai|gunakan)|izinkan)\b",
    re.IGNORECASE,
)
CONTRAST = re.compile(r"\b(?:tapi|tetapi|namun|sedangkan)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PreferenceUpdate:
    inventory_text: str
    excluded: list[str]
    has_directive: bool


def update_preferences(current: list[str], message: str) -> PreferenceUpdate:
    """Extract exclusion/permission clauses without treating them as inventory.

    A list stays in its current mode until a contrast, a new directive, or an
    explicit ownership statement. Named unavailable ingredients also exclude
    recipes; a generic denial remains an answer to the pending question.
    """
    excluded = list(dict.fromkeys(normalize_ingredient_name(item) for item in current))
    text = user_ingredient_text(message)
    has_directive = False
    if RESET.match(text) and not DIRECTIVE.match(text):
        excluded = []
        has_directive = True
    inventory = []
    mode = None
    for block in re.split(r"(\b(?:tapi|tetapi|namun|sedangkan)\b)", text, flags=re.IGNORECASE):
        if CONTRAST.fullmatch(block):
            mode = None
            inventory.append(block)
            continue
        for segment in USER_SEPARATORS.split(block):
            segment = segment.strip()
            if not segment:
                continue
            matches = list(DIRECTIVE.finditer(segment))
            if matches:
                prefix = segment[:matches[0].start()].strip()
                if prefix and prefix.lower() not in {"aku", "saya", "tolong"}:
                    inventory.append(prefix)
                for i, match in enumerate(matches):
                    mode = match.lastgroup
                    if mode != "unavailable":
                        has_directive = True
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
                    names_text = segment[match.end():end].strip()
                    if mode == "clear":
                        names = parse_user_ingredients(names_text)
                        excluded = [item for item in excluded if item not in names] if names else []
                        mode = None
                        continue
                    names_text = re.sub(r"\b(?!punya\b)(\w+)nya\b", r"\1", names_text, flags=re.IGNORECASE)
                    names_text = re.sub(r"\s+lagi$", "", names_text, flags=re.IGNORECASE)
                    names = parse_user_ingredients(names_text)
                    if mode in {"avoid", "unavailable"}:
                        excluded = list(dict.fromkeys(excluded + names))
                        if mode == "unavailable":
                            has_directive = has_directive or bool(names)
                            inventory.extend(f"tidak punya {name}" for name in names)
                            if not names:
                                inventory.append(segment[match.start():end])
                    else:
                        excluded = [item for item in excluded if item not in names]
                continue
            if EXPLICIT_AVAILABILITY.search(segment):
                mode = None
            if mode in {"avoid", "allow", "unavailable"}:
                segment = re.sub(r"\s+lagi$", "", segment, flags=re.IGNORECASE)
                names = parse_user_ingredients(segment)
                if mode in {"avoid", "unavailable"}:
                    excluded = list(dict.fromkeys(excluded + names))
                    if mode == "unavailable":
                        has_directive = has_directive or bool(names)
                        inventory.extend(f"tidak punya {name}" for name in names)
                else:
                    excluded = [item for item in excluded if item not in names]
            else:
                inventory.append(segment)
    return PreferenceUpdate(", ".join(inventory), excluded, has_directive)


def ingredient_is_excluded(name: str, excluded: list[str]) -> bool:
    """Match whole words/phrases, including derived products naming the food.

    Unlike inventory equivalence, avoiding ayam also excludes kaldu ayam.
    This is a literal filter over recorded ingredient names, not an ontology.
    """
    name = " ".join(re.findall(r"\w+", normalize_ingredient_name(name)))
    return any(
        bool(target) and not (target == "msg" and NON_MSG.search(name))
        and bool(re.search(rf"(?<!\w){re.escape(target)}(?!\w)", name))
        for item in excluded
        for target in [" ".join(re.findall(r"\w+", normalize_ingredient_name(item)))]
    )
