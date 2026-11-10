"""Resolve unspecified soy sauce before recipe retrieval."""

from __future__ import annotations

import re

from .pantry import ADD, REMOVE, REPLACE, RESET, update_inventory
from .preprocessing import EXPLICIT_AVAILABILITY, strip_reply_particles

SOY_SAUCE_QUESTION = (
    "Kecap yang kamu punya jenis apa: **kecap manis**, **kecap asin**, "
    "atau **jenis kecap lainnya**? Kalau jenis lain, sebutkan namanya, ya."
)
TYPE_WORD = re.compile(r"\b(?:manis|asin|dua[ -]?duanya|keduanya)\b", re.IGNORECASE)
REPLY_FRAMING = re.compile(
    r"(?:(?:aku|saya|gue|gua|iya|ya|yang|jenis|punya|memiliki|ada|tersedia|cuma|hanya|juga"
    r"|tidak|nggak|enggak|gak|ga|ngga|belum|habis|bukan|tanpa"
    r"|mungkin|sepertinya|kayaknya)\s*)*", re.IGNORECASE,
)


def expand_soy_sauce_reply(message: str) -> str:
    """Expand contextual short answers without changing foods like ikan asin."""
    segments = re.split(
        r"([,;.!?]|\b(?:dan|sama|serta|tapi|tetapi|namun|sedangkan)\b)",
        message, flags=re.IGNORECASE,
    )
    for index in range(0, len(segments), 2):
        segment = re.sub(r"\bkecapnya\b", "kecap", segments[index], flags=re.IGNORECASE)
        framed = strip_reply_particles(segment.strip())
        if TYPE_WORD.search(framed) and REPLY_FRAMING.fullmatch(TYPE_WORD.sub("", framed).strip()):
            segment = TYPE_WORD.sub(
                lambda match: (f"kecap {match.group().lower()}"
                               if match.group().lower() in {"manis", "asin"}
                               else "kecap manis dan kecap asin"),
                segment,
            )
        segment = re.sub(r"\b(?:yang|jenis)\s+(?=kecap\b)", "", segment, flags=re.IGNORECASE)
        segments[index] = re.sub(r"^(\s*)(?:iya|ya)\s+(?=kecap\b)", r"\1", segment, flags=re.IGNORECASE)
    return "".join(segments)


def refine_soy_sauce_inventory(inventory: list[str]) -> list[str]:
    """A named type replaces the redundant, overly broad 'kecap' item."""
    if any(re.fullmatch(r"kecap\s+(?!lain(?:nya)?\b|jenis\b).+", item) for item in inventory):
        return [item for item in inventory if item != "kecap"]
    return inventory


def update_soy_sauce_inventory(inventory: list[str], message: str) -> list[str]:
    """Keep a type question pending until a type or an inventory edit is given."""
    # A question is not a statement of ownership. Bare fillers such as 'lainnya'
    # and 'iya' must not become ingredients while the user is choosing a type.
    if "?" in message or re.search(
        r"\b(?:apa|apakah|gimana|bagaimana|maksudnya|kenapa|lupa)\b", message, re.IGNORECASE,
    ) or not (
        re.search(r"\bkecap\b", message, re.IGNORECASE)
        or EXPLICIT_AVAILABILITY.search(message)
        or any(pattern.match(message) for pattern in (ADD, REMOVE, REPLACE, RESET))
    ):
        return inventory
    updated = [item for item in update_inventory(inventory, message)
               if not re.fullmatch(r"(?:jenis\s+)?kecap\s+(?:lain(?:nya)?|jenis lain)", item)]
    return refine_soy_sauce_inventory(updated)
