from __future__ import annotations

import html
import re
from functools import lru_cache

INGREDIENT_ALIASES = {
    "telor": "telur",
    "baput": "bawang putih",
    "bamer": "bawang merah",
    "bombai": "bombay",
    "saos": "saus",
}
ALIAS_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, INGREDIENT_ALIASES)) + r")\b")
# These product names describe a different pantry item from their source food.
# Keep the same product category on both sides of a phrase match; e.g. generic
# "kaldu" can match "kaldu ayam", while "ayam" cannot.
PRODUCT_MARKERS = frozenset({"kaldu", "terasi", "petis", "kerupuk"})

UNITS = [
    "sendok makan",
    "sendok teh",
    "sdm",
    "sdt",
    "gram",
    "gr",
    "kg",
    "ons",
    "ml",
    "mili",
    "mil",
    "liter",
    "ltr",
    "l",
    "butir",
    "btr",
    "buah",
    "bh",
    "siung",
    "lembar",
    "lbr",
    "batang",
    "btg",
    "ikat",
    "ruas",
    "cm",
    "potong",
    "ptg",
    "biji",
    "bj",
    "ekor",
    "ekr",
    "genggam",
    "bungkus",
    "bks",
    "sachet",
    "sct",
    "papan",
    "buku",
    "helai",
    "hlai",
    "gelas",
    "gls",
    "cup",
    "mangkuk",
    "piring",
    "sisir",
    "kotak",
    "ktk",
    "kaleng",
    "klg",
    "botol",
    "btl",
    "pcs",
]
UNITS_SORTED = sorted(UNITS, key=len, reverse=True)
UNIT_PATTERN = "|".join(re.escape(u) for u in UNITS_SORTED)
QTY_PATTERN = (
    r"\d+\s*-\s*\d+"
    r"|\d+\s+\d+\s*/\s*\d+"
    r"|\d+\s*/\s*\d+"
    r"|\d+[.,]\d+"
    r"|\d+"
    r"|sesuai selera"
    r"|secukupnya"
    r"|sejumput"
    r"|beberapa"
)
INGREDIENT_REGEX = re.compile(
    rf"^\s*(?P<qty>{QTY_PATTERN})?\s*(?:(?P<unit>{UNIT_PATTERN})\b)?\.?\s*(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
TRAILING_AMOUNT = re.compile(
    rf"\s+(?:{QTY_PATTERN})\s*(?:{UNIT_PATTERN})\b\.?\s*$",
    re.IGNORECASE,
)
PREPARATION_NOTE = re.compile(
    r"\s+(?:(?:di)?(?:iris|potong|cincang|geprek|rajang|kupas|suwir|giling)\b"
    r"|(?:di)?(?:memarkan|haluskan|parut|cuci)\b"
    r"|sesuai\s+selera\b|secukupnya\b|sebagai\s+(?:topping|taburan)\b)",
    re.IGNORECASE,
)

# Apply these rules only to user messages, never to corpus ingredient names.
SUBJECT = r"(?:aku|saya|gue|gua|kami|kita)"
MODIFIERS = r"(?:(?:cuma|hanya|masih|kebetulan|sekarang|lagi|juga)\s+)*"
USER_PREFIXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"^(?:halo|hai|hei|hey)(?:\s+(?:dapuria|kak|min))?[\s,:.!]+",
    r"^di\s+(?:rumah|dapur|kulkas)(?:ku)?\s*[:,]?\s*",
    rf"^bahan(?:-bahan)?\s+yang\s+(?:{SUBJECT}\s+)?(?:punya|miliki|ada)\s*[:,]?\s*",
    r"^bahan(?:-bahan)?\s*:\s*",
    (rf"^(?:(?:tolong|coba|bantu(?:in)?(?:\s+{SUBJECT})?)\s+)?"
    r"(?:cari(?:kan)?|rekomendasi(?:kan)?)\s+(?:resep|menu|masakan)\b"
    r"(?:\s+(?:dari|dengan|pakai|pake|berbahan)(?:\s+(?:bahan|dasar))?)?\s*[:,]?\s*"),
    rf"^(?:{SUBJECT}\s+)?(?:mau|ingin|pengen)\s+masak\s+(?:pakai|pake|dengan|dari)\s+",
    rf"^(?:{SUBJECT}\s+)?{MODIFIERS}(?:punya|memiliki|ada(?:nya)?|tersedia)\b\s*{MODIFIERS}[:,]?\s*",
    r"^yang\s+(?:ada(?:nya)?|tersedia)\b\s*[:,]?\s*",
    r"^(?:cuma|hanya)\s+",
))
USER_SUFFIX = re.compile(
    r"(?:\s+(?:nih|dong|ya|yah|saja|aja|doang|kok|secukupnya)"
    r"|\s+(?:(?:juga|yang)\s+)?(?:ada|tersedia)"
    r"|\s+di\s+(?:rumah|dapur|kulkas)(?:ku)?)$", re.IGNORECASE,
)
REQUEST_SUFFIX = re.compile(
    r"(?:[.!?,]\s*|\s+)(?:(?:bisa|boleh|enaknya|cocoknya)\s+(?:di)?(?:masak|bikin|olah|buat)\b"
    r"|masak\s+apa\b|(?:tolong\s+)?cari(?:kan)?\s+resep\b).*$",
    re.IGNORECASE | re.DOTALL,
)
USER_SEPARATORS = re.compile(
    # A comma between digits belongs to a decimal quantity, e.g. 0,5 kg.
    r"\s*(?:(?<!\d)[,.]|[,.](?!\d)|;|[\r\n!?]+|[&+]"
    r"|\b(?:dan|sama|serta|plus)\b|\bjuga\b(?!\s+(?:punya|memiliki|ada|tersedia)\b)"
    r"|\s+(?=tanpa\b))\s*",
    re.IGNORECASE,
)
UNAVAILABLE = re.compile(
    r"\b(?:tidak|nggak|enggak|gak|ga|ngga|belum|tanpa|bukan|jangan|habis|abis)\b",
    re.IGNORECASE,
)
UNCERTAIN = re.compile(
    r"\b(?:mungkin|kayaknya|sepertinya|entah|atau)\b", re.IGNORECASE,
)
EXPLICIT_AVAILABILITY = re.compile(
    r"\b(?:punya|memiliki|miliki|ada(?:nya)?|tersedia)\b|^(?:cuma|hanya)\b",
    re.IGNORECASE,
)


def _strip_user_framing(text: str) -> str:
    """Remove anchored conversational phrases while preserving compound names."""
    text = text.strip(" \t\r\n.,!?;")
    while True:
        previous = text
        for prefix in USER_PREFIXES:
            text = prefix.sub("", text).strip()
        text = USER_SUFFIX.sub("", text).strip()
        if text == previous:
            return text.strip(" :")


def normalize_ingredient_name(name: str) -> str:
    """Normalize whitespace/case and explicitly registered whole-word aliases."""
    name = str(name).lower().strip()
    name = re.sub(r"\s+", " ", name)
    return ALIAS_PATTERN.sub(lambda match: INGREDIENT_ALIASES[match.group()], name)


@lru_cache(maxsize=16384)
def clean_name(name: str) -> str:
    """Keep the ingredient identity, excluding amounts and preparation notes."""
    name = html.unescape(str(name))
    name = re.sub(r"\([^)]*(?:\)|$)", "", name)
    name = PREPARATION_NOTE.split(name, maxsplit=1)[0]
    name = TRAILING_AMOUNT.sub("", name.strip())
    name = normalize_ingredient_name(name).strip(" .:;/-()")
    if not re.search(r"[^\W\d_]", name) or name in UNITS:
        return ""
    return name


def parse_ingredient(raw: str) -> dict[str, str]:
    raw = html.unescape(str(raw)).strip()
    raw = re.sub(r"^[\-\*•]+\s*", "", raw)
    main = re.split(r"(?<!\d),|,(?!\d)", raw, maxsplit=1)[0].strip()
    main = re.sub(r"(?<=\d),(?=\d)", ".", main)
    match = INGREDIENT_REGEX.match(main)
    name = (
        clean_name(match.group("name"))
        if (match and match.group("name") and (
            match.group("qty") or (match.group("unit") and match.group("unit").lower() != "buah")
        ))
        else clean_name(main)
    )
    return {"raw": raw, "name": name}


def user_ingredient_text(text: str) -> str:
    """Strip framing before trailing recipe requests so prefixes are preserved."""
    return REQUEST_SUFFIX.sub("", _strip_user_framing(html.unescape(str(text))))


def parse_user_ingredients(text: str) -> list[str]:
    """Extract a deterministic inventory from lists or common Indonesian phrases.

    Negated or uncertain lists stay excluded until a contrast or an explicit
    ownership statement starts a new list. This is not a general language model.
    """
    text = user_ingredient_text(text)
    parsed: list[str] = []
    for clause in re.split(r"\b(?:tapi|tetapi|sedangkan|namun)\b", text, flags=re.IGNORECASE):
        available = True
        for item in USER_SEPARATORS.split(clause):
            item = item.strip()
            if not item:
                continue
            if UNAVAILABLE.search(item):
                available = False
                continue
            if UNCERTAIN.search(item):
                available = False
                continue
            framed = _strip_user_framing(item)
            if EXPLICIT_AVAILABILITY.search(item) or EXPLICIT_AVAILABILITY.search(framed):
                available = True
            if not available:
                continue
            framed = re.sub(r"(?<=\d),(?=\d)", ".", framed)
            name = parse_ingredient(framed)["name"] if framed else ""
            if name:
                parsed.append(name)
    return list(dict.fromkeys(parsed))


def ingredient_matches(user_name: str, recipe_name: str) -> bool:
    """Match aliases and whole phrases within the same product category.

    It avoids substring errors such as ``teri`` matching ``terigu`` while still
    allowing ``ayam`` to match ``dada ayam``. Product markers keep source foods
    distinct from stock, shrimp paste and crackers in both directions.
    """
    user_name = clean_name(user_name)
    recipe_name = clean_name(recipe_name)
    if not user_name or not recipe_name:
        return False
    if user_name == recipe_name:
        return True
    user_products = PRODUCT_MARKERS.intersection(re.findall(r"\w+", user_name))
    recipe_products = PRODUCT_MARKERS.intersection(re.findall(r"\w+", recipe_name))
    if user_products != recipe_products:
        return False
    # Stock is commonly named without its form. Other powders are not assumed
    # available merely because the fresh ingredient is present (or vice versa).
    if "kaldu" not in user_products and (
        bool(re.search(r"\bbubuk\b", user_name)) != bool(re.search(r"\bbubuk\b", recipe_name))
    ):
        return False
    pattern_u = rf"(?<!\w){re.escape(user_name)}(?!\w)"
    pattern_r = rf"(?<!\w){re.escape(recipe_name)}(?!\w)"
    return bool(re.search(pattern_u, recipe_name) or re.search(pattern_r, user_name))
