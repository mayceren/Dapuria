from __future__ import annotations

import argparse

import re

from pathlib import Path

import pandas as pd

from dapuria.preprocessing import parse_ingredient, parse_user_ingredients

MAIN_INGREDIENT_KEYWORDS = [
    # Serealia
    "beras",
    "nasi",
    "ketan",
    "jagung",
    "jagung manis",
    "oat",
    "oatmeal",
    "gandum",
    "tepung beras",
    "tepung gandum",
    "mie",
    "mi",
    "bihun",
    "soun",
    "kwetiau",
    "makaroni",
    "pasta",
    # Umbi berpati
    "kentang",
    "singkong",
    "ubi",
    "ubi jalar",
    "ubi ungu",
    "talas",
    "sagu",
    "tapioka",
    "tepung tapioka",
    # Kacang-kacangan
    "tahu",
    "tempe",
    "kacang tanah",
    "kacang merah",
    "kacang hijau",
    "kedelai",
    "edamame",
    "kacang polong",
    # Sayuran
    "bayam",
    "kangkung",
    "sawi",
    "pakcoy",
    "pokcoy",
    "kol",
    "kubis",
    "wortel",
    "buncis",
    "kacang panjang",
    "labu siam",
    "labu kuning",
    "terong",
    "tomat",
    "tomat hijau",
    "brokoli",
    "kembang kol",
    "pare",
    "oyong",
    "gambas",
    "selada",
    "timun",
    "mentimun",
    "rebung",
    "nangka muda",
    "jantung pisang",
    "paprika",
    "jamur",
    # Daging
    "ayam",
    "daging ayam",
    "dada ayam",
    "paha ayam",
    "sapi",
    "daging sapi",
    "kambing",
    "daging kambing",
    "bebek",
    "hati ayam",
    "ati ayam",
    "ampela",
    # Ikan/hasil laut
    "ikan",
    "tongkol",
    "ikan tongkol",
    "tuna",
    "ikan tuna",
    "kembung",
    "lele",
    "nila",
    "bandeng",
    "patin",
    "gurame",
    "tenggiri",
    "teri",
    "cakalang",
    "dori",
    "salmon",
    "kakap",
    "mujair",
    "gabus",
    "sarden",
    "udang",
    "ebi",
    "cumi",
    "cumi-cumi",
    "sotong",
    "gurita",
    "kerang",
    "kepiting",
    "rajungan",
    "ikan gambolo",
    "gambolo",
    # Telur
    "telur",
    "telur ayam",
    "telur bebek",
    "telur puyuh",
]

TOOL_WORDS = {
    "blender",
    "piring",
    "piring saji",
    "wajan",
    "teflon",
    "oven",
    "kompor",
    "cobek",
    "loyang",
    "kukusan",
    "magic com",
    "rice cooker",
    "rolling pin",
    "kuas",
    "spatula",
    "sendok",
    "garpu",
}


def norm(text: object) -> str:

    value = "" if pd.isna(text) else str(text)

    value = value.lower().strip()

    value = re.sub(r"\s+", " ", value)

    return value


def boundary_contains(text: str, phrase: str) -> bool:

    text = norm(text)

    phrase = norm(phrase)

    if not text or not phrase:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def has_main_ingredient(items: list[str]) -> bool:

    for item in items:
        if any(
            boundary_contains(item, kw) or boundary_contains(kw, item)
            for kw in MAIN_INGREDIENT_KEYWORDS
        ):
            return True

    return False


def split_source_ingredients(raw: object) -> list[str]:

    text = "" if pd.isna(raw) else str(raw)

    parts = re.split(r"\s*\|\s*|\r?\n+", text)

    return [p.strip() for p in parts if p.strip()]


# Normalisasi khusus awalan jumlah desimal dengan koma pada sebuah fragmen
# ingredients_raw. Ini tidak mengubah CSV atau bahan yang tidak terkena pola.
DECIMAL_COMMA_AMOUNT = re.compile(
    r"^\s*\d+\s*,\s*\d+\s*"
    r"(?:kilogram|kg|gram|gr|g|ons|miligram|mg|mililiter|ml|liter|l|"
    r"sdm|sdt|sendok makan|sendok teh|buah|bh|butir|ekor|pcs|potong)\b\s*",
    flags=re.IGNORECASE,
)


def normalized_source_names(raw: object) -> list[str]:
    names: list[str] = []
    for part in split_source_ingredients(raw):
        # Misal "1,5 kg kepiting laut" -> "kepiting laut".
        # Hanya untuk awalan berformat desimal koma + satuan.
        cleaned = DECIMAL_COMMA_AMOUNT.sub("", part)
        if cleaned != part and cleaned.strip():
            name = norm(cleaned)
        else:
            try:
                parsed = parse_ingredient(part)
                name = norm(parsed.get("name", ""))
            except Exception:
                name = norm(part)
        if name:
            names.append(name)
    return names


def match_query_item(query_item: str, source_names: list[str]) -> tuple[str, str]:

    q = norm(query_item)

    exact = [s for s in source_names if s == q]

    if exact:
        return "exact", exact[0]

    # Containment dipakai sebagai indikator review yang masih dapat diterima,

    # misalnya "ayam" terhadap "daging ayam bagian dada".

    contained = [
        s for s in source_names if boundary_contains(s, q) or boundary_contains(q, s)
    ]

    if contained:
        contained.sort(key=len)

        return "containment", contained[0]

    return "not_found", ""


def validate_manifest(
    df: pd.DataFrame,
    expected_split: str,
    expected_n_rows: int,
) -> tuple[pd.DataFrame, list[str]]:

    required = {
        "query_id",
        "query_text",
        "source_title",
        "source_url",
        "split",
        "target_n_ingredients",
        "primary_main_group",
        "ingredients_raw",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise SystemExit(f"{expected_split}: missing required columns: {missing}")

    global_errors: list[str] = []

    if len(df) != expected_n_rows:
        global_errors.append(
            f"{expected_split}: expected {expected_n_rows} rows, found {len(df)}"
        )

    if df["query_id"].duplicated().any():
        global_errors.append(f"{expected_split}: duplicate query_id found")

    if df["source_url"].duplicated().any():
        global_errors.append(f"{expected_split}: duplicate source_url found")

    wrong_split = df["split"].fillna("").astype(str) != expected_split

    if wrong_split.any():
        global_errors.append(
            f"{expected_split}: {int(wrong_split.sum())} rows have wrong split value"
        )

    report_rows: list[dict] = []

    for row in df.to_dict("records"):
        issues: list[str] = []

        warnings: list[str] = []

        qid = str(row["query_id"])

        query_text = (
            "" if pd.isna(row["query_text"]) else str(row["query_text"]).strip()
        )

        try:
            target_n = int(float(row["target_n_ingredients"]))

        except Exception:
            target_n = -1

            issues.append("invalid target_n_ingredients")

        parsed: list[str] = []

        if not query_text:
            issues.append("query_text kosong")

        else:
            try:
                parsed = parse_user_ingredients(query_text)

            except Exception as exc:
                issues.append(f"parser error: {exc}")

        if parsed and len(parsed) != target_n:
            issues.append(f"jumlah bahan hasil parser={len(parsed)}, target={target_n}")

        parsed_norm = [norm(x) for x in parsed]

        if len(parsed_norm) != len(set(parsed_norm)):
            issues.append("ada bahan query duplikat setelah normalisasi")

        if parsed and not has_main_ingredient(parsed):
            issues.append("query tidak memuat bahan utama yang dikenali validator")

        tool_hits = [
            item
            for item in parsed
            if any(boundary_contains(item, tool) for tool in TOOL_WORDS)
        ]

        if tool_hits:
            issues.append("query mengandung alat/non-bahan: " + ", ".join(tool_hits))

        source_names = normalized_source_names(row["ingredients_raw"])

        match_notes: list[str] = []

        for item in parsed:
            kind, matched = match_query_item(item, source_names)

            if kind == "not_found":
                issues.append(
                    f"bahan query tidak ditemukan pada ingredients_raw: {item}"
                )

                match_notes.append(f"{item}=NOT_FOUND")

            elif kind == "containment":
                warnings.append(f"cek kecocokan '{item}' dengan source '{matched}'")

                match_notes.append(f"{item}=containment:{matched}")

            else:
                match_notes.append(f"{item}=exact:{matched}")

        if issues:
            status = "ERROR"

        elif warnings:
            status = "REVIEW"

        else:
            status = "PASS"

        report_rows.append(
            {
                "query_id": qid,
                "split": expected_split,
                "source_title": row["source_title"],
                "source_url": row["source_url"],
                "target_n_ingredients": target_n,
                "query_text": query_text,
                "parsed_query": " | ".join(parsed),
                "primary_main_group": row["primary_main_group"],
                "status": status,
                "issues": " ; ".join(issues),
                "warnings": " ; ".join(warnings),
                "source_match_details": " ; ".join(match_notes),
            }
        )

    return pd.DataFrame(report_rows), global_errors


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate filled validation/test query manifests before freezing "
            "or running any recommendation pipeline."
        )
    )

    parser.add_argument("--validation", required=True)

    parser.add_argument("--test", required=True)

    parser.add_argument("--pilot", required=True)

    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    validation_path = Path(args.validation)

    test_path = Path(args.test)

    pilot_path = Path(args.pilot)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(validation_path)

    test = pd.read_csv(test_path)

    pilot = pd.read_csv(pilot_path)

    val_report, val_global = validate_manifest(
        validation,
        expected_split="validation",
        expected_n_rows=30,
    )

    test_report, test_global = validate_manifest(
        test,
        expected_split="test",
        expected_n_rows=70,
    )

    global_errors = val_global + test_global

    pilot_urls = set(pilot["source_url"].dropna().astype(str).str.strip().tolist())

    val_urls = set(validation["source_url"].dropna().astype(str).str.strip().tolist())

    test_urls = set(test["source_url"].dropna().astype(str).str.strip().tolist())

    if val_urls & test_urls:
        global_errors.append(
            f"validation/test URL overlap: {len(val_urls & test_urls)}"
        )

    if val_urls & pilot_urls:
        global_errors.append(
            f"validation/pilot URL overlap: {len(val_urls & pilot_urls)}"
        )

    if test_urls & pilot_urls:
        global_errors.append(f"test/pilot URL overlap: {len(test_urls & pilot_urls)}")

    all_report = pd.concat(
        [val_report, test_report],
        ignore_index=True,
    )

    report_path = out_dir / "query_manifest_validation_report.csv"

    all_report.to_csv(report_path, index=False)

    summary = (
        all_report.groupby(["split", "status"]).size().rename("n_rows").reset_index()
    )

    summary_path = out_dir / "query_manifest_validation_summary.csv"

    summary.to_csv(summary_path, index=False)

    n_error = int((all_report["status"] == "ERROR").sum())

    n_review = int((all_report["status"] == "REVIEW").sum())

    n_pass = int((all_report["status"] == "PASS").sum())

    print("Manifest validation finished.")

    print(f"PASS   : {n_pass}")

    print(f"REVIEW : {n_review}")

    print(f"ERROR  : {n_error}")

    print(f"Report : {report_path}")

    print(f"Summary: {summary_path}")

    if global_errors:
        print("\nGLOBAL ERRORS:")

        for item in global_errors:
            print(f"- {item}")

    if n_error or global_errors:
        raise SystemExit(
            "\nManifest belum siap di-freeze. "
            "Perbaiki semua ERROR/global error terlebih dahulu."
        )

    if n_review:
        print(
            "\nTidak ada ERROR, tetapi masih ada REVIEW. "
            "Periksa semua baris REVIEW secara manual sebelum freeze."
        )

    else:
        print("\nSemua baris PASS. Manifest siap untuk pemeriksaan akhir lalu freeze.")


if __name__ == "__main__":
    main()
