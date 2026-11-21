from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


MAIN_GROUPS = [
    "Serealia",
    "Umbi Berpati",
    "Kacang-kacangan",
    "Sayuran",
    "Daging",
    "Ikan/Kerang/Udang/dll",
    "Telur",
]

# Kuota validation:
# total 30 = 15 query 2 bahan + 15 query 3 bahan.
VALIDATION_QUOTAS = {
    "Serealia": {2: 3, 3: 2},  # 5
    "Umbi Berpati": {2: 2, 3: 3},  # 5
    "Kacang-kacangan": {2: 2, 3: 2},  # 4
    "Sayuran": {2: 2, 3: 2},  # 4
    "Daging": {2: 2, 3: 2},  # 4
    "Ikan/Kerang/Udang/dll": {2: 2, 3: 2},  # 4
    "Telur": {2: 2, 3: 2},  # 4
}

# Kuota test:
# 7 kelompok x 10 source = 70.
# Tiap kelompok: 5 query 2 bahan + 5 query 3 bahan.
TEST_QUOTAS = {group: {2: 5, 3: 5} for group in MAIN_GROUPS}


GROUP_KEYWORDS = {
    "Serealia": [
        "beras",
        "nasi",
        "beras merah",
        "beras ketan",
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
    ],
    "Umbi Berpati": [
        "kentang",
        "singkong",
        "ubi",
        "ubi jalar",
        "ubi ungu",
        "talas",
        "sagu",
        "tepung tapioka",
        "tapioka",
    ],
    "Kacang-kacangan": [
        "tahu",
        "tempe",
        "kacang tanah",
        "kacang merah",
        "kacang hijau",
        "kacang kedelai",
        "kedelai",
        "kacang polong",
        "edamame",
    ],
    "Sayuran": [
        "bayam",
        "kangkung",
        "sawi",
        "sawi hijau",
        "sawi putih",
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
        "daun singkong",
        "daun pepaya",
        "daun katuk",
        "genjer",
        "pare",
        "oyong",
        "gambas",
        "selada",
        "timun",
        "mentimun",
        "jagung muda",
        "rebung",
        "nangka muda",
        "jantung pisang",
        "paprika",
        "jamur",
        "jamur tiram",
        "jamur kancing",
    ],
    "Daging": [
        "ayam",
        "daging ayam",
        "dada ayam",
        "paha ayam",
        "daging sapi",
        "sapi",
        "daging kambing",
        "kambing",
        "bebek",
        "daging bebek",
        "hati ayam",
        "ati ayam",
        "ampela",
        "jeroan",
    ],
    "Ikan/Kerang/Udang/dll": [
        "ikan",
        "ikan tongkol",
        "tongkol",
        "ikan tuna",
        "tuna",
        "ikan kembung",
        "kembung",
        "ikan lele",
        "lele",
        "ikan nila",
        "nila",
        "ikan bandeng",
        "bandeng",
        "ikan patin",
        "patin",
        "ikan gurame",
        "gurame",
        "ikan tenggiri",
        "tenggiri",
        "ikan teri",
        "teri",
        "ikan cakalang",
        "cakalang",
        "ikan dori",
        "dori",
        "ikan salmon",
        "salmon",
        "ikan kakap",
        "kakap",
        "ikan mujair",
        "mujair",
        "ikan gabus",
        "gabus",
        "ikan mas",
        "ikan sarden",
        "sarden",
        "ikan asin",
        "ikan gambolo",
        "gambolo",
        "udang",
        "ebi",
        "cumi",
        "cumi-cumi",
        "sotong",
        "gurita",
        "kerang",
        "kepiting",
        "rajungan",
    ],
    "Telur": [
        "telur",
        "telur ayam",
        "telur bebek",
        "telur puyuh",
    ],
}


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_boundary_match(text: str, keyword: str) -> bool:
    """
    Mencocokkan keyword sebagai frasa/token utuh untuk mengurangi
    false-positive substring.
    """
    text = normalize_text(text)
    keyword = normalize_text(keyword)

    if not text or not keyword:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_main_groups(ingredients_raw: object) -> list[str]:
    text = normalize_text(ingredients_raw)
    detected: list[str] = []

    for group in MAIN_GROUPS:
        keywords = GROUP_KEYWORDS[group]
        if any(token_boundary_match(text, kw) for kw in keywords):
            detected.append(group)

    return detected


def stable_hash(seed: int, value: str) -> str:
    payload = f"{seed}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_primary_group(
    detected_groups: list[str],
    remaining_quota_by_group: dict[str, int],
    hash_key: str,
) -> str | None:
    """
    Memilih satu kelompok utama untuk stratifikasi jika sebuah resep
    terdeteksi memiliki lebih dari satu kelompok bahan utama.

    Prioritas:
    1. Kelompok yang masih memiliki kebutuhan kuota terbesar.
    2. Tie-break deterministik berdasarkan hash.
    """
    eligible = [g for g in detected_groups if remaining_quota_by_group.get(g, 0) > 0]

    if not eligible:
        return None

    max_remaining = max(remaining_quota_by_group[g] for g in eligible)
    tied = [g for g in eligible if remaining_quota_by_group[g] == max_remaining]

    tied.sort(
        key=lambda g: hashlib.sha256(f"{hash_key}|{g}".encode("utf-8")).hexdigest()
    )

    return tied[0]


def build_quota_rows(
    quotas: dict[str, dict[int, int]],
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []

    for group in MAIN_GROUPS:
        for target_n in (2, 3):
            n = int(quotas[group][target_n])
            rows.extend([(group, target_n)] * n)

    return rows


def select_split(
    pool: pd.DataFrame,
    quotas: dict[str, dict[int, int]],
    seed: int,
    split_name: str,
    already_used_urls: set[str],
) -> pd.DataFrame:
    required_total_by_group = {
        group: int(sum(quotas[group].values())) for group in MAIN_GROUPS
    }

    remaining_total_by_group = required_total_by_group.copy()
    selected_records: list[dict] = []

    candidates = pool.loc[~pool["source_url"].isin(already_used_urls)].copy()

    candidates["_sort_key"] = candidates["source_url"].map(
        lambda x: stable_hash(seed, f"{split_name}|{x}")
    )
    candidates = candidates.sort_values(["_sort_key", "source_url"]).reset_index(
        drop=True
    )

    # Tahap 1: tetapkan primary group secara deterministik sambil
    # memenuhi total kuota kelompok.
    assigned: list[dict] = []

    for row in candidates.to_dict("records"):
        detected = row["detected_main_groups_list"]

        group = choose_primary_group(
            detected,
            remaining_total_by_group,
            row["_sort_key"],
        )

        if group is None:
            continue

        item = dict(row)
        item["primary_main_group"] = group
        assigned.append(item)
        remaining_total_by_group[group] -= 1

        if all(v == 0 for v in remaining_total_by_group.values()):
            break

    missing_groups = {g: n for g, n in remaining_total_by_group.items() if n > 0}

    if missing_groups:
        raise SystemExit(
            f"Unable to satisfy {split_name} main-group quotas. "
            f"Still missing: {missing_groups}"
        )

    assigned_df = pd.DataFrame(assigned)

    # Tahap 2: di dalam tiap kelompok, bagi secara deterministik
    # menjadi target query 2 dan 3 bahan.
    for group in MAIN_GROUPS:
        subset = assigned_df[assigned_df["primary_main_group"] == group].copy()

        subset["_target_sort"] = subset["source_url"].map(
            lambda x: stable_hash(
                seed,
                f"{split_name}|{group}|target_n|{x}",
            )
        )
        subset = subset.sort_values(["_target_sort", "source_url"]).reset_index(
            drop=True
        )

        n_two = int(quotas[group][2])
        n_three = int(quotas[group][3])

        if len(subset) != n_two + n_three:
            raise SystemExit(
                f"{split_name}/{group}: expected "
                f"{n_two + n_three} rows, found {len(subset)}."
            )

        subset.loc[: n_two - 1, "target_n_ingredients"] = 2
        subset.loc[n_two:, "target_n_ingredients"] = 3

        selected_records.extend(subset.to_dict("records"))

    selected = pd.DataFrame(selected_records)

    selected["_final_sort"] = selected["source_url"].map(
        lambda x: stable_hash(
            seed,
            f"{split_name}|final_order|{x}",
        )
    )
    selected = selected.sort_values(["_final_sort", "source_url"]).reset_index(
        drop=True
    )

    prefix = "V" if split_name == "validation" else "T"

    selected["query_id"] = [f"{prefix}{i:03d}" for i in range(1, len(selected) + 1)]
    selected["query_text"] = ""
    selected["split"] = split_name

    already_used_urls.update(selected["source_url"].astype(str).tolist())

    return selected


def validate_input_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{name}: missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible validation/test source manifests "
            "from Yummy.co.id evaluation-source data."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="CSV containing Yummy.co.id source recipes.",
    )
    parser.add_argument(
        "--pilot",
        required=True,
        help="Pilot query manifest; its source URLs will be excluded.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261022,
        help="Deterministic sampling seed.",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    pilot_path = Path(args.pilot)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_path)
    pilot = pd.read_csv(pilot_path)

    validate_input_columns(
        source,
        {"source_id", "title", "url", "ingredients_raw"},
        "source",
    )
    validate_input_columns(
        pilot,
        {"query_id", "source_url"},
        "pilot",
    )

    source = source.copy()
    source["source_url"] = source["url"].astype(str).str.strip()
    source["source_title"] = source["title"].astype(str).str.strip()

    # Hapus URL kosong / duplikat agar satu source recipe tidak
    # dapat masuk ke lebih dari satu split.
    source = (
        source[source["source_url"].str.len() > 0]
        .drop_duplicates(
            subset=["source_url"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    pilot_urls = set(pilot["source_url"].dropna().astype(str).str.strip().tolist())

    # Jangan gunakan lagi source pilot.
    pool = source[~source["source_url"].isin(pilot_urls)].copy()

    pool["detected_main_groups_list"] = pool["ingredients_raw"].map(detect_main_groups)

    pool = pool[pool["detected_main_groups_list"].map(len) > 0].copy()

    pool["detected_main_groups"] = pool["detected_main_groups_list"].map(
        lambda x: " | ".join(x)
    )

    # Urutan input awal dibuat deterministik juga.
    pool["_base_hash"] = pool["source_url"].map(
        lambda x: stable_hash(args.seed, f"base|{x}")
    )
    pool = pool.sort_values(["_base_hash", "source_url"]).reset_index(drop=True)

    used_urls = set(pilot_urls)

    validation = select_split(
        pool=pool,
        quotas=VALIDATION_QUOTAS,
        seed=args.seed,
        split_name="validation",
        already_used_urls=used_urls,
    )

    test = select_split(
        pool=pool,
        quotas=TEST_QUOTAS,
        seed=args.seed,
        split_name="test",
        already_used_urls=used_urls,
    )

    # Pastikan benar-benar tidak overlap.
    val_urls = set(validation["source_url"].astype(str))
    test_urls = set(test["source_url"].astype(str))

    if val_urls & test_urls:
        raise SystemExit("Internal error: validation and test source URLs overlap.")

    if val_urls & pilot_urls:
        raise SystemExit("Internal error: validation overlaps pilot.")

    if test_urls & pilot_urls:
        raise SystemExit("Internal error: test overlaps pilot.")

    output_cols = [
        "query_id",
        "query_text",
        "source_title",
        "source_url",
        "split",
        "target_n_ingredients",
        "primary_main_group",
        "detected_main_groups",
        "ingredients_raw",
        "source_id",
    ]

    # Tambahkan kolom optional jika tersedia pada source CSV.
    optional_cols = [
        "slug",
        "category",
        "steps",
        "author_name",
        "author_is_official",
    ]

    output_cols += [col for col in optional_cols if col in validation.columns]

    validation_out = validation[output_cols].copy()
    test_out = test[output_cols].copy()

    validation_path = out_dir / "query_manifest_validation_DRAFT.csv"
    test_path = out_dir / "query_manifest_test_DRAFT.csv"

    validation_out.to_csv(validation_path, index=False)
    test_out.to_csv(test_path, index=False)

    summary_rows: list[dict] = []

    for split_name, df in [
        ("validation", validation_out),
        ("test", test_out),
    ]:
        for group in MAIN_GROUPS:
            for target_n in (2, 3):
                count = int(
                    (
                        (df["primary_main_group"] == group)
                        & (df["target_n_ingredients"] == target_n)
                    ).sum()
                )

                summary_rows.append(
                    {
                        "split": split_name,
                        "primary_main_group": group,
                        "target_n_ingredients": target_n,
                        "n_sources": count,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "source_sampling_summary.csv"
    summary.to_csv(summary_path, index=False)

    record = {
        "seed": args.seed,
        "source_csv": str(source_path),
        "source_csv_sha256": sha256_file(source_path),
        "pilot_manifest": str(pilot_path),
        "pilot_manifest_sha256": sha256_file(pilot_path),
        "n_source_rows_original": int(len(source)),
        "n_pilot_urls_excluded": int(len(pilot_urls)),
        "n_eligible_after_pilot_and_main_group_filter": int(len(pool)),
        "validation_n": int(len(validation_out)),
        "test_n": int(len(test_out)),
        "validation_query_size_counts": {
            str(k): int(v)
            for k, v in validation_out["target_n_ingredients"]
            .value_counts()
            .sort_index()
            .items()
        },
        "test_query_size_counts": {
            str(k): int(v)
            for k, v in test_out["target_n_ingredients"]
            .value_counts()
            .sort_index()
            .items()
        },
        "validation_main_group_counts": {
            str(k): int(v)
            for k, v in validation_out["primary_main_group"]
            .value_counts()
            .sort_index()
            .items()
        },
        "test_main_group_counts": {
            str(k): int(v)
            for k, v in test_out["primary_main_group"]
            .value_counts()
            .sort_index()
            .items()
        },
        "rules": {
            "pilot_sources_excluded": True,
            "validation_test_source_overlap_allowed": False,
            "query_text_generated_by_script": False,
            "query_text_must_be_filled_before_retrieval": True,
            "selection_uses_system_recommendation_results": False,
        },
    }

    record_path = out_dir / "source_sampling_record.json"
    record_path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Evaluation source sampling complete.")
    print(f"Seed: {args.seed}")
    print(f"Eligible sources: {len(pool)}")
    print(f"Validation sources: {len(validation_out)}")
    print(f"Test sources: {len(test_out)}")
    print()
    print("Validation target size counts:")
    print(
        validation_out["target_n_ingredients"].value_counts().sort_index().to_string()
    )
    print()
    print("Test target size counts:")
    print(test_out["target_n_ingredients"].value_counts().sort_index().to_string())
    print()
    print(f"Saved: {validation_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {record_path}")
    print()
    print(
        "IMPORTANT: Fill query_text in both DRAFT manifests "
        "before running any recommendation/evaluation pipeline."
    )


if __name__ == "__main__":
    main()
