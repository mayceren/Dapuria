from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapuria import RecipeService, Settings


REQUIRED_COLUMNS = {"query_id", "query_text"}


def code_for(pass_label: str, query_id: str, recipe_id: str, seed: int) -> str:
    raw = f"{pass_label}|{query_id}|{recipe_id}|{seed}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10].upper()


def make_form(
    rows: list[dict], pass_label: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    form_rows: list[dict] = []
    map_rows: list[dict] = []

    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    for query_id, items in by_query.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        for item in shuffled:
            blind_code = code_for(pass_label, query_id, str(item["recipe_id"]), seed)
            form_rows.append(
                {
                    "query_id": query_id,
                    "query_text": item["query_text"],
                    "blind_code": blind_code,
                    "recipe_title": item["recipe_title"],
                    "recipe_ingredients": item["recipe_ingredients"],
                    "recipe_steps": item["recipe_steps"],
                    "grade": "",
                    "notes": "",
                }
            )
            map_rows.append(
                {
                    "pass": pass_label,
                    "query_id": query_id,
                    "query_text": item["query_text"],
                    "blind_code": blind_code,
                    "recipe_id": item["recipe_id"],
                    "recipe_title": item["recipe_title"],
                    "original_rank": item["original_rank"],
                    "final_score": item["final_score"],
                    "retrieval_source": item["retrieval_source"],
                }
            )

    return pd.DataFrame(form_rows), pd.DataFrame(map_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate two blind Top-5 grading forms for thesis evaluation."
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="CSV with query_id, query_text, and optional split columns.",
    )
    parser.add_argument(
        "--out-dir", default="evaluation/runs/latest", help="Output directory."
    )
    parser.add_argument(
        "--seed", type=int, default=20261016, help="Randomization seed."
    )
    args = parser.parse_args()

    query_path = Path(args.queries)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(query_path)
    missing = REQUIRED_COLUMNS - set(queries.columns)
    if missing:
        raise SystemExit(f"Missing query columns: {sorted(missing)}")
    if queries["query_id"].duplicated().any():
        raise SystemExit("query_id must be unique.")

    service = RecipeService(Settings())
    candidate_rows: list[dict] = []
    status_rows: list[dict] = []

    for row in queries.to_dict("records"):
        query_id = str(row["query_id"])
        query_text = str(row["query_text"])
        result = service.handle_user_input(query_text, top_n=5)
        status_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "split": row.get("split", ""),
                "status": result.get("status", "unknown"),
                "n_displayed": len(result.get("results", [])),
            }
        )
        if result.get("status") != "ok":
            continue
        for rank, recipe in enumerate(result["results"], start=1):
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "recipe_id": recipe["id"],
                    "recipe_title": recipe["title"],
                    "recipe_ingredients": ", ".join(recipe["ingredient_names"]),
                    "recipe_steps": recipe.get("steps_text", ""),
                    "original_rank": rank,
                    "final_score": recipe["final_score"],
                    "retrieval_source": recipe["retrieval_source"],
                }
            )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / "run_status.csv", index=False)

    pass1, map1 = make_form(candidate_rows, "pass1", args.seed)
    pass2, map2 = make_form(candidate_rows, "pass2", args.seed + 104729)
    pass1.to_csv(out_dir / "blind_judgment_pass1.csv", index=False)
    pass2.to_csv(out_dir / "blind_judgment_pass2.csv", index=False)
    pd.concat([map1, map2], ignore_index=True).to_csv(
        out_dir / "blind_mapping_PRIVATE.csv", index=False
    )

    print(f"Saved blind evaluation files to: {out_dir}")
    print(
        "Grade pass1 first. Grade pass2 later without opening blind_mapping_PRIVATE.csv."
    )
    print("Allowed grades: 0 = unsuitable, 1 = acceptable, 2 = strongly relevant.")


if __name__ == "__main__":
    main()
