from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapuria.config import Settings
from dapuria.preprocessing import parse_user_ingredients
from dapuria.retrieval import HybridRetriever
from dapuria.reranking import (
    coverage_stats,
    harmonic_efficiency,
    minmax_normalize,
    weighted_linear_score,
)
from dapuria.validation import validate_initial, is_relevant_enough


REQUIRED_COLUMNS = {"query_id", "query_text"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_for(pass_label: str, query_id: str, recipe_id: str, seed: int) -> str:
    raw = f"{pass_label}|{query_id}|{recipe_id}|{seed}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10].upper()


def make_form(
    pooled_rows: list[dict],
    pass_label: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    form_rows: list[dict] = []
    map_rows: list[dict] = []

    by_query: dict[str, list[dict]] = {}
    for row in pooled_rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)

    for query_id, items in by_query.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)

        for item in shuffled:
            blind_code = code_for(
                pass_label,
                query_id,
                str(item["recipe_id"]),
                seed,
            )

            form_rows.append({
                "query_id": query_id,
                "query_text": item["query_text"],
                "blind_code": blind_code,
                "recipe_title": item["recipe_title"],
                "recipe_ingredients": item["recipe_ingredients"],
                "recipe_steps": item["recipe_steps"],
                "grade": "",
                "notes": "",
            })

            # Mapping ini sengaja tidak menyimpan rank/score/configuration
            # agar file grading dan reconciliation dapat tetap blind.
            map_rows.append({
                "pass": pass_label,
                "query_id": query_id,
                "query_text": item["query_text"],
                "blind_code": blind_code,
                "recipe_id": item["recipe_id"],
                "recipe_title": item["recipe_title"],
            })

    return pd.DataFrame(form_rows), pd.DataFrame(map_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run pooled blind WLC validation over the frozen validation queries. "
            "Each WLC configuration contributes its Top-5; candidates are pooled "
            "and deduplicated before two-pass blind grading."
        )
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Frozen validation manifest CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/runs/wlc_validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261022,
        help="Random seed for blind form order.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Semantic-weight grid step. Default 0.1 => 11 configs from 0.0 to 1.0.",
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
    if queries["query_text"].isna().any():
        raise SystemExit("query_text contains blank values.")
    if "split" in queries.columns:
        bad = queries["split"].fillna("").astype(str).str.lower() != "validation"
        if bad.any():
            raise SystemExit(
                "This script is only for the validation split. "
                "Some rows do not have split=validation."
            )

    if args.step <= 0 or args.step > 1:
        raise SystemExit("--step must be in (0, 1].")

    n_steps = round(1.0 / args.step)
    weights = [round(i * args.step, 10) for i in range(n_steps + 1)]
    if abs(weights[-1] - 1.0) > 1e-9:
        raise SystemExit(
            "--step must divide 1.0 exactly for this precommitted grid."
        )

    base_settings = Settings()
    retriever = HybridRetriever(base_settings)

    wlc_run_rows: list[dict] = []
    pooled_lookup: dict[tuple[str, str], dict] = {}
    status_rows: list[dict] = []

    for qrow in queries.to_dict("records"):
        query_id = str(qrow["query_id"])
        query_text = str(qrow["query_text"]).strip()
        parsed = parse_user_ingredients(query_text)

        valid_initial, initial_status = validate_initial(
            parsed,
            base_settings.min_ingredient_count,
        )

        if not valid_initial:
            status_rows.append({
                "query_id": query_id,
                "query_text": query_text,
                "status": initial_status,
                "n_retrieved": 0,
                "n_valid_after_t_relevansi": 0,
                "n_pooled_unique": 0,
            })
            continue

        # Canonical query: separator/punctuation variants must map to the
        # same semantic retrieval input.
        canonical_query = ", ".join(parsed)

        candidates = retriever.retrieve(canonical_query, parsed)

        if not candidates:
            status_rows.append({
                "query_id": query_id,
                "query_text": query_text,
                "status": "no_candidates",
                "n_retrieved": 0,
                "n_valid_after_t_relevansi": 0,
                "n_pooled_unique": 0,
            })
            continue

        normalized = minmax_normalize(
            [row["sbert_score"] for row in candidates]
        )

        enriched: list[dict] = []
        for candidate, semantic_norm in zip(candidates, normalized):
            stats = coverage_stats(
                parsed,
                candidate["ingredient_names"],
            )
            efficiency = harmonic_efficiency(
                stats["coverage_recipe"],
                stats["coverage_user"],
            )

            item = {
                **candidate,
                **stats,
                "ingredient_efficiency": float(efficiency),
                "sbert_score_norm": float(semantic_norm),
            }

            # Final relevance validation is per candidate and independent
            # of WLC weights. Only candidates passing T_relevansi are ranked.
            if is_relevant_enough(item, parsed):
                enriched.append(item)

        if not enriched:
            status_rows.append({
                "query_id": query_id,
                "query_text": query_text,
                "status": "no_match",
                "n_retrieved": len(candidates),
                "n_valid_after_t_relevansi": 0,
                "n_pooled_unique": 0,
            })
            continue

        query_pool_ids: set[str] = set()

        for semantic_weight in weights:
            ingredient_weight = round(1.0 - semantic_weight, 10)

            ranked: list[dict] = []
            for candidate in enriched:
                final_score = weighted_linear_score(
                    candidate["sbert_score_norm"],
                    candidate["ingredient_efficiency"],
                    semantic_weight,
                    ingredient_weight,
                )
                ranked.append({
                    **candidate,
                    "final_score": float(final_score),
                })

            ranked.sort(
                key=lambda row: (
                    -row["final_score"],
                    -row["sbert_score_norm"],
                    str(row["id"]),
                )
            )

            top5 = ranked[:5]

            for rank, recipe in enumerate(top5, start=1):
                wlc_run_rows.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "semantic_weight": semantic_weight,
                    "ingredient_weight": ingredient_weight,
                    "original_rank": rank,
                    "recipe_id": recipe["id"],
                    "recipe_title": recipe["title"],
                    "final_score": recipe["final_score"],
                    "sbert_score_norm": recipe["sbert_score_norm"],
                    "ingredient_efficiency": recipe["ingredient_efficiency"],
                    "retrieval_source": recipe["retrieval_source"],
                })

                key = (query_id, str(recipe["id"]))
                query_pool_ids.add(str(recipe["id"]))

                if key not in pooled_lookup:
                    pooled_lookup[key] = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "recipe_id": recipe["id"],
                        "recipe_title": recipe["title"],
                        "recipe_ingredients": ", ".join(
                            recipe["ingredient_names"]
                        ),
                        "recipe_steps": recipe.get("steps_text", ""),
                    }

        status_rows.append({
            "query_id": query_id,
            "query_text": query_text,
            "status": "ok",
            "n_retrieved": len(candidates),
            "n_valid_after_t_relevansi": len(enriched),
            "n_pooled_unique": len(query_pool_ids),
        })

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(
        out_dir / "validation_run_status.csv",
        index=False,
    )

    wlc_runs = pd.DataFrame(wlc_run_rows)
    wlc_runs.to_csv(
        out_dir / "wlc_runs_PRIVATE.csv",
        index=False,
    )

    pooled_rows = list(pooled_lookup.values())
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(
        out_dir / "pooled_candidates_PRIVATE.csv",
        index=False,
    )

    pass1, map1 = make_form(
        pooled_rows,
        "pass1",
        args.seed,
    )
    pass2, map2 = make_form(
        pooled_rows,
        "pass2",
        args.seed + 104729,
    )

    pass1.to_csv(
        out_dir / "blind_judgment_pass1.csv",
        index=False,
    )
    pass2.to_csv(
        out_dir / "blind_judgment_pass2.csv",
        index=False,
    )
    pd.concat(
        [map1, map2],
        ignore_index=True,
    ).to_csv(
        out_dir / "blind_mapping_PRIVATE.csv",
        index=False,
    )

    record = {
        "validation_manifest": str(query_path),
        "validation_manifest_sha256": sha256_file(query_path),
        "seed": args.seed,
        "grid": [
            {
                "semantic_weight": sw,
                "ingredient_weight": round(1.0 - sw, 10),
            }
            for sw in weights
        ],
        "top_k_sbert": base_settings.top_k_sbert,
        "max_keyword_extra": base_settings.max_keyword_extra,
        "top_n_per_config": 5,
        "model_name": base_settings.model_name,
        "collection_name": base_settings.collection_name,
        "n_queries": int(len(queries)),
        "n_pooled_candidates": int(len(pooled_df)),
        "method": (
            "Retrieve once per query; compute semantic normalization and "
            "ingredient efficiency once; apply T_relevansi per candidate; "
            "rerank valid candidates under every WLC weight; pool and "
            "deduplicate the Top-5 candidates from all configurations."
        ),
    }

    (out_dir / "wlc_validation_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved WLC validation files to: {out_dir}")
    print(f"Queries: {len(queries)}")
    print(f"WLC configurations: {len(weights)}")
    print(f"Pooled unique candidates: {len(pooled_df)}")
    print()
    print("IMPORTANT:")
    print("- Grade blind_judgment_pass1.csv first.")
    print("- Grade blind_judgment_pass2.csv later.")
    print("- Do NOT open wlc_runs_PRIVATE.csv, pooled_candidates_PRIVATE.csv,")
    print("  or blind_mapping_PRIVATE.csv during grading.")
    print("- Allowed grades: 0, 1, 2 using the frozen grading rubric.")


if __name__ == "__main__":
    main()
