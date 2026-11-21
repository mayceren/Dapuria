from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def validate_grades(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "grade" not in df.columns:
        raise SystemExit(f"{name}: missing grade column")

    out = df.copy()
    out["grade"] = pd.to_numeric(out["grade"], errors="coerce")

    if out["grade"].isna().any():
        bad = out.loc[
            out["grade"].isna(),
            ["query_id", "blind_code"],
        ]
        raise SystemExit(
            f"{name}: some grades are blank/invalid. Example:\n{bad.head()}"
        )

    if not out["grade"].isin([0, 1, 2]).all():
        raise SystemExit(f"{name}: grades must be 0, 1, or 2")

    out["grade"] = out["grade"].astype(int)
    return out


def discounts_at_5() -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, 7))


def dcg_at_5(grades: list[int]) -> float:
    padded = list(grades)[:5] + [0] * max(0, 5 - len(grades))
    arr = np.asarray(padded[:5], dtype=float)
    return float(np.sum((2 ** arr - 1) * discounts_at_5()))


def pooled_ndcg_at_5(
    ranked_grades: list[int],
    all_pool_grades: list[int],
) -> float:
    dcg = dcg_at_5(ranked_grades)

    ideal = sorted(
        [int(x) for x in all_pool_grades],
        reverse=True,
    )[:5]
    idcg = dcg_at_5(ideal)

    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile pooled blind WLC validation grades, compute metrics "
            "for every WLC configuration, and select the validation winner."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--reconciliation",
        default=None,
        help="CSV containing resolved_grade for disagreements.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    p1 = validate_grades(
        pd.read_csv(run_dir / "blind_judgment_pass1.csv"),
        "pass1",
    )
    p2 = validate_grades(
        pd.read_csv(run_dir / "blind_judgment_pass2.csv"),
        "pass2",
    )
    mapping = pd.read_csv(
        run_dir / "blind_mapping_PRIVATE.csv"
    )
    pooled = pd.read_csv(
        run_dir / "pooled_candidates_PRIVATE.csv"
    )
    runs = pd.read_csv(
        run_dir / "wlc_runs_PRIVATE.csv"
    )
    status = pd.read_csv(
        run_dir / "validation_run_status.csv"
    )

    m1 = mapping[
        mapping["pass"] == "pass1"
    ].merge(
        p1[["blind_code", "grade"]],
        on="blind_code",
        how="left",
    ).rename(columns={"grade": "grade_pass1"})

    m2 = mapping[
        mapping["pass"] == "pass2"
    ].merge(
        p2[["blind_code", "grade"]],
        on="blind_code",
        how="left",
    ).rename(columns={"grade": "grade_pass2"})

    joined = m1[
        [
            "query_id",
            "query_text",
            "recipe_id",
            "recipe_title",
            "grade_pass1",
        ]
    ].merge(
        m2[
            [
                "query_id",
                "recipe_id",
                "grade_pass2",
            ]
        ],
        on=["query_id", "recipe_id"],
        how="inner",
    )

    # Tambahkan informasi resep hanya untuk reconciliation blind.
    joined = joined.merge(
        pooled[
            [
                "query_id",
                "recipe_id",
                "recipe_ingredients",
                "recipe_steps",
            ]
        ],
        on=["query_id", "recipe_id"],
        how="left",
    )

    joined["agree"] = (
        joined["grade_pass1"] == joined["grade_pass2"]
    )

    agreement_rate = (
        float(joined["agree"].mean())
        if len(joined)
        else 1.0
    )

    agreement_summary = pd.DataFrame([{
        "n_items": len(joined),
        "n_agreements": int(joined["agree"].sum()),
        "n_disagreements": int((~joined["agree"]).sum()),
        "two_pass_raw_agreement": agreement_rate,
    }])
    agreement_summary.to_csv(
        run_dir / "grading_agreement.csv",
        index=False,
    )

    print(f"Two-pass raw agreement: {agreement_rate:.4f}")

    disagreements = joined[
        ~joined["agree"]
    ].copy()

    reconciliation_path = (
        run_dir / "grading_disagreements_to_resolve.csv"
    )

    if not disagreements.empty and not args.reconciliation:
        reconciliation_view = disagreements[
            [
                "query_id",
                "query_text",
                "recipe_id",
                "recipe_title",
                "recipe_ingredients",
                "recipe_steps",
                "grade_pass1",
                "grade_pass2",
            ]
        ].copy()
        reconciliation_view["resolved_grade"] = ""
        reconciliation_view.to_csv(
            reconciliation_path,
            index=False,
        )

        print(f"Found {len(disagreements)} disagreements.")
        print(f"Resolve them in: {reconciliation_path}")
        print(
            "Then rerun with --reconciliation <completed CSV>."
        )
        return

    joined["final_grade"] = joined["grade_pass1"]

    if not disagreements.empty:
        resolved = pd.read_csv(args.reconciliation)

        required = {
            "query_id",
            "recipe_id",
            "resolved_grade",
        }
        missing = required - set(resolved.columns)
        if missing:
            raise SystemExit(
                f"Reconciliation missing columns: {sorted(missing)}"
            )

        key_cols = ["query_id", "recipe_id"]

        if resolved.duplicated(key_cols).any():
            raise SystemExit(
                "Reconciliation contains duplicate query_id + recipe_id rows."
            )

        resolved["resolved_grade"] = pd.to_numeric(
            resolved["resolved_grade"],
            errors="coerce",
        )

        if (
            resolved["resolved_grade"].isna().any()
            or not resolved["resolved_grade"].isin([0, 1, 2]).all()
        ):
            raise SystemExit(
                "All resolved_grade values must be 0, 1, or 2."
            )

        expected_keys = set(
            map(
                tuple,
                disagreements[key_cols]
                .astype(str)
                .to_numpy(),
            )
        )
        received_keys = set(
            map(
                tuple,
                resolved[key_cols]
                .astype(str)
                .to_numpy(),
            )
        )

        if expected_keys != received_keys:
            raise SystemExit(
                "Reconciliation keys do not exactly match disagreements."
            )

        resolved_map = resolved[
            key_cols + ["resolved_grade"]
        ]

        joined = joined.merge(
            resolved_map,
            on=key_cols,
            how="left",
        )

        mask = ~joined["agree"]

        joined.loc[mask, "final_grade"] = (
            joined.loc[mask, "resolved_grade"]
            .astype(int)
        )

        joined = joined.drop(
            columns=["resolved_grade"]
        )

    final_labels = joined[
        [
            "query_id",
            "query_text",
            "recipe_id",
            "recipe_title",
            "final_grade",
        ]
    ].copy()

    final_labels.to_csv(
        run_dir / "pooled_blind_labels_FINAL.csv",
        index=False,
    )

    grade_map = {
        (str(row.query_id), str(row.recipe_id)): int(row.final_grade)
        for row in final_labels.itertuples(index=False)
    }

    pool_grades_by_query: dict[str, list[int]] = {}
    for qid, group in final_labels.groupby("query_id"):
        pool_grades_by_query[str(qid)] = (
            group["final_grade"]
            .astype(int)
            .tolist()
        )

    query_ids = status["query_id"].astype(str).tolist()

    config_pairs = (
        runs[
            [
                "semantic_weight",
                "ingredient_weight",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["semantic_weight", "ingredient_weight"]
        )
        .reset_index(drop=True)
    )

    per_query_rows: list[dict] = []

    for cfg in config_pairs.to_dict("records"):
        sw = float(cfg["semantic_weight"])
        iw = float(cfg["ingredient_weight"])

        subset = runs[
            np.isclose(runs["semantic_weight"], sw)
            & np.isclose(runs["ingredient_weight"], iw)
        ]

        by_query = {
            str(qid): group.sort_values("original_rank")
            for qid, group in subset.groupby("query_id")
        }

        for qid in query_ids:
            if qid not in by_query:
                grades: list[int] = []
            else:
                group = by_query[qid]
                grades = [
                    grade_map[(qid, str(recipe_id))]
                    for recipe_id in group["recipe_id"].tolist()[:5]
                ]

            pool_grades = pool_grades_by_query.get(qid, [])

            per_query_rows.append({
                "query_id": qid,
                "semantic_weight": sw,
                "ingredient_weight": iw,
                "acceptable_success_at_5": int(
                    any(g >= 1 for g in grades)
                ),
                "strong_success_at_5": int(
                    any(g >= 2 for g in grades)
                ),
                "acceptable_precision_at_5": (
                    sum(g >= 1 for g in grades) / 5.0
                ),
                "strong_precision_at_5": (
                    sum(g >= 2 for g in grades) / 5.0
                ),
                "pooled_ndcg_at_5": pooled_ndcg_at_5(
                    grades,
                    pool_grades,
                ),
                "mean_grade_at_5": sum(grades) / 5.0,
                "n_displayed": len(grades),
            })

    per_query = pd.DataFrame(per_query_rows)
    per_query.to_csv(
        run_dir / "wlc_validation_metrics_per_query.csv",
        index=False,
    )

    metric_cols = [
        "acceptable_success_at_5",
        "strong_success_at_5",
        "acceptable_precision_at_5",
        "strong_precision_at_5",
        "pooled_ndcg_at_5",
        "mean_grade_at_5",
    ]

    summary = (
        per_query.groupby(
            ["semantic_weight", "ingredient_weight"],
            as_index=False,
        )[metric_cols]
        .mean()
    )

    counts = (
        per_query.groupby(
            ["semantic_weight", "ingredient_weight"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_queries"})
    )

    summary = counts.merge(
        summary,
        on=["semantic_weight", "ingredient_weight"],
        how="left",
    )

    # PRECOMMITTED selection rule:
    # 1) pooled nDCG@5
    # 2) Strong Precision@5
    # 3) Strong Success@5
    # 4) Acceptable Precision@5
    # 5) Acceptable Success@5
    # 6) Mean Grade@5
    # 7) Higher semantic weight as deterministic final tie-break.
    summary = summary.sort_values(
        [
            "pooled_ndcg_at_5",
            "strong_precision_at_5",
            "strong_success_at_5",
            "acceptable_precision_at_5",
            "acceptable_success_at_5",
            "mean_grade_at_5",
            "semantic_weight",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.insert(
        0,
        "validation_rank",
        np.arange(1, len(summary) + 1),
    )

    summary.to_csv(
        run_dir / "wlc_validation_summary.csv",
        index=False,
    )

    winner = summary.iloc[[0]].copy()
    winner.to_csv(
        run_dir / "wlc_selected_config.csv",
        index=False,
    )

    print()
    print("Validation ranking:")
    print(summary.to_string(index=False))
    print()
    print("SELECTED WLC CONFIG:")
    print(winner.to_string(index=False))
    print()
    print(
        "Do not run the frozen TEST manifest until this selected "
        "configuration has been reviewed and frozen."
    )


if __name__ == "__main__":
    main()
