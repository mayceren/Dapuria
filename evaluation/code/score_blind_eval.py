from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd


def validate_grades(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "grade" not in df.columns:
        raise SystemExit(f"{name}: missing grade column")
    out = df.copy()
    out["grade"] = pd.to_numeric(out["grade"], errors="coerce")
    if out["grade"].isna().any():
        bad = out.loc[out["grade"].isna(), ["query_id", "blind_code"]]
        raise SystemExit(
            f"{name}: some grades are blank/invalid. Example:\n{bad.head()}"
        )
    if not out["grade"].isin([0, 1, 2]).all():
        raise SystemExit(f"{name}: grades must be 0, 1, or 2")
    out["grade"] = out["grade"].astype(int)
    return out


def ndcg_at_5(grades: list[int]) -> float:
    grades = list(grades)[:5]
    if not grades:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(grades)))
    arr = np.asarray(grades, dtype=float)
    ideal = np.sort(arr)[::-1]
    dcg = float(np.sum((2**arr - 1) * discounts))
    idcg = float(np.sum((2**ideal - 1) * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile two blind grading passes and compute Top-5 metrics."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Directory created by prepare_blind_eval.py"
    )
    parser.add_argument(
        "--reconciliation",
        default=None,
        help="Optional CSV containing resolved_grade for disagreements.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    p1 = validate_grades(pd.read_csv(run_dir / "blind_judgment_pass1.csv"), "pass1")
    p2 = validate_grades(pd.read_csv(run_dir / "blind_judgment_pass2.csv"), "pass2")
    mapping = pd.read_csv(run_dir / "blind_mapping_PRIVATE.csv")
    status = pd.read_csv(run_dir / "run_status.csv")

    m1 = (
        mapping[mapping["pass"] == "pass1"]
        .merge(p1[["blind_code", "grade"]], on="blind_code", how="left")
        .rename(columns={"grade": "grade_pass1"})
    )

    m2 = (
        mapping[mapping["pass"] == "pass2"]
        .merge(p2[["blind_code", "grade"]], on="blind_code", how="left")
        .rename(columns={"grade": "grade_pass2"})
    )

    joined = m1[
        [
            "query_id",
            "query_text",
            "recipe_id",
            "recipe_title",
            "original_rank",
            "grade_pass1",
        ]
    ].merge(
        m2[["query_id", "recipe_id", "grade_pass2"]],
        on=["query_id", "recipe_id"],
        how="inner",
    )

    joined["agree"] = joined["grade_pass1"] == joined["grade_pass2"]

    # ==========================================================
    # TAMBAHAN BARU: Simpan two-pass grading agreement
    # ==========================================================

    agreement_rate = float(joined["agree"].mean()) if len(joined) else 1.0

    agreement_summary = pd.DataFrame(
        [
            {
                "n_items": len(joined),
                "n_agreements": int(joined["agree"].sum()),
                "n_disagreements": int((~joined["agree"]).sum()),
                "two_pass_raw_agreement": agreement_rate,
            }
        ]
    )

    agreement_summary.to_csv(
        run_dir / "grading_agreement.csv",
        index=False,
    )

    print(f"Two-pass raw agreement: {agreement_rate:.4f}")
    print(f"Saved grading agreement in: {run_dir / 'grading_agreement.csv'}")

    # ==========================================================
    # AKHIR TAMBAHAN BARU
    # ==========================================================

    disagreements = joined[~joined["agree"]].copy()
    reconciliation_path = run_dir / "grading_disagreements_to_resolve.csv"

    if not disagreements.empty and not args.reconciliation:
        reconciliation_view = disagreements.drop(
            columns=["original_rank", "agree"],
            errors="ignore",
        ).copy()

        reconciliation_view["resolved_grade"] = ""
        reconciliation_view.to_csv(
            reconciliation_path,
            index=False,
        )

        print(f"Agreement between passes: {agreement_rate:.4f}")
        print(f"Found {len(disagreements)} disagreements.")
        print(f"Resolve them in: {reconciliation_path}")
        print("Then rerun with --reconciliation <completed CSV>.")
        return

    joined["final_grade"] = joined["grade_pass1"]

    if not disagreements.empty:
        resolved = pd.read_csv(args.reconciliation)

        if "resolved_grade" not in resolved.columns:
            raise SystemExit("Reconciliation file must contain resolved_grade.")

        resolved["resolved_grade"] = pd.to_numeric(
            resolved["resolved_grade"], errors="coerce"
        )

        if (
            resolved["resolved_grade"].isna().any()
            or not resolved["resolved_grade"].isin([0, 1, 2]).all()
        ):
            raise SystemExit("All resolved_grade values must be 0, 1, or 2.")

        key_cols = ["query_id", "recipe_id"]

        if resolved.duplicated(key_cols).any():
            duplicated = resolved.loc[
                resolved.duplicated(key_cols, keep=False),
                key_cols,
            ]

            raise SystemExit(
                "Reconciliation contains duplicate query_id + recipe_id rows:\n"
                f"{duplicated.head()}"
            )

        resolved_map = resolved[key_cols + ["resolved_grade"]]

        expected_keys = set(map(tuple, disagreements[key_cols].astype(str).to_numpy()))

        received_keys = set(map(tuple, resolved_map[key_cols].astype(str).to_numpy()))

        if expected_keys != received_keys:
            missing = expected_keys - received_keys
            extra = received_keys - expected_keys

            raise SystemExit(
                "Reconciliation keys do not exactly match disagreements.\n"
                f"Missing: {list(missing)[:5]}\n"
                f"Extra: {list(extra)[:5]}"
            )

        joined = joined.merge(resolved_map, on=key_cols, how="left")

        mask = ~joined["agree"]

        if joined.loc[mask, "resolved_grade"].isna().any():
            raise SystemExit("Some disagreements do not have resolved_grade values.")

        joined.loc[mask, "final_grade"] = joined.loc[mask, "resolved_grade"].astype(int)

        joined = joined.drop(columns=["resolved_grade"])

    final_labels = joined.sort_values(["query_id", "original_rank"]).copy()

    final_labels.to_csv(run_dir / "blind_labels_FINAL.csv", index=False)

    metric_rows: list[dict] = []

    labels_by_query = {qid: group for qid, group in final_labels.groupby("query_id")}

    for row in status.to_dict("records"):
        qid = str(row["query_id"])

        if qid not in labels_by_query:
            grades = []
        else:
            group = labels_by_query[qid].sort_values("original_rank")
            grades = group["final_grade"].astype(int).tolist()[:5]

        metric_rows.append(
            {
                "query_id": qid,
                "query_text": row["query_text"],
                "split": row.get("split", ""),
                "status": row["status"],
                "acceptable_success_at_5": int(any(g >= 1 for g in grades)),
                "strong_success_at_5": int(any(g >= 2 for g in grades)),
                "acceptable_precision_at_5": (sum(g >= 1 for g in grades) / 5.0),
                "strong_precision_at_5": (sum(g >= 2 for g in grades) / 5.0),
                "ndcg_at_5": ndcg_at_5(grades),
                "mean_grade_at_5": sum(grades) / 5.0,
            }
        )

    per_query = pd.DataFrame(metric_rows)

    per_query.to_csv(run_dir / "blind_metrics_per_query.csv", index=False)

    metric_cols = [
        "acceptable_success_at_5",
        "strong_success_at_5",
        "acceptable_precision_at_5",
        "strong_precision_at_5",
        "ndcg_at_5",
        "mean_grade_at_5",
    ]

    summary = pd.DataFrame(
        [
            {
                "n_queries": len(per_query),
                **{col: float(per_query[col].mean()) for col in metric_cols},
            }
        ]
    )

    summary.to_csv(run_dir / "blind_metrics_summary.csv", index=False)

    if (
        "split" in per_query.columns
        and per_query["split"].fillna("").astype(str).str.len().gt(0).any()
    ):
        by_split = (
            per_query.groupby("split", dropna=False)[metric_cols].mean().reset_index()
        )

        counts = (
            per_query.groupby("split", dropna=False)
            .size()
            .rename("n_queries")
            .reset_index()
        )

        by_split = counts.merge(by_split, on="split")

        by_split.to_csv(run_dir / "blind_metrics_by_split.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved final labels and metrics in: {run_dir}")


if __name__ == "__main__":
    main()
