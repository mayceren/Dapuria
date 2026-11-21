from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ASPECTS = [
    "recommendation_fit_1_5",
    "additional_ingredients_reasonable_1_5",
    "ease_of_use_1_5",
    "information_clarity_1_5",
    "result_feasibility_1_5",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 1-5 human evaluation ratings."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = set(ASPECTS) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing aspect columns: {sorted(missing)}")
    for col in ASPECTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or not df[col].between(1, 5).all():
            raise SystemExit(f"{col} must contain only values 1-5.")

    rows = [
        {
            "aspect": col,
            "mean": df[col].mean(),
            "std": df[col].std(ddof=1),
            "n": df[col].count(),
        }
        for col in ASPECTS
    ]
    overall = df[ASPECTS].mean(axis=1)
    rows.append(
        {
            "aspect": "overall_mean",
            "mean": overall.mean(),
            "std": overall.std(ddof=1),
            "n": overall.count(),
        }
    )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
