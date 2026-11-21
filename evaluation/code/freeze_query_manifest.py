from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze a thesis query manifest with SHA-256."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    df = pd.read_csv(src)
    required = {"query_id", "query_text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if df["query_id"].duplicated().any():
        raise SystemExit("query_id must be unique before freezing.")
    if df["query_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("query_text must not be blank.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = sha256_file(dst)
    dst.with_suffix(dst.suffix + ".sha256").write_text(
        f"{digest}  {dst.name}\n", encoding="utf-8"
    )
    print(f"Frozen: {dst}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
