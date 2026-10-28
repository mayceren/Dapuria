from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapuria.config import Settings


def parse_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return list(parsed) if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the ChromaDB index from the cleaned 5,000-recipe corpus."
    )
    parser.add_argument(
        "--corpus", default=str(ROOT / "data" / "corpus" / "recipes.csv")
    )
    parser.add_argument("--chroma-dir", default=str(ROOT / "data" / "chroma_db"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing Chroma directory before rebuilding.",
    )
    args = parser.parse_args()

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install -r requirements.txt"
        ) from exc

    settings = Settings()
    corpus = pd.read_csv(args.corpus)
    required = {
        "title_clean",
        "ingredient_names",
        "steps_text",
        "text_for_embedding",
        "num_ingredients_clean",
    }
    missing = required - set(corpus.columns)
    if missing:
        raise SystemExit(f"Corpus is missing columns: {sorted(missing)}")

    chroma_dir = Path(args.chroma_dir)
    if chroma_dir.exists() and args.force:
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(settings.model_name)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"recipe_{i:05d}" for i in range(len(corpus))]
    documents = corpus["text_for_embedding"].fillna("").astype(str).tolist()
    passage_texts = [f"passage: {text}" for text in documents]

    for start in range(0, len(corpus), args.batch_size):
        end = min(start + args.batch_size, len(corpus))
        batch = corpus.iloc[start:end]
        embeddings = model.encode(
            passage_texts[start:end],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        metadatas = []
        for _, row in batch.iterrows():
            ingredient_names = parse_list(row["ingredient_names"])
            metadatas.append(
                {
                    "title": str(row["title_clean"]),
                    "ingredient_names": json.dumps(
                        ingredient_names, ensure_ascii=False
                    ),
                    "num_ingredients": int(row["num_ingredients_clean"]),
                    "steps_text": str(row.get("steps_text", "")),
                }
            )
        collection.upsert(
            ids=ids[start:end],
            embeddings=np.asarray(embeddings).tolist(),
            documents=documents[start:end],
            metadatas=metadatas,
        )
        print(f"Indexed {end}/{len(corpus)}")

    print(f"Finished. Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
