from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import Settings
from .preprocessing import ingredient_matches


def add_query_prefix(text: str) -> str:
    return f"query: {text}"


class HybridRetriever:
    """SBERT semantic retrieval plus boundary-aware lexical candidate expansion."""

    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing retrieval dependencies. Run `pip install -r requirements.txt` first."
            ) from exc

        self.model = SentenceTransformer(settings.model_name)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_collection(settings.collection_name)
        self._all_docs_cache: list[tuple[str, Any, Any, Any]] | None = None

    @staticmethod
    def _parse_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
        meta = meta or {}
        raw_names = meta.get("ingredient_names") or "[]"
        try:
            ingredient_names = json.loads(str(raw_names))
        except json.JSONDecodeError:
            ingredient_names = []
        if not isinstance(ingredient_names, list) or not all(
            isinstance(name, str) for name in ingredient_names
        ):
            ingredient_names = []
        return {
            "title": meta.get("title", ""),
            "ingredient_names": ingredient_names,
            "steps_text": meta.get("steps_text", ""),
            "num_ingredients": meta.get("num_ingredients"),
        }

    def healthcheck(self) -> dict[str, Any]:
        return {
            "model_name": self.settings.model_name,
            "collection_name": self.settings.collection_name,
            "document_count": self.collection.count(),
            "chroma_dir": str(self.settings.chroma_dir),
        }

    def _semantic_candidates(
        self, user_text: str
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        query_embedding = self.model.encode(
            [add_query_prefix(user_text)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.settings.top_k_sbert,
            include=["documents", "metadatas", "distances"],
        )
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        if metadatas is None or distances is None:
            raise RuntimeError("Chroma query did not return metadatas/distances.")

        candidates: list[dict[str, Any]] = []
        for idx, recipe_id in enumerate(results["ids"][0]):
            meta = self._parse_meta(metadatas[0][idx])
            candidates.append(
                {
                    "id": recipe_id,
                    **meta,
                    "sbert_score": float(1 - distances[0][idx]),
                    "retrieval_source": "semantic",
                }
            )
        return query_embedding, candidates

    def _get_all_documents(self) -> list[tuple[str, Any, Any, Any]]:
        if self._all_docs_cache is None:
            raw = self.collection.get(include=["metadatas", "documents", "embeddings"])
            if (
                raw.get("metadatas") is None
                or raw.get("documents") is None
                or raw.get("embeddings") is None
            ):
                raise RuntimeError("Chroma get did not return required fields.")
            self._all_docs_cache = [
                (doc_id, self._parse_meta(meta), document, embedding)
                for doc_id, meta, document, embedding in zip(
                    raw["ids"], raw["metadatas"], raw["documents"], raw["embeddings"]
                )
            ]
        return self._all_docs_cache

    def _keyword_candidates(
        self, user_ingredients: list[str], exclude_ids: set[str]
    ) -> list[tuple]:
        scored: list[tuple] = []
        if self.settings.max_keyword_extra == 0:
            return scored
        for doc_id, parsed_meta, document, embedding in self._get_all_documents():
            if doc_id in exclude_ids:
                continue
            names = parsed_meta["ingredient_names"]
            matched_count = sum(
                1
                for user_item in user_ingredients
                if any(
                    ingredient_matches(user_item, recipe_item) for recipe_item in names
                )
            )
            if matched_count <= 0:
                continue
            recipe_coverage = matched_count / len(names) if names else 0.0
            scored.append(
                (
                    matched_count,
                    recipe_coverage,
                    doc_id,
                    parsed_meta,
                    document,
                    embedding,
                )
            )
        scored.sort(key=lambda row: (-row[0], -row[1], str(row[2])))
        return scored[: self.settings.max_keyword_extra]

    def retrieve(
        self, user_text: str, user_ingredients: list[str]
    ) -> list[dict[str, Any]]:
        query_embedding, candidates = self._semantic_candidates(user_text)
        existing_ids = {row["id"] for row in candidates}
        for (
            matched_count,
            _cov,
            doc_id,
            meta,
            _document,
            embedding,
        ) in self._keyword_candidates(user_ingredients, existing_ids):
            semantic_score = float(np.dot(query_embedding, np.asarray(embedding)))
            candidates.append(
                {
                    "id": doc_id,
                    **meta,
                    "sbert_score": semantic_score,
                    "retrieval_source": "keyword",
                    "keyword_matched_count": int(matched_count),
                }
            )
        return candidates
