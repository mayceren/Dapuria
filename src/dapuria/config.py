from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Central configuration for the thesis system.

    Retrieval defaults follow the recovered production configuration; WLC weights
    follow wlc_selected_config_FROZEN.csv. Environment variables can override them.
    """

    model_name: str = os.getenv("DAPURIA_MODEL_NAME", "LazarusNLP/all-indo-e5-small-v4")
    chroma_dir: Path = Path(
        os.getenv("DAPURIA_CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
    )
    collection_name: str = os.getenv("DAPURIA_COLLECTION_NAME", "resep_masakan")
    corpus_path: Path = Path(
        os.getenv("DAPURIA_CORPUS_PATH", str(PROJECT_ROOT / "data" / "corpus" / "recipes.csv"))
    )

    top_k_sbert: int = int(os.getenv("DAPURIA_TOP_K_SBERT", "50"))
    max_keyword_extra: int = int(os.getenv("DAPURIA_MAX_KEYWORD_EXTRA", "20"))
    semantic_weight: float = float(os.getenv("DAPURIA_SEMANTIC_WEIGHT", "0.9"))
    ingredient_weight: float = float(os.getenv("DAPURIA_INGREDIENT_WEIGHT", "0.1"))
    top_n: int = int(os.getenv("DAPURIA_TOP_N", "5"))

    min_ingredient_count: int = int(os.getenv("DAPURIA_MIN_INGREDIENT_COUNT", "1"))

    ollama_url: str = os.getenv("DAPURIA_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("DAPURIA_OLLAMA_MODEL", "llama3")
    ollama_timeout_seconds: int = int(os.getenv("DAPURIA_OLLAMA_TIMEOUT", "60"))

    def validate(self) -> None:
        if self.top_k_sbert < 1:
            raise ValueError("top_k_sbert must be >= 1")
        if self.max_keyword_extra < 0:
            raise ValueError("max_keyword_extra must be >= 0")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if abs((self.semantic_weight + self.ingredient_weight) - 1.0) > 1e-9:
            raise ValueError("semantic_weight + ingredient_weight must equal 1.0")
