from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class AnswerEmbeddingRetriever:
    """Global answer-only E5 retrieval over a precomputed normalized matrix."""

    def __init__(
        self,
        embedding_path,
        expected_rows,
        model_name="intfloat/multilingual-e5-large",
        device=None,
    ):
        self.embedding_path = Path(embedding_path)
        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Answer embedding file not found: {self.embedding_path}"
            )

        self.matrix = np.load(self.embedding_path, mmap_mode="r")
        if self.matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D answer embedding matrix, got {self.matrix.shape}"
            )
        if self.matrix.shape[0] != int(expected_rows):
            raise ValueError(
                "Answer embedding row count does not match taxonomy rows: "
                f"{self.matrix.shape[0]} != {expected_rows}. "
                "Do not continue because row alignment would be unsafe."
            )

        kwargs = {}
        if device:
            kwargs["device"] = device

        self.model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name

    def search(self, query, top_k=5):
        query = str(query or "").strip()
        if not query:
            return []

        query_vector = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        if query_vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "Query/document embedding dimension mismatch: "
                f"{query_vector.shape[0]} != {self.matrix.shape[1]}"
            )

        scores = np.asarray(self.matrix @ query_vector, dtype=np.float32)
        k = min(max(1, int(top_k)), len(scores))

        if k == len(scores):
            order = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -k)[-k:]
            order = idx[np.argsort(scores[idx])[::-1]]

        return [
            {
                "idx": int(row_idx),
                "dense": float(scores[int(row_idx)]),
                "channel": "answer_embedding",
                "channel_rank": rank,
                "route_rank": None,
                "route_score": 0.0,
                "path": "GLOBAL_ANSWER_EMBEDDING",
            }
            for rank, row_idx in enumerate(order, start=1)
        ]
