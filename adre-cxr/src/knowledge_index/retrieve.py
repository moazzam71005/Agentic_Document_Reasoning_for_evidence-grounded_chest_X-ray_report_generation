"""Hybrid retrieval with layout-aware context expansion."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(
        self,
        index_dir: Path,
        alpha: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.alpha = alpha
        self.model = SentenceTransformer(embedding_model)

        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks: list[dict] = data["chunks"]
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_norm = self._normalize(bm25_scores)

        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        dense_scores, dense_idx = self.faiss_index.search(q_emb, top_k * 2)
        dense_norm = np.zeros(len(self.chunks))
        for score, idx in zip(dense_scores[0], dense_idx[0]):
            if idx >= 0:
                dense_norm[idx] = score
        dense_norm = self._normalize(dense_norm)

        fused = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm
        top_indices = np.argsort(fused)[::-1][:30]

        results = []
        seen_texts = set()
        for idx in top_indices:
            if len(results) >= top_k:
                break
            chunk = self.chunks[idx]
            expanded = self._expand_context(chunk)
            sig = " ".join(expanded.split()[:20])
            if sig in seen_texts:
                continue
            seen_texts.add(sig)

            results.append({
                "id": chunk["id"],
                "score": float(fused[idx]),
                "title_path": chunk.get("title_path", ""),
                "text": expanded,
            })
        return results

    def _expand_context(self, chunk: dict, max_words: int = 500) -> str:
        """±1 sibling expansion under same parent_id."""
        parent = chunk.get("parent_id")
        if parent is None:
            text = chunk.get("title_path", "") + "\n" + chunk["text"]
            words = text.split()
            return " ".join(words[:max_words])

        sib_idx = chunk.get("sibling_idx", 0)
        siblings = [
            c for c in self.chunks
            if c.get("parent_id") == parent
            and abs(c.get("sibling_idx", 0) - sib_idx) <= 1
        ]
        siblings.sort(key=lambda c: c.get("sibling_idx", 0))
        text = chunk.get("title_path", "") + "\n" + "\n".join(s["text"] for s in siblings)
        words = text.split()
        return " ".join(words[:max_words])

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)
