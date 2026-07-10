"""BM25 + FAISS index construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def build_index(
    chunks: list[dict[str, Any]],
    output_dir: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    np.save(output_dir / "embeddings.npy", embeddings)
