import json
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL, INDEX_DIR


class SearchEngine:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.chunks: List[Dict] = []

        self.index_path = INDEX_DIR / "books.index"
        self.meta_path = INDEX_DIR / "chunks.json"

        self._load_existing()

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def _load_existing(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        self._save()

    def is_empty(self) -> bool:
        return self.index is None or len(self.chunks) == 0

    def rebuild(self, chunks: List[Dict]) -> None:
        self.chunks = chunks

        if not chunks:
            self.index = None
            self._save()
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self._save()

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        if not new_chunks:
            return
        self.rebuild(self.chunks + new_chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.is_empty():
            return []

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        similarities, indices = self.index.search(query_vector, top_k)

        results: List[Dict] = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            item = dict(self.chunks[idx])
            item["score"] = float(1.0 - similarity)
            item["similarity"] = float(similarity)
            results.append(item)

        return results

    def list_books(self) -> List[str]:
        return sorted({chunk["source"] for chunk in self.chunks})