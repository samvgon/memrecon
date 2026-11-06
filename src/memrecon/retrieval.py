"""Retrieval: which memories are relevant to the current turn (D14, D35).

Memories are embedded as documents and each turn as a query (see embed.py);
relevance is the cosine similarity between the two. A memory is recalled for a
turn only if it is among the `k` most similar live memories **and** its
similarity reaches `threshold`; both are set on development personas only
(D14). Results come back most relevant first, so the order the model sees
memories in carries no hint of their age or lineage (D28).
"""

from __future__ import annotations

import numpy as np


class Retriever:
    def __init__(self, embedder, k: int = 3, threshold: float = 0.0) -> None:
        self.embedder = embedder
        self.k = k
        self.threshold = threshold
        self._text: dict[str, str] = {}
        self._vec: dict[str, np.ndarray] = {}
        self._cache: tuple[list[str], np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._text)

    def add(self, items: list[tuple[str, str]]) -> None:
        """Index (memory_id, text) pairs; re-adding an ID with new text replaces it."""
        todo = [(mid, t) for mid, t in items if self._text.get(mid) != t]
        if not todo:
            return
        vectors = self.embedder.embed([t for _, t in todo], "document")
        for (mid, t), v in zip(todo, vectors):
            self._text[mid] = t
            self._vec[mid] = np.asarray(v, dtype=np.float32)
        self._cache = None

    def remove(self, ids: list[str]) -> None:
        """Drop memories from the index (retired from the store)."""
        for mid in ids:
            self._text.pop(mid, None)
            self._vec.pop(mid, None)
        self._cache = None

    def sync(self, store) -> None:
        """Make the index match the store's live memories: new or changed texts in, retired ones out."""
        live = {m.id: m.text for m in store.active()}
        self.remove([mid for mid in self._text if mid not in live])
        self.add(list(live.items()))

    def scores(self, query: str) -> list[tuple[str, float]]:
        """Similarity of the query to every indexed memory, most similar first (ties: insertion order)."""
        if not self._text:
            return []
        if self._cache is None:
            ids = list(self._vec)
            self._cache = (ids, np.vstack([self._vec[i] for i in ids]))
        ids, matrix = self._cache
        q = np.asarray(self.embedder.embed([query], "query")[0], dtype=np.float32)
        sims = matrix @ q
        order = np.argsort(-sims, kind="stable")
        return [(ids[i], float(sims[i])) for i in order]

    def search(self, query: str, k: int | None = None, threshold: float | None = None) -> list[tuple[str, float]]:
        """The memories recalled for this turn: the top k that reach the threshold, most relevant first."""
        k = self.k if k is None else k
        threshold = self.threshold if threshold is None else threshold
        return [(mid, s) for mid, s in self.scores(query)[:k] if s >= threshold]
