"""In-memory dense embedding index over the SOP corpus.

No vector store (faiss etc.) - at 18 documents, brute-force cosine search
over a cached embedding matrix is both simpler and faster than standing up
an index, and docs/TECHNICAL_ARCHITECTURE.md 2.6 explicitly calls faiss
"optional overkill" at this corpus size.

Embeddings are cached to data/cache/ keyed by a hash of the corpus content,
so re-running retrieval/eval/tests doesn't re-hit Ollama for the same 18
docs every time, but a corpus edit invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from rag.corpus import SopDoc, load_corpus
from rag.embeddings import cosine_similarity, embed_text, embed_texts

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_PATH = CACHE_DIR / "rag_dense_embeddings.npz"


def _corpus_hash(docs: tuple[SopDoc, ...]) -> str:
    joined = "\x1f".join(f"{d.id}\x1e{d.full_text}" for d in docs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class DenseIndex:
    def __init__(self, docs: tuple[SopDoc, ...] | None = None, use_cache: bool = True):
        self.docs = docs or load_corpus()
        self.doc_ids = [d.id for d in self.docs]
        self._hash = _corpus_hash(self.docs)
        self.embeddings = self._load_or_build(use_cache)

    def _load_or_build(self, use_cache: bool) -> np.ndarray:
        if use_cache and CACHE_PATH.exists():
            cached = np.load(CACHE_PATH, allow_pickle=True)
            if str(cached["hash"]) == self._hash:
                return cached["embeddings"]
        embeddings = embed_texts([d.full_text for d in self.docs])
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(CACHE_PATH, embeddings=embeddings, hash=self._hash)
        return embeddings

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Returns (doc_id, cosine_similarity) pairs, best first."""
        q_vec = embed_text(query)
        sims = cosine_similarity(q_vec, self.embeddings)
        order = np.argsort(-sims)
        if k is not None:
            order = order[:k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
