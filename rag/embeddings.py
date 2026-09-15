"""Dense embeddings via a local Ollama model (nomic-embed-text).

Deviation from the architecture doc's plan (documented in CLAUDE.md Current
Status under M7): the doc named `sentence-transformers` for local dense
embeddings. Ollama was already required for M8's LLM agent and already had
`nomic-embed-text` pulled locally, so calling Ollama's embedding endpoint
over the `httpx` client (already a Part-1 dependency) avoids adding
`sentence-transformers` + its transformers/torch-adjacent footprint purely
for embeddings. This is an implementation choice below the Locked Decisions
level - RAG is still hybrid BM25 + dense embedding similarity, still fully
local and free, per docs/TECHNICAL_ARCHITECTURE.md 2.6, which also notes
faiss is "optional overkill" at this corpus size - so a plain in-memory
cosine search is used instead of a vector store.
"""

from __future__ import annotations

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached."""


def embed_text(text: str, model: str = EMBED_MODEL, timeout: float = 30.0) -> np.ndarray:
    import httpx

    try:
        resp = httpx.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaUnavailableError(
            f"could not reach Ollama at {OLLAMA_URL} (is `ollama serve` running?): {e}"
        ) from e
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    return vec


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """No batch endpoint in the older Ollama embeddings API - one call per text."""
    return np.stack([embed_text(t, model=model) for t in texts])


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    return d @ q
