"""Chunking and lightweight scoring without a vector DB."""

from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping character windows.

    TODO: sentence-aware splits; token-based limits for LLM context.
    """
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def retrieve_top_chunks(query: str, chunks: list[str], *, k: int = 5) -> list[tuple[int, str]]:
    """
    Score chunks by simple token overlap with query; return top-k (index, text).

    TODO: stem/lemmatize; optional BM25; optional embeddings.
    """
    q_tokens = set(query.lower().split())
    scored: list[tuple[float, int, str]] = []
    for i, ch in enumerate(chunks):
        c_tokens = set(ch.lower().split())
        overlap = len(q_tokens & c_tokens)
        scored.append((float(overlap), i, ch))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(i, t) for _, i, t in scored[:k]]
