"""Simple text retrieval (keyword / chunk overlap; vectors optional later)."""

from retrieval.memory_retriever import get_relevant_paper_memories
from retrieval.paper_retriever import (
    RetrievedChunk,
    format_evidence_block,
    lexical_score,
    retrieve_relevant_chunks,
)
from retrieval.simple import chunk_text, retrieve_top_chunks

__all__ = [
    "RetrievedChunk",
    "chunk_text",
    "format_evidence_block",
    "get_relevant_paper_memories",
    "lexical_score",
    "retrieve_relevant_chunks",
    "retrieve_top_chunks",
]
