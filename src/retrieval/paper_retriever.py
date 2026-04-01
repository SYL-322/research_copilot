"""Lexical retrieval over stored paper chunks (no vector DB)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.models import PaperChunk

# Very common English stopwords — reduces noise for short queries (MVP heuristic).
_STOPWORDS: frozenset[str] = frozenset(
    "a an the and or but if in on at to for of as is was are were be been being "
    "it this that these those with from by not no yes do does did can could should "
    "would will may might must we you they he she how what when where why which "
    "than then into about over such".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens; drop very short tokens and stopwords."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if len(t) > 2 and t not in _STOPWORDS}


def lexical_score(query: str, document: str) -> float:
    """
    Lightweight similarity: Jaccard index on token sets plus a small overlap density term.

    ``document`` may concatenate chunk body with section title for better matching.
    """
    q = _tokenize(query)
    if not q:
        return 0.0
    d = _tokenize(document)
    if not d:
        return 0.0
    inter = len(q & d)
    union = len(q | d)
    jaccard = inter / union if union else 0.0
    # Slight boost when many query terms hit a long chunk
    density = inter / max(1, len(d) ** 0.5)
    return jaccard + 0.15 * density


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk with a retrieval score and stable identifier for citations."""

    chunk_index: int
    section_title: str | None
    content: str
    score: float


def retrieve_relevant_chunks(
    chunks: list[PaperChunk],
    question: str,
    *,
    top_k: int = 8,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Rank chunks by lexical overlap with ``question``; return up to ``top_k`` with
    ``score > min_score`` (default ``min_score`` is 0, so zero-overlap chunks are dropped).
    """
    if not chunks or not question.strip():
        return []

    scored: list[tuple[float, PaperChunk]] = []
    for ch in chunks:
        sec = (ch.section_title or "").strip()
        augmented = f"{sec}\n{ch.content}" if sec else ch.content
        s = lexical_score(question, augmented)
        scored.append((s, ch))

    scored.sort(key=lambda x: (-x[0], x[1].chunk_index))
    out: list[RetrievedChunk] = []
    for s, ch in scored:
        if s <= min_score:
            continue
        out.append(
            RetrievedChunk(
                chunk_index=ch.chunk_index,
                section_title=ch.section_title,
                content=ch.content,
                score=s,
            )
        )
        if len(out) >= top_k:
            break

    return out


def format_evidence_block(retrieved: list[RetrievedChunk]) -> str:
    """Human-readable block for the prompt with chunk identifiers."""
    if not retrieved:
        return "(No evidence chunks were retrieved. Answer from the paper memory summary only, or state that evidence is missing.)"
    lines: list[str] = []
    for r in retrieved:
        sec = f' section="{r.section_title}"' if r.section_title else ""
        lines.append(f"--- chunk_index={r.chunk_index}{sec} score={r.score:.4f} ---")
        lines.append(r.content.strip())
        lines.append("")
    return "\n".join(lines).strip()
