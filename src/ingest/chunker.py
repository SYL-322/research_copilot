"""Split paper text into overlapping chunks with optional section labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ingest.pdf_loader import SectionBlock

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """One retrieval chunk with character offsets into the full document."""

    chunk_index: int
    content: str
    section_title: str | None
    char_start: int
    char_end: int


def chunk_sections(
    sections: list[SectionBlock],
    full_text: str,
    *,
    chunk_size: int = 3500,
    overlap: int = 400,
) -> list[TextChunk]:
    """
    Build overlapping chunks, tagging each with an inferred section title when known.

    Offsets refer to ``full_text``. If a section body cannot be located as a substring,
    offsets for that section are set to ``-1``.
    """
    if not full_text.strip():
        return []

    if not sections:
        return _chunk_plain(full_text, chunk_size=chunk_size, overlap=overlap)

    out: list[TextChunk] = []
    global_idx = 0
    search_from = 0

    for sec in sections:
        body = sec.text.strip()
        if not body:
            continue
        pos = full_text.find(body, search_from)
        if pos == -1:
            logger.debug("Section substring not found; offsets unset for this section")
            pos = None

        for start_in_body, end_in_body, window in _slide_windows_with_offsets(
            body, chunk_size=chunk_size, overlap=overlap
        ):
            label = sec.title
            content = _prefix_section(label, window)
            if pos is not None:
                char_start = pos + start_in_body
                char_end = pos + end_in_body
            else:
                char_start = -1
                char_end = -1
            out.append(
                TextChunk(
                    chunk_index=global_idx,
                    content=content,
                    section_title=label,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            global_idx += 1

        if pos is not None:
            search_from = pos + len(body)

    if not out:
        return _chunk_plain(full_text, chunk_size=chunk_size, overlap=overlap)

    for i, ch in enumerate(out):
        ch.chunk_index = i
    return out


def _prefix_section(title: str | None, body: str) -> str:
    if title:
        return f"[{title}]\n{body}"
    return body


def _chunk_plain(full_text: str, *, chunk_size: int, overlap: int) -> list[TextChunk]:
    out: list[TextChunk] = []
    for i, (s, e, window) in enumerate(
        _slide_windows_with_offsets(full_text, chunk_size=chunk_size, overlap=overlap)
    ):
        out.append(
            TextChunk(
                chunk_index=i,
                content=window,
                section_title=None,
                char_start=s,
                char_end=e,
            )
        )
    return out


def _slide_windows_with_offsets(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, slice)`` for each overlapping window (``end`` exclusive)."""
    if not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    n = len(text)
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        out.append((i, j, text[i:j]))
        if j >= n:
            break
        i = j - overlap
    return out
