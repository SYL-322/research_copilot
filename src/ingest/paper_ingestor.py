"""Orchestrate PDF/arXiv ingestion: extract, chunk, persist, cache."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from core.config import Settings
from core.models import PaperChunk, PaperMetadata
from db.repository import Repository
from ingest.arxiv_client import (
    ArxivMetadata,
    arxiv_id_without_version,
    download_arxiv_pdf,
    fetch_arxiv_metadata,
    parse_arxiv_id_from_url_or_id,
)
from ingest.chunker import TextChunk, chunk_sections
from ingest.exceptions import IngestError, PdfLoadError, UnsupportedInputError
from ingest.pdf_loader import extract_from_pdf
from utils.files import write_json, write_text

logger = logging.getLogger(__name__)

SourceInput = Union[str, Path]


@dataclass
class IngestResult:
    """Outcome of a successful ingest run."""

    paper_id: int
    external_id: str
    metadata: PaperMetadata
    pdf_path: Path | None
    text_path: Path
    cache_json_path: Path
    chunk_count: int


def _project_root() -> Path:
    from utils.paths import project_root

    return project_root()


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:200]


def _sha256_prefix(path: Path, *, nbytes: int = 2_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = nbytes
        while remaining > 0:
            buf = f.read(min(65536, remaining))
            if not buf:
                break
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()[:24]


def ingest(
    source: SourceInput,
    *,
    settings: Settings | None = None,
    repository: Repository | None = None,
    project_root: Path | None = None,
    chunk_size: int = 3500,
    chunk_overlap: int = 400,
) -> IngestResult:
    """
    Ingest from a local ``.pdf`` path or an arXiv URL / id string.

    Parameters
    ----------
    source
        Filesystem path to a PDF, or ``https://arxiv.org/abs/...``, or ``2401.12345``.
    settings
        Defaults to :func:`core.config.load_settings`.
    repository
        Uses ``repository.connection``; if None, opens DB from settings (caller must
        close when using a transient connection — see :func:`ingest_with_connection`).
    project_root
        Repo root (parent of ``src``). Defaults to :func:`utils.paths.project_root`.
    chunk_size, chunk_overlap
        Passed to :func:`ingest.chunker.chunk_sections`.

    Raises
    ------
    UnsupportedInputError
        If ``source`` is not a PDF file or arXiv reference.
    IngestError
        Wraps arXiv, PDF, or DB failures.
    """
    from core.config import load_settings

    settings = settings or load_settings()
    root = project_root or _project_root()
    data_dir = settings.resolve_data_dir(root)
    papers_dir = data_dir / "papers"
    cache_dir = data_dir / "cache"
    papers_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    close_conn = False
    if repository is None:
        from db.database import initialize_database

        conn = initialize_database(settings=settings, project_root=root)
        close_conn = True
        repository = Repository(conn)
    else:
        repository.init_schema()

    try:
        normalized = _normalize_source(source)
        if normalized.kind == "arxiv":
            return _ingest_arxiv(
                normalized.raw or "",
                settings=settings,
                repository=repository,
                papers_dir=papers_dir,
                cache_dir=cache_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        if normalized.kind == "pdf":
            return _ingest_local_pdf(
                normalized.path,
                repository=repository,
                papers_dir=papers_dir,
                cache_dir=cache_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        raise UnsupportedInputError(f"Unsupported normalized kind: {normalized.kind!r}")
    except IngestError:
        raise
    except Exception as e:
        logger.exception("Ingest failed")
        raise IngestError(str(e)) from e
    finally:
        if close_conn:
            repository.connection.close()


@dataclass
class _Normalized:
    kind: str
    raw: str | None = None
    path: Path | None = None


def _normalize_source(source: SourceInput) -> _Normalized:
    if isinstance(source, Path):
        p = source.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".pdf":
            return _Normalized(kind="pdf", path=p)
        raise UnsupportedInputError(f"Not a PDF file: {source}")

    s = str(source).strip()
    if not s:
        raise UnsupportedInputError("Empty source string")

    p = Path(s).expanduser()
    if p.is_file() and p.suffix.lower() == ".pdf":
        return _Normalized(kind="pdf", path=p.resolve())

    if parse_arxiv_id_from_url_or_id(s):
        return _Normalized(kind="arxiv", raw=s)
    if "arxiv.org" in s.lower():
        return _Normalized(kind="arxiv", raw=s)

    raise UnsupportedInputError(
        "Expected a path to a .pdf file or an arXiv URL / id (generic URLs not supported yet)."
    )


def _ingest_arxiv(
    raw: str,
    *,
    settings: Settings,
    repository: Repository,
    papers_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    meta_arx: ArxivMetadata = fetch_arxiv_metadata(
        raw,
        timeout=settings.http_timeout,
    )
    slug = _safe_slug(meta_arx.arxiv_id_base)
    pdf_dest = papers_dir / f"{slug}.pdf"
    txt_dest = papers_dir / f"{slug}.txt"
    cache_dest = cache_dir / f"{slug}_ingest.json"

    download_arxiv_pdf(meta_arx.pdf_url, pdf_dest, timeout=max(60.0, settings.http_timeout))

    try:
        extracted = extract_from_pdf(pdf_dest)
    except PdfLoadError as e:
        raise IngestError(f"PDF text extraction failed: {e}") from e

    write_text(txt_dest, extracted.full_text)
    chunks_tc = chunk_sections(
        extracted.sections,
        extracted.full_text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    paper_chunks = _to_paper_chunks(chunks_tc)

    year = meta_arx.published.year if meta_arx.published else None
    pm = PaperMetadata(
        external_id=meta_arx.arxiv_id_base,
        title=meta_arx.title,
        authors=meta_arx.authors,
        year=year,
        venue="arXiv",
        source="arxiv",
        source_url=meta_arx.abs_url,
        pdf_path=str(pdf_dest),
        text_path=str(txt_dest),
    )
    paper_id = repository.upsert_paper(pm)
    repository.replace_paper_chunks(paper_id, _with_paper_id(paper_chunks, paper_id))

    cache_payload = {
        "arxiv": {
            "arxiv_id": meta_arx.arxiv_id,
            "arxiv_id_base": meta_arx.arxiv_id_base,
            "abstract": meta_arx.abstract,
            "pdf_url": meta_arx.pdf_url,
            "primary_category": meta_arx.primary_category,
        },
        "extraction": {
            "source": extracted.source,
            "section_count": len(extracted.sections),
        },
        "paths": {
            "pdf": str(pdf_dest),
            "text": str(txt_dest),
            "paper_id": paper_id,
        },
    }
    write_json(cache_dest, cache_payload)

    logger.info(
        "Ingested arXiv paper id=%s chunks=%s text=%s",
        paper_id,
        len(paper_chunks),
        txt_dest,
    )
    return IngestResult(
        paper_id=paper_id,
        external_id=meta_arx.arxiv_id_base,
        metadata=pm,
        pdf_path=pdf_dest,
        text_path=txt_dest,
        cache_json_path=cache_dest,
        chunk_count=len(paper_chunks),
    )


def _ingest_local_pdf(
    pdf_path: Path,
    *,
    repository: Repository,
    papers_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    digest = _sha256_prefix(pdf_path)
    external_id = f"file:{digest}"
    slug = _safe_slug(f"{pdf_path.stem}_{digest[:8]}")
    txt_dest = papers_dir / f"{slug}.txt"
    cache_dest = cache_dir / f"{slug}_ingest.json"

    try:
        extracted = extract_from_pdf(pdf_path)
    except PdfLoadError as e:
        raise IngestError(f"PDF text extraction failed: {e}") from e

    write_text(txt_dest, extracted.full_text)
    chunks_tc = chunk_sections(
        extracted.sections,
        extracted.full_text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    paper_chunks = _to_paper_chunks(chunks_tc)

    title_guess = pdf_path.stem.replace("_", " ").replace("-", " ")
    pm = PaperMetadata(
        external_id=external_id,
        title=title_guess,
        authors=[],
        year=None,
        venue=None,
        source="upload",
        source_url=None,
        pdf_path=str(pdf_path.resolve()),
        text_path=str(txt_dest),
    )
    paper_id = repository.upsert_paper(pm)
    repository.replace_paper_chunks(paper_id, _with_paper_id(paper_chunks, paper_id))

    cache_payload = {
        "local_pdf": {"path": str(pdf_path.resolve()), "external_id": external_id},
        "extraction": {
            "source": extracted.source,
            "section_count": len(extracted.sections),
        },
        "paths": {"text": str(txt_dest), "paper_id": paper_id},
    }
    write_json(cache_dest, cache_payload)

    logger.info(
        "Ingested local PDF id=%s chunks=%s text=%s",
        paper_id,
        len(paper_chunks),
        txt_dest,
    )
    return IngestResult(
        paper_id=paper_id,
        external_id=external_id,
        metadata=pm,
        pdf_path=pdf_path.resolve(),
        text_path=txt_dest,
        cache_json_path=cache_dest,
        chunk_count=len(paper_chunks),
    )


def _to_paper_chunks(chunks: list[TextChunk]) -> list[PaperChunk]:
    out: list[PaperChunk] = []
    for tc in chunks:
        out.append(
            PaperChunk(
                paper_id=0,
                chunk_index=tc.chunk_index,
                content=tc.content,
                char_start=tc.char_start,
                char_end=tc.char_end,
                section_title=tc.section_title,
            )
        )
    return out


def _with_paper_id(chunks: list[PaperChunk], paper_id: int) -> list[PaperChunk]:
    return [c.model_copy(update={"paper_id": paper_id}) for c in chunks]


def ingest_with_connection(
    source: SourceInput,
    conn: sqlite3.Connection,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
    chunk_size: int = 3500,
    chunk_overlap: int = 400,
) -> IngestResult:
    """
    Like :func:`ingest` but uses an existing SQLite connection (caller manages lifecycle).
    """
    from core.config import load_settings

    settings = settings or load_settings()
    repo = Repository(conn)
    repo.init_schema()
    root = project_root or _project_root()
    data_dir = settings.resolve_data_dir(root)
    papers_dir = data_dir / "papers"
    cache_dir = data_dir / "cache"
    papers_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_source(source)
    if normalized.kind == "arxiv":
        return _ingest_arxiv(
            normalized.raw or "",
            settings=settings,
            repository=repo,
            papers_dir=papers_dir,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if normalized.kind == "pdf" and normalized.path:
        return _ingest_local_pdf(
            normalized.path,
            repository=repo,
            papers_dir=papers_dir,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    raise UnsupportedInputError(f"Unsupported source: {source!r}")


def run_cli_example() -> None:
    """
    Minimal CLI-style example: ``python -m ingest.paper_ingestor <arxiv-url-or-id>``.

    Requires ``OPENAI_API_KEY`` only if other code imports settings that validate it;
    ingestion itself does not call the LLM.
    """
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Ingest one arXiv paper or local PDF.")
    p.add_argument("source", help="arXiv URL/id or path to .pdf")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        result = ingest(args.source, project_root=root)
        print(
            json.dumps(
                {
                    "paper_id": result.paper_id,
                    "external_id": result.external_id,
                    "title": result.metadata.title,
                    "chunks": result.chunk_count,
                    "text_path": str(result.text_path),
                    "cache": str(result.cache_json_path),
                },
                indent=2,
            )
        )
    except IngestError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    run_cli_example()
