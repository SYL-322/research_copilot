"""Paper-scoped Q&A using stored memory + retrieved chunks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from core.config import Settings, load_settings
from core.models import PaperMemory
from db.database import initialize_database
from db.repository import Repository
from llm.openai_client import OpenAIClient
from retrieval.paper_retriever import format_evidence_block, retrieve_relevant_chunks
from utils.files import append_jsonl
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "paper_qa_prompt.md"


class PaperQAError(Exception):
    """Raised when Q&A cannot run (missing data, invalid id, etc.)."""


def _parse_paper_id(paper_id: str) -> int:
    try:
        pid = int(str(paper_id).strip())
    except ValueError as e:
        raise PaperQAError(f"Invalid paper_id: {paper_id!r}") from e
    if pid < 1:
        raise PaperQAError(f"paper_id must be positive, got {pid}")
    return pid


def load_qa_prompt_template(project_root: Path) -> str:
    """Load ``prompts/paper_qa_prompt.md``."""
    path = project_root / "prompts" / PROMPT_FILENAME
    if not path.is_file():
        raise PaperQAError(f"QA prompt template missing: {path}")
    return path.read_text(encoding="utf-8")


def _format_chat_history(
    history: Sequence[tuple[str, str]] | None,
    *,
    max_turns: int = 12,
) -> str:
    if not history:
        return "(none — this is the first turn.)"
    lines: list[str] = []
    for user, assistant in history[-max_turns:]:
        lines.append(f"User: {user.strip()}")
        lines.append(f"Assistant: {assistant.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def _memory_json_for_prompt(memory: PaperMemory) -> str:
    return json.dumps(memory.to_memory_json(), indent=2, ensure_ascii=False)


def paper_qa_log_path(paper_id: int, *, project_root: Path, settings: Settings) -> Path:
    """Filesystem path for optional persisted paper Q&A turns."""
    data_dir = settings.resolve_data_dir(project_root)
    return data_dir / "papers" / "qa" / f"paper_{paper_id}.jsonl"


def save_paper_qa_turn(
    paper_id: str,
    question: str,
    answer: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> Path:
    """Append one paper-QA turn to local JSONL for later review."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    pid = _parse_paper_id(paper_id)
    path = paper_qa_log_path(pid, project_root=root, settings=settings)
    append_jsonl(
        path,
        {
            "paper_id": pid,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "question": question.strip(),
            "answer": answer.strip(),
        },
    )
    return path


def load_paper_qa_turns(
    paper_id: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> list[dict[str, object]]:
    """Load saved paper-QA turns from local JSONL, newest last."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    pid = _parse_paper_id(paper_id)
    path = paper_qa_log_path(pid, project_root=root, settings=settings)
    if not path.is_file():
        return []

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid paper QA log line in %s", path)
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def answer_paper_question(
    paper_id: str,
    question: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
    top_k_chunks: int = 8,
    temperature: float = 0.2,
    chat_history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """
    Answer ``question`` using stored paper memory and lexically retrieved chunks.

    Parameters
    ----------
    paper_id
        Primary key in ``papers`` (string for CLI ergonomics).
    question
        User question.
    settings
        Defaults to :func:`core.config.load_settings`.
    project_root
        Repository root (contains ``prompts/``).
    top_k_chunks
        Number of chunks to pass as evidence.
    temperature
        LLM temperature (keep low for faithfulness).
    chat_history
        Optional prior ``(user, assistant)`` turns for the same paper session.

    Raises
    ------
    PaperQAError
        Missing memory, empty question, or missing API key.
    """
    settings = settings or load_settings()
    root = project_root or default_project_root()
    if not (settings.openai_api_key or "").strip():
        raise PaperQAError("OPENAI_API_KEY is not set in the environment.")
    q = question.strip()
    if not q:
        raise PaperQAError("Question is empty.")

    pid = _parse_paper_id(paper_id)
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        memory = repo.get_paper_memory(pid)
        if memory is None:
            raise PaperQAError(
                f"No paper memory for paper_id={pid}. Run build_paper_memory first."
            )
        chunks = repo.list_chunks_for_paper(pid)
        retrieved = retrieve_relevant_chunks(
            chunks,
            q,
            top_k=top_k_chunks,
            min_score=0.01,
        )
        evidence = format_evidence_block(retrieved)
        evidence_status = (
            "Evidence retrieved: none. No direct paper excerpts matched this question. "
            "Use the paper memory only as a possibly incomplete summary; avoid overstating unsupported details."
            if not retrieved
            else f"Evidence retrieved: {len(retrieved)} chunk(s)."
        )

        template = load_qa_prompt_template(root)
        for key in (
            "{{PAPER_MEMORY_JSON}}",
            "{{EVIDENCE_STATUS}}",
            "{{EVIDENCE_CHUNKS}}",
            "{{CHAT_HISTORY}}",
            "{{QUESTION}}",
        ):
            if key not in template:
                raise PaperQAError(f"QA prompt template must contain placeholder {key}")

        user_body = (
            template.replace("{{PAPER_MEMORY_JSON}}", _memory_json_for_prompt(memory))
            .replace("{{EVIDENCE_STATUS}}", evidence_status)
            .replace("{{EVIDENCE_CHUNKS}}", evidence)
            .replace("{{CHAT_HISTORY}}", _format_chat_history(chat_history))
            .replace("{{QUESTION}}", q)
        )

        client = OpenAIClient(settings)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant. "
                    "Follow the user instructions in the following message. "
                    "Do not invent paper content. Cite chunk_index when using evidence excerpts."
                ),
            },
            {"role": "user", "content": user_body},
        ]
        model = settings.openai_model
        answer = client.chat(messages, model=model, temperature=temperature)
        if not answer.strip():
            raise PaperQAError("Model returned an empty answer.")
        logger.info("Answered paper_id=%s question=%r", pid, q[:80])
        return answer.strip()
    finally:
        conn.close()


class PaperChatSession:
    """
    Lightweight multi-turn chat for one paper (in-memory history only).

    History is a list of ``(user_question, assistant_answer)`` pairs.
    """

    def __init__(
        self,
        paper_id: str,
        *,
        settings: Settings | None = None,
        project_root: Path | None = None,
        max_history_turns: int = 12,
    ) -> None:
        self.paper_id = paper_id
        self._settings = settings
        self._project_root = project_root
        self._max_history_turns = max_history_turns
        self._history: list[tuple[str, str]] = []

    @property
    def history(self) -> list[tuple[str, str]]:
        """Copy of (user, assistant) turns."""
        return list(self._history)

    def clear(self) -> None:
        """Drop all prior turns."""
        self._history.clear()

    def ask(
        self,
        question: str,
        *,
        top_k_chunks: int = 8,
        temperature: float = 0.2,
    ) -> str:
        """Ask a question and append this turn to session history."""
        hist = self._history[-self._max_history_turns :]
        answer = answer_paper_question(
            self.paper_id,
            question,
            settings=self._settings,
            project_root=self._project_root,
            chat_history=hist,
            top_k_chunks=top_k_chunks,
            temperature=temperature,
        )
        self._history.append((question.strip(), answer))
        return answer


def run_cli() -> None:
    """``python -m memory.paper_chat <paper_id> \"question\"``."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Ask a question about an ingested paper.")
    p.add_argument("paper_id")
    p.add_argument("question", nargs="?", default="")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        if not args.question:
            print("Usage: paper_id and question required", file=sys.stderr)
            raise SystemExit(2)
        out = answer_paper_question(args.paper_id, args.question, project_root=root)
        print(out)
    except PaperQAError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    run_cli()
