#!/usr/bin/env python3
"""
Research Copilot — SSH-friendly CLI (no extra dependencies).

Run from the repository root:

    python cli.py --help
    python cli.py ingest 1706.03762
    python cli.py memory 1
    python cli.py ask 1 "What is the main contribution?"
    python cli.py topic "diffusion models"
    python cli.py digest --days 5 topic_a topic_b
    python cli.py subscribe "my topic"   # optional: for digest without explicit topic args
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def cmd_ingest(args: argparse.Namespace) -> int:
    from ingest.exceptions import IngestError
    from ingest.paper_ingestor import ingest
    from utils.paths import project_root

    try:
        result = ingest(args.source, project_root=project_root())
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
        return 0
    except IngestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_memory(args: argparse.Namespace) -> int:
    from memory.paper_memory_builder import PaperMemoryBuildError, build_paper_memory
    from utils.paths import project_root

    try:
        mem = build_paper_memory(args.paper_id, project_root=project_root())
        print(json.dumps(mem.to_memory_json(), indent=2, ensure_ascii=False))
        return 0
    except PaperMemoryBuildError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_ask(args: argparse.Namespace) -> int:
    from memory.paper_chat import PaperQAError, answer_paper_question
    from utils.paths import project_root

    q = " ".join(args.question).strip()
    if not q:
        print("Error: question is empty.", file=sys.stderr)
        return 2
    try:
        out = answer_paper_question(args.paper_id, q, project_root=project_root())
        print(out)
        return 0
    except PaperQAError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_topic(args: argparse.Namespace) -> int:
    from topic.topic_service import TopicScanError, build_topic_report
    from utils.paths import project_root

    try:
        report = build_topic_report(
            args.topic,
            max_papers=args.max_papers,
            force_refresh=args.force,
            force_high_quality=getattr(args, "high_quality", False),
            project_root=project_root(),
        )
        print(report.report_md_path or "", file=sys.stderr)
        print(report.summary or "")
        return 0
    except TopicScanError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_digest(args: argparse.Namespace) -> int:
    from digest.digest_builder import DigestBuildError, build_daily_digest
    from utils.paths import project_root

    topics = list(args.topics) if args.topics else None
    try:
        d = build_daily_digest(
            topics=topics,
            days_back=args.days,
            max_per_topic=args.max_per_topic,
            project_root=project_root(),
        )
        print(d.digest_md_path or "", file=sys.stderr)
        print("items:", len(d.items))
        return 0
    except DigestBuildError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_subscribe(args: argparse.Namespace) -> int:
    """Add a digest topic subscription (SQLite)."""
    from digest.subscription_service import SubscriptionServiceError, subscribe
    from utils.paths import project_root

    try:
        sub = subscribe(args.topic, project_root=project_root())
        print(json.dumps({"topic": sub.topic, "slug": sub.slug, "id": sub.id}, indent=2))
        return 0
    except SubscriptionServiceError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_subscriptions(args: argparse.Namespace) -> int:
    from digest.subscription_service import list_subscriptions
    from utils.paths import project_root

    subs = list_subscriptions(active_only=not args.all, project_root=project_root())
    for s in subs:
        flag = "active" if s.is_active else "inactive"
        print(f"{s.slug}\t{s.topic}\t{flag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Research Copilot — ingest papers, build memory, topic scan, digest (run from repo root).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="Ingest PDF path or arXiv URL/id")
    pi.add_argument("source", help="Path to .pdf or arXiv id/URL")
    pi.set_defaults(func=cmd_ingest)

    pm = sub.add_parser("memory", help="Build structured paper memory (LLM); needs OPENAI_API_KEY")
    pm.add_argument("paper_id", help="Integer id from papers table (see ingest output)")
    pm.set_defaults(func=cmd_memory)

    pa = sub.add_parser("ask", help="Ask a question (needs memory built first)")
    pa.add_argument("paper_id", help="papers.id")
    pa.add_argument(
        "question",
        nargs=argparse.REMAINDER,
        help="Question text (quote for multiple words)",
    )
    pa.set_defaults(func=cmd_ask)

    pt = sub.add_parser("topic", help="Scan literature and write a topic report")
    pt.add_argument("topic", help="Research topic string")
    pt.add_argument("--max-papers", type=int, default=30, dest="max_papers")
    pt.add_argument("--force", action="store_true", help="Ignore cache")
    pt.add_argument(
        "--high-quality",
        action="store_true",
        dest="high_quality",
        help="Always run main-model refinement pass",
    )
    pt.set_defaults(func=cmd_topic)

    pd = sub.add_parser("digest", help="Build daily digest (topics or active subscriptions)")
    pd.add_argument(
        "topics",
        nargs="*",
        help="Topic strings; omit to use active subscriptions",
    )
    pd.add_argument("--days", type=int, default=3, help="Publication window (days)")
    pd.add_argument(
        "--max-per-topic",
        type=int,
        default=15,
        dest="max_per_topic",
        help="Cap papers per topic before merge",
    )
    pd.set_defaults(func=cmd_digest)

    ps = sub.add_parser("subscribe", help="Subscribe a topic for digest (optional; use with digest)")
    ps.add_argument("topic", help="Topic label")
    ps.set_defaults(func=cmd_subscribe)

    pl = sub.add_parser("subscriptions", help="List digest subscriptions")
    pl.add_argument("-a", "--all", action="store_true", help="Include inactive")
    pl.set_defaults(func=cmd_subscriptions)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
