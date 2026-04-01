"""Simple string normalization for matching LLM output to known papers (no fuzzy match)."""


def normalize_title(title: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return " ".join((title or "").lower().split())
