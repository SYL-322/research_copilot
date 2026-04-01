"""SQLite connection (compat alias for :mod:`db.database`)."""

from __future__ import annotations

from db.database import connect as get_connection

__all__ = ["get_connection"]
