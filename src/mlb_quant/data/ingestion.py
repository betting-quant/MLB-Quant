"""Shared ingestion validation helpers."""

from datetime import datetime


def validate_as_of(as_of: datetime) -> datetime:
    """Require callers to declare the information cutoff for a data query."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return as_of
