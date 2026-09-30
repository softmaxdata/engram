"""Shared embedding-similarity helpers used by storage backends and the
materialization engine. Keeping one definition avoids subtle drift between
SQLite, Postgres, and materialization paths."""

from __future__ import annotations

import math


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1.0, 1.0]; 0.0 for invalid/empty/mismatched inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def validated_cosine_similarity(a, b) -> float | None:
    """Return None when comparison is unknown, never a false finite nonmatch."""
    try:
        if not a or not b or len(a) != len(b):
            return None
        if not all(math.isfinite(value) for value in (*a, *b)):
            return None
        norm_a, norm_b = math.hypot(*a), math.hypot(*b)
        denominator = norm_a * norm_b
        if not math.isfinite(denominator) or denominator <= 0:
            return None
        dot = sum(x * y for x, y in zip(a, b))
        similarity = dot / denominator
        if not math.isfinite(similarity):
            return None
        return max(-1.0, min(1.0, similarity))
    except (TypeError, ValueError, OverflowError):
        return None
