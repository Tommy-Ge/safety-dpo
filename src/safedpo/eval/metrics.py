"""Small statistics helpers."""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(flags: list[bool]) -> dict:
    n, k = len(flags), sum(bool(f) for f in flags)
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n if n else 0.0, "k": k, "n": n, "ci95": [lo, hi]}


def relative_reduction(before: float, after: float) -> float | None:
    """(before - after) / before, e.g. ASR 0.40 -> 0.10 is a 75% relative reduction."""
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before
