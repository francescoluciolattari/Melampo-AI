"""Small helpers that were written again in many modules (27 November 2026): one copy, the same behaviour."""

from __future__ import annotations

import unicodedata
from typing import Any


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """The value within [lower, upper]."""
    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """The value as a float, or the default when it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """The value as an int, or the default when it is not a number."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalise(value: str) -> str:
    """Lower case, spaces collapsed."""
    return " ".join(str(value).lower().split())


def fold(text: str) -> str:
    """Lower case without accents ("Perché" -> "perche")."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def box(mask, margin_vox) -> tuple[slice, ...]:
    """The slices of the smallest box holding a mask, widened by a margin in voxels and kept within the array."""
    import numpy as np

    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=False))


__all__ = ["box", "clamp", "fold", "normalise", "safe_float", "safe_int"]
