"""Per-track property computation for the Tracks layer's filter panel.

Pure-Python; no Qt imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute per-track geometric (and optionally intensity) properties.

    Input ``tracked_df`` must have columns: frame (int), y (float), x (float),
    particle (int) — the output of ``track_blobs``.

    When ``image`` is given (2D (Y,X) or 3D (T,Y,X)), intensity stats are
    sampled at each track vertex's nearest integer coordinates.

    Returns a DataFrame with one row per track, sorted by ``track_id``.
    """
    raise NotImplementedError
