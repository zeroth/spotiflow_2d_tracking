"""Pure-Python helpers for the PAPA widget.

No Qt imports; functions are unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_particles_per_frame(points: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Convert a Points layer (N, 3) array into a per-frame count DataFrame.

    Input frames are 0-indexed (napari convention); output frames are 1-indexed
    to match the canonical analysis in papa_analysis.py.

    Returns
    -------
    per_frame : DataFrame with columns ``frame`` (1-indexed int) and
        ``n_particles`` (int). Reindexed so every frame in [1, max_frame] is
        present (zero-detection frames included with ``n_particles=0``).
    n_negative_clipped : number of input rows that had negative frame indices
        (clipped to 0 before the +1 shift). Caller may surface this as a
        warning.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with columns (frame, y, x)")

    raw_frames = np.rint(points[:, 0]).astype(int)
    n_negative_clipped = int((raw_frames < 0).sum())
    frames_0idx = np.clip(raw_frames, 0, None)
    frames_1idx = frames_0idx + 1

    if frames_1idx.size == 0:
        per_frame = pd.DataFrame({"frame": [], "n_particles": []}).astype(int)
        return per_frame, n_negative_clipped

    counts = (
        pd.Series(frames_1idx)
        .value_counts()
        .sort_index()
    )
    max_frame = int(frames_1idx.max())
    full_index = pd.Index(range(1, max_frame + 1), name="frame")
    counts = counts.reindex(full_index, fill_value=0).astype(int)

    per_frame = counts.reset_index()
    per_frame.columns = ["frame", "n_particles"]
    per_frame["frame"] = per_frame["frame"].astype(int)
    return per_frame, n_negative_clipped


def build_cycle_info(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Generate cycle/group boundary information.

    Logic mirrors papa_analysis.py:47-79. Cycles and frames are 1-indexed.
    """
    raise NotImplementedError


def annotate_with_cycle_group(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
) -> pd.DataFrame:
    """Add ``cycle`` and ``group`` columns (nullable Int64) to ``per_frame``.

    Frames falling in skip-gap regions receive ``NaN`` in both columns.
    Returns a new DataFrame; does not mutate the input.
    """
    raise NotImplementedError
