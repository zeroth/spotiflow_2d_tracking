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
    cycle_info: list[dict] = []
    cycle_num = 0
    current_frame = 1

    while current_frame <= max_time:
        cycle_num += 1

        g1_start = current_frame
        g1_end = min(current_frame + group1_size - 1, max_time)
        cycle_info.append({
            "cycle": cycle_num,
            "group": 1,
            "start_frame": g1_start,
            "end_frame": g1_end,
        })
        current_frame = g1_end + 1 + skip_frames

        if current_frame <= max_time:
            g2_start = current_frame
            g2_end = min(current_frame + group2_size - 1, max_time)
            cycle_info.append({
                "cycle": cycle_num,
                "group": 2,
                "start_frame": g2_start,
                "end_frame": g2_end,
            })
            current_frame = g2_end + 1 + skip_frames

    return cycle_info


def annotate_with_cycle_group(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
) -> pd.DataFrame:
    """Add ``cycle`` and ``group`` columns (nullable Int64) to ``per_frame``.

    Frames falling in skip-gap regions receive ``NaN`` in both columns.
    Returns a new DataFrame; does not mutate the input.
    """
    out = per_frame.copy()
    cycle_col = pd.array([pd.NA] * len(out), dtype="Int64")
    group_col = pd.array([pd.NA] * len(out), dtype="Int64")

    frame_to_idx = {int(f): i for i, f in enumerate(out["frame"].tolist())}
    for info in cycle_info:
        for f in range(info["start_frame"], info["end_frame"] + 1):
            i = frame_to_idx.get(f)
            if i is not None:
                cycle_col[i] = info["cycle"]
                group_col[i] = info["group"]

    out["cycle"] = cycle_col
    out["group"] = group_col
    return out


def compute_per_cycle_summary(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
    label1: str = "G1",
    label2: str = "G2",
) -> pd.DataFrame:
    """Aggregate ``per_frame`` counts to per-cycle totals and ratios.

    Returns a DataFrame with one row per cycle and columns:
      ``cycle``, ``<label1>_count``, ``<label2>_count``,
      ``<label1>_over_<label2>``, ``<label2>_over_<label1>``.

    Ratios use NaN when the denominator is zero. This is the data behind the
    bar/ratio subplots in :class:`PAPAWidget`'s plots tab — exposed here so it
    can be exported as well as rendered.
    """
    cycles = sorted({info["cycle"] for info in cycle_info})
    rows = []
    for cyc in cycles:
        g1_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 1]
        g2_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 2]

        g1_total = 0
        if g1_info:
            mask = (
                (per_frame["frame"] >= g1_info[0]["start_frame"])
                & (per_frame["frame"] <= g1_info[0]["end_frame"])
            )
            g1_total = int(per_frame.loc[mask, "n_particles"].sum())

        g2_total = 0
        if g2_info:
            mask = (
                (per_frame["frame"] >= g2_info[0]["start_frame"])
                & (per_frame["frame"] <= g2_info[0]["end_frame"])
            )
            g2_total = int(per_frame.loc[mask, "n_particles"].sum())

        rows.append({
            "cycle": int(cyc),
            f"{label1}_count": g1_total,
            f"{label2}_count": g2_total,
            f"{label1}_over_{label2}":
                g1_total / g2_total if g2_total > 0 else float("nan"),
            f"{label2}_over_{label1}":
                g2_total / g1_total if g1_total > 0 else float("nan"),
        })

    return pd.DataFrame(rows)
