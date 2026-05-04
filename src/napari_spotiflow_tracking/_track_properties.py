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
    cols = ["track_id", "length", "total_displacement",
            "net_displacement", "mean_speed"]
    if image is not None:
        cols += ["mean_intensity", "max_intensity", "intensity_std"]

    if len(tracked_df) == 0:
        return pd.DataFrame(columns=cols)

    if image is not None:
        if image.ndim == 3:
            t_max = int(tracked_df["frame"].max())
            if image.shape[0] <= t_max:
                raise ValueError("Image T-axis does not span tracked frames")
        elif image.ndim != 2:
            raise ValueError("Image must be 2D (Y, X) or 3D (T, Y, X)")

    rows = []
    for tid, grp in tracked_df.groupby("particle"):
        grp = grp.sort_values("frame")
        coords = grp[["y", "x"]].to_numpy()
        length = len(grp)
        if length > 1:
            diffs = np.diff(coords, axis=0)
            total = float(np.linalg.norm(diffs, axis=1).sum())
            net = float(np.linalg.norm(coords[-1] - coords[0]))
        else:
            total = 0.0
            net = 0.0
        mean_speed = total / max(length - 1, 1)
        row = {
            "track_id": int(tid),
            "length": length,
            "total_displacement": total,
            "net_displacement": net,
            "mean_speed": mean_speed,
        }
        if image is not None:
            frames = grp["frame"].astype(int).to_numpy()
            ys = np.clip(np.rint(coords[:, 0]).astype(int), 0, image.shape[-2] - 1)
            xs = np.clip(np.rint(coords[:, 1]).astype(int), 0, image.shape[-1] - 1)
            vals = image[frames, ys, xs] if image.ndim == 3 else image[ys, xs]
            row["mean_intensity"] = float(vals.mean())
            row["max_intensity"] = float(vals.max())
            row["intensity_std"] = float(vals.std())
        rows.append(row)

    return pd.DataFrame(rows, columns=cols).sort_values("track_id").reset_index(drop=True)
