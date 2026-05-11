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


def msd(pos: np.ndarray, limit: int = 25) -> pd.Series:
    """Mean squared displacement for a single track.

    Parameters
    ----------
    pos : np.ndarray of shape (N, 2)
        Per-frame [x, y] coordinates of one track, in chronological order.
    limit : int
        Maximum lag time (clipped to ``N - 1``).

    Returns
    -------
    pd.Series indexed by lag time (1..limit), values are MSD per lag.

    Adapted from
    napari-particle-tracking/src/napari_particle_tracking/libs/_tracking.py.
    """
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape (N, 2)")
    if pos.shape[0] < 2:
        raise ValueError("pos must have at least 2 rows")
    pos_columns = ["x", "y"]
    result_columns = [f"<{p}>" for p in pos_columns] + [
        f"<{p}^2>" for p in pos_columns
    ]
    limit = min(limit, len(pos[:, 0]) - 1)
    lagtimes = np.arange(1, limit + 1)
    rows = []
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        rows.append(
            np.concatenate(
                (np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0))
            )
        )
    result = pd.DataFrame(rows, columns=result_columns, index=lagtimes)
    result["msd"] = result[result_columns[-len(pos_columns):]].sum(axis=1)
    return result["msd"]


def find_steps(
    data: np.ndarray,
    window: int = 20,
    threshold: float = 0.5,
):
    """Detect step-like transitions in a 1-D time series.

    Returns ``(table, fitx, gaussian_normalise)``:
      - ``table``: list of ``[index, level_before, level_after, step_height,
        dwell_before, dwell_after, step_error]`` per detected step.
      - ``fitx``: piecewise-constant fit aligned with ``data``.
      - ``gaussian_normalise``: the normalized gaussian-filtered derivative
        used for peak detection (useful for diagnostics).

    Adapted from napari-tracking-analysis/src/napari_tracking_analysis/utils/
    __init__.py::FindSteps.
    """
    from math import sqrt

    from scipy.ndimage import gaussian_filter1d

    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return [], np.array([]), np.array([])

    gaussian_data = gaussian_filter1d(data, window, order=1)
    peak_amp = float(np.abs(gaussian_data).max())
    if peak_amp > 0:
        gaussian_normalise = gaussian_data / peak_amp
    else:
        gaussian_normalise = gaussian_data
    gaussian_normalise = np.abs(gaussian_normalise)

    peaks = np.where(gaussian_normalise > threshold, 1, 0)
    peaks_dif = np.diff(peaks)
    ups = np.where(peaks_dif == 1)[0]
    dns = np.where(peaks_dif == -1)[0]
    indices = []
    for u, d in zip(ups, dns):
        g_slice = gaussian_normalise[u:d]
        if not len(g_slice):
            continue
        loc = int(np.argmax(g_slice))
        indices.append(loc + int(u))

    last = len(indices) - 1
    table = []
    fitx = np.zeros(data.shape)
    for i, index in enumerate(indices):
        if i == 0:
            level_before = data[0:index]
            if i == last:
                level_after = data[index:]
                dwell_after = len(data) - index
                fitx[index:] = level_after.mean()
            else:
                level_after = data[index:indices[i + 1]]
                dwell_after = indices[i + 1] - index
                fitx[index:indices[i + 1]] = level_after.mean()
            dwell_before = index
            fitx[0:index] = level_before.mean() if level_before.size else 0.0
        elif i == last:
            level_before = data[indices[i - 1]:index]
            level_after = data[index:]
            dwell_before = index - indices[i - 1]
            dwell_after = len(data) - index
            fitx[indices[i - 1]:index] = level_before.mean()
            fitx[index:] = level_after.mean()
        else:
            level_before = data[indices[i - 1]:index]
            level_after = data[index:indices[i + 1]]
            dwell_before = index - indices[i - 1]
            dwell_after = indices[i + 1] - index
            fitx[indices[i - 1]:index] = level_before.mean()
            fitx[index:indices[i + 1]] = level_after.mean()

        if level_before.size and level_after.size:
            step_error = sqrt(level_after.var() + level_before.var())
            step_height = float(level_after.mean() - level_before.mean())
            level_before_mean = float(level_before.mean())
            level_after_mean = float(level_after.mean())
        else:
            step_error = 0.0
            step_height = 0.0
            level_before_mean = float(level_before.mean()) if level_before.size else 0.0
            level_after_mean = float(level_after.mean()) if level_after.size else 0.0
        table.append([index, level_before_mean, level_after_mean,
                      step_height, dwell_before, dwell_after, step_error])

    # If no steps detected, fitx is all-zero; fill with the data mean so the
    # plot still shows a reasonable baseline.
    if len(indices) == 0:
        fitx[:] = float(data.mean())

    return table, fitx, gaussian_normalise
