"""Per-track property computation for the Tracks layer's filter panel.

Pure-Python; no Qt imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Motility classification thresholds on the anomalous-diffusion exponent α,
# after Nature Communications Biology 4, 137 (2021), "Motility analysis":
# α < 0.4 constrained, 0.4 < α < 1.2 diffusive, α > 1.2 directed.
# The exact boundary values are assigned to "diffusive".
ALPHA_CONSTRAINED_MAX = 0.4
ALPHA_DIRECTED_MIN = 1.2
MOTILITY_CLASSES = ("constrained", "diffusive", "directed")


def classify_motility(alpha) -> str:
    """Bin an anomalous-diffusion exponent into a motility class.

    Returns one of :data:`MOTILITY_CLASSES`, or ``"unclassified"`` when
    ``alpha`` is None/NaN (e.g. a track whose MSD fit did not converge).
    """
    if alpha is None:
        return "unclassified"
    a = float(alpha)
    if not np.isfinite(a):
        return "unclassified"
    if a < ALPHA_CONSTRAINED_MAX:
        return "constrained"
    if a > ALPHA_DIRECTED_MIN:
        return "directed"
    return "diffusive"


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


def add_physical_columns(
    df: pd.DataFrame,
    pixel_size_nm: float | None = None,
    frame_interval_s: float | None = None,
) -> pd.DataFrame:
    """Append calibrated copies of the geometry columns to a track table.

    ``compute_track_properties`` works in the units napari uses -- pixels and
    frames. This adds ``duration_s``, ``total_displacement_um``,
    ``net_displacement_um`` and ``mean_speed_um_s`` beside them, leaving the
    pixel columns alone: the bulk script's ``--summary-only`` and ``--compare``
    paths read those back and scale them themselves, so changing their meaning
    would double-convert, and ``length`` in frames is what
    ``--min-track-length`` cuts on.

    Both parameters are needed. Half a calibration is no calibration -- a µm
    column derived from a missing pixel size would be pixels wearing the wrong
    label -- so anything less returns ``df`` unchanged.

    A track's duration spans the intervals *between* its localisations, so an
    n-frame track lasts ``(n - 1) * frame_interval_s``. A single-frame track
    therefore has zero duration and an undefined speed (NaN rather than an
    infinity, which would otherwise own every axis limit downstream).
    """
    if pixel_size_nm is None or frame_interval_s is None:
        return df

    out = df.copy()
    um_per_px = pixel_size_nm / 1000.0
    duration = (out["length"].astype(float) - 1.0) * float(frame_interval_s)
    out["duration_s"] = duration
    out["total_displacement_um"] = out["total_displacement"].astype(float) * um_per_px
    out["net_displacement_um"] = out["net_displacement"].astype(float) * um_per_px
    out["mean_speed_um_s"] = (
        out["total_displacement_um"] / duration.where(duration > 0)
    )
    return out


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


def msd_stats(pos: np.ndarray, limit: int = 25) -> pd.DataFrame:
    """Per-lag MSD with its spread and pair count, for MSD error bars.

    Same mean as :func:`msd`, plus the standard error and the number of
    displacement pairs averaged at each lag.

    Parameters
    ----------
    pos : np.ndarray of shape (N, 2)
        Per-frame [x, y] coordinates of one track, in chronological order.
    limit : int
        Maximum lag time (clipped to ``N - 1``).

    Returns
    -------
    pd.DataFrame indexed by lag time (1..limit) with columns ``msd``, ``sem``
    and ``n_pairs``. ``sem`` is NaN where only one pair contributes.

    Notes
    -----
    ``sem`` is the conventional standard error over displacements, but those
    displacements overlap in time and are therefore correlated — so it
    understates the true uncertainty, increasingly so at long lags. Treat the
    error bars as indicative rather than as confidence intervals.
    """
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape (N, 2)")
    if pos.shape[0] < 2:
        raise ValueError("pos must have at least 2 rows")

    limit = min(limit, pos.shape[0] - 1)
    lagtimes = np.arange(1, limit + 1)
    means, sems, counts = [], [], []
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        squared = np.nansum(diff ** 2, axis=1)  # |Δr|² per displacement pair
        n = int(np.count_nonzero(~np.isnan(squared)))
        means.append(float(np.nanmean(squared)))
        counts.append(n)
        sems.append(float(np.nanstd(squared, ddof=1) / np.sqrt(n)) if n > 1
                    else float("nan"))

    return pd.DataFrame(
        {"msd": means, "sem": sems, "n_pairs": counts}, index=lagtimes,
    )


def msd_basic_fit(
    msd_series: pd.Series,
    delta: float = 1.0,
    sigma: float = 0.0,
    maxfev: int = 1_000_000,
) -> dict:
    """Fit the anomalous-diffusion model to an MSD curve.

    The model is ``msd = 4 * D * t^alpha + 4 * sigma^2``, where the constant
    term is the static localization-error floor. With the default
    ``sigma=0`` this reduces to the plain ``4 * D * t^alpha``.

    Parameters
    ----------
    msd_series : pd.Series
        Output of :func:`msd` — indexed by integer lag time (in frames).
    delta : float
        Real-world time per frame (e.g. seconds). Default 1.0 keeps lag in
        frames so D has units of px²/frame^alpha and alpha is unitless.
    sigma : float
        Fixed (not fitted) localization error, in the same length units as
        the coordinates the MSD was computed from. Only meaningful when the
        coordinates are calibrated — e.g. 0.035 µm for the 35 nm figure in
        Nature Communications Biology 4, 137 (2021).
    maxfev : int
        Maximum function evaluations for ``scipy.optimize.curve_fit``.

    Returns
    -------
    dict with keys:
      - ``"x"`` : lag-time array used for the fit (same length as input).
      - ``"y_fit"`` : fitted MSD values at those lag times.
      - ``"diffusion_coefficient"`` (float, D).
      - ``"alpha"`` (float, anomalous diffusion exponent).

    Adopted from https://github.com/zeroth/cellphy
    """
    from scipy.optimize import curve_fit

    y = np.asarray(msd_series, dtype=float)
    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("Need at least 2 finite MSD values to fit")

    x = np.arange(1, len(y) + 1, dtype=float) * float(delta)
    offset = 4.0 * float(sigma) ** 2

    def _fit_function(t, d, alfa):
        return (4.0 * d) * np.power(t, alfa) + offset

    best_value, _ = curve_fit(
        _fit_function, x, y, p0=np.array([0.001, 0.01]), maxfev=maxfev,
    )
    d, alpha = float(best_value[0]), float(best_value[1])
    y_fit = _fit_function(x, d, alpha)
    return {
        "x": x,
        "y_fit": y_fit,
        "diffusion_coefficient": d,
        "alpha": alpha,
    }


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

    Adopted from https://github.com/zeroth/step_detection
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
