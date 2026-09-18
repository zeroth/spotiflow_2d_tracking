"""MSD / motility computation: fits, ensemble average, class counts, metrics.

No plotting, no file writing, no printing — those belong to the figures and
group-summary tasks that build on this module. Ported from
``scripts/bulk_track_analysis.py`` (``compute_msd_analysis``, ``ensemble_msd``,
``class_counts``, ``_directed_metrics``, and the ``track_props`` block of
``analyze_folder``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...._track_properties import (
    MOTILITY_CLASSES,
    add_physical_columns,
    classify_motility,
    compute_track_properties,
    msd_basic_fit,
    msd_stats,
)
from .units import Units

MSD_CURVE_COLS = ["track_id", "lag", "msd", "sem", "n_pairs"]
MSD_FIT_COLS = ["track_id", "length", "n_lags", "D", "alpha", "motility_class"]
# Lags supported by fewer than this many tracks are too noisy to average.
MIN_TRACKS_PER_LAG = 3


def _empty_msd_frames() -> tuple:
    """Empty MSD frames with real dtypes.

    ``pd.DataFrame(columns=[...])`` would give every column object dtype, and
    ``np.isfinite`` raises a TypeError on those — so a stack with no fittable
    track would crash the plotting instead of drawing an empty figure.
    """
    curves = pd.DataFrame({
        "track_id": pd.Series(dtype="int64"),
        "lag": pd.Series(dtype="float64"),
        "msd": pd.Series(dtype="float64"),
        "sem": pd.Series(dtype="float64"),
        "n_pairs": pd.Series(dtype="int64"),
    })
    fits = pd.DataFrame({
        "track_id": pd.Series(dtype="int64"),
        "length": pd.Series(dtype="int64"),
        "n_lags": pd.Series(dtype="int64"),
        "D": pd.Series(dtype="float64"),
        "alpha": pd.Series(dtype="float64"),
        "motility_class": pd.Series(dtype="object"),
    })
    return curves[MSD_CURVE_COLS], fits[MSD_FIT_COLS]


def compute_msd_analysis(
    tracked_df: pd.DataFrame,
    units: Units,
    max_lag: int,
    min_length: int,
) -> tuple:
    """Per-track MSD curves and anomalous-diffusion fits.

    Fits ``msd = 4·D·t^α + 4·σ²`` per track and bins α into a motility class
    (constrained / diffusive / directed) following Commun Biol 4, 137 (2021).

    Returns ``(curves, fits)``:
      - ``curves``: long form — one row per (track_id, lag).
      - ``fits``: one row per track — D, alpha, motility_class. Tracks whose
        fit fails to converge get NaN and ``"unclassified"`` rather than
        aborting the run.
    """
    if "particle" not in tracked_df.columns or len(tracked_df) == 0:
        return _empty_msd_frames()

    # msd() needs ≥2 positions and the fit needs ≥2 lag points → ≥3 frames.
    min_length = max(int(min_length), 3)
    curve_parts, fit_rows = [], []

    for tid, grp in tracked_df.groupby("particle"):
        grp = grp.sort_values("frame")
        if len(grp) < min_length:
            continue
        # _track_properties.msd expects (N, 2) as [x, y].
        coords_xy = grp[["x", "y"]].to_numpy(dtype=float) * units.length_scale
        try:
            stats = msd_stats(coords_xy, limit=max_lag)
        except ValueError:
            continue

        values = stats["msd"].to_numpy(dtype=float)
        if len(stats) < 2 or not np.all(np.isfinite(values)):
            continue
        series = stats["msd"]

        curve_parts.append(pd.DataFrame({
            "track_id": int(tid),
            "lag": stats.index.to_numpy(dtype=float) * units.delta,
            "msd": values,
            "sem": stats["sem"].to_numpy(dtype=float),
            "n_pairs": stats["n_pairs"].to_numpy(dtype="int64"),
        }))

        d_fit, alpha = np.nan, np.nan
        try:
            fit = msd_basic_fit(series, delta=units.delta, sigma=units.sigma)
            d_fit, alpha = fit["diffusion_coefficient"], fit["alpha"]
        except Exception:
            pass  # non-convergent fit → unclassified, keep the curve

        fit_rows.append({
            "track_id": int(tid),
            "length": int(len(grp)),
            "n_lags": int(len(series)),
            "D": d_fit,
            "alpha": alpha,
            "motility_class": classify_motility(alpha),
        })

    if not fit_rows:
        return _empty_msd_frames()

    curves = pd.concat(curve_parts, ignore_index=True)[MSD_CURVE_COLS]
    fits = pd.DataFrame(fit_rows, columns=MSD_FIT_COLS)
    return curves, fits


def ensemble_msd(curves: pd.DataFrame) -> pd.DataFrame:
    """Track-averaged MSD per lag, restricted to well-supported lags.

    Drops the tail where fewer than ``MIN_TRACKS_PER_LAG`` tracks contribute.
    Since every track contributes a contiguous run of lags starting at the
    first lag, the survivors stay a contiguous prefix — which is what
    ``msd_basic_fit`` assumes when it rebuilds its lag axis.

    ``sem`` is the standard error of the per-track MSD values contributing to
    a lag (``ddof=1``, NaN where only one track contributes) — distinct from
    ``curves["sem"]``, which is the per-track error over displacement pairs.
    """
    if len(curves) == 0:
        return pd.DataFrame(columns=["lag", "msd", "sem", "n_tracks"])
    agg = (
        curves.groupby("lag")["msd"]
        .agg(msd="mean", sem="sem", n_tracks="size")
        .reset_index()
        .sort_values("lag")
    )
    return agg[agg["n_tracks"] >= MIN_TRACKS_PER_LAG].reset_index(drop=True)


def class_counts(fits: pd.DataFrame) -> dict:
    """Track count per motility class, always in :data:`MOTILITY_CLASSES` order.

    ``unclassified`` is included only when non-empty, so a clean run doesn't
    carry an always-zero fourth category through every figure.
    """
    if len(fits) == 0:
        return {c: 0 for c in MOTILITY_CLASSES}
    col = fits["motility_class"]
    counts = {c: int((col == c).sum()) for c in MOTILITY_CLASSES}
    n_unc = int((col == "unclassified").sum())
    if n_unc:
        counts["unclassified"] = n_unc
    return counts


def directed_metrics(df: pd.DataFrame, units: Units) -> pd.DataFrame:
    """Directed tracks with run length and velocity, in working units.

    ``_disp`` is the net (start-to-end) displacement, i.e. the run length, and
    ``_vel`` is that displacement over the track's duration. Velocity here is
    deliberately not ``mean_speed``: that is total path length per frame, which
    stays high for a track that wanders energetically without getting
    anywhere. For processive transport the net rate is the quantity of
    interest.
    """
    cols = list(df.columns) + ["_disp", "_vel"]
    if "motility_class" not in df.columns or not len(df):
        return pd.DataFrame(columns=cols)
    d = df[df["motility_class"] == "directed"].copy()
    if not len(d):
        return pd.DataFrame(columns=cols)
    steps = (d["length"].astype(float) - 1.0).clip(lower=1.0)
    net = d["net_displacement"].astype(float)
    d["_disp"] = net * units.length_scale
    d["_vel"] = (net / steps) * units.length_scale / units.delta
    return d


def track_table(
    tracked: pd.DataFrame,
    image,
    fits: pd.DataFrame,
    units: Units,
) -> pd.DataFrame:
    """Per-track geometry/intensity table, with MSD fit and physical columns.

    Ported from the ``track_props`` block of ``analyze_folder``: per-track
    geometry (and intensity, when ``image`` is given) from
    ``compute_track_properties``, with ``D``/``alpha``/``motility_class``
    merged on from ``fits`` (tracks with no fit — too short, or below
    ``min_length`` — get ``"unclassified"`` rather than NaN) and calibrated
    copies of the geometry columns appended beside the pixel ones: downstream
    summaries read this table back and do their own scaling, and ``length``
    in frames is what ``min_track_length`` cuts on.
    """
    table = compute_track_properties(tracked, image=image)

    # Carry D / alpha / class onto the per-track table too.
    table = table.merge(
        fits[["track_id", "D", "alpha", "motility_class"]],
        on="track_id", how="left",
    )
    table["motility_class"] = table["motility_class"].fillna("unclassified")
    # Calibrated copies of the geometry columns, beside the pixel ones rather
    # than over them: downstream summaries read this table back and do their
    # own scaling, and `length` in frames is what min_track_length cuts on.
    table = add_physical_columns(
        table, units.calibration.pixel_size_nm, units.calibration.frame_interval_s,
    )
    return table
