"""Bulk track analysis CLI.

Headless batch pipeline: preprocess, detect (Spotiflow), fit, mask, track
(trackpy), and analyze every .tif/.tiff stack in an input directory. One
sub-folder of artifacts per input file.

Analysis includes an MSD pass: each track is fitted with
``msd = 4·D·t^α + 4·σ²`` and binned by the scaling exponent α into
constrained (α < 0.4), diffusive (0.4 ≤ α ≤ 1.2) or directed (α > 1.2), after
Nature Communications Biology 4, 137 (2021).

This script reuses the napari-spotiflow-tracking plugin's processing functions
directly (no napari / Qt / UI). Install the plugin into the same environment
first (e.g. ``pip install -e .``). See
docs/superpowers/specs/2026-05-08-bulk-track-analysis-cli-design.md
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting — no display needed
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import pandas as pd
import tifffile
from tqdm import tqdm

from napari_spotiflow_tracking._fitting import fit_all_frames_gpu, make_gpu_spot_mask
from napari_spotiflow_tracking._preprocessing import (
    _get_device,
    apply_calculation,
    gaussian_blur,
    walking_average,
)
from napari_spotiflow_tracking._segmentation import (
    detect_spots,
    detect_spots_log,
    load_model,
)
from napari_spotiflow_tracking._track_properties import (
    ALPHA_CONSTRAINED_MAX,
    ALPHA_DIRECTED_MIN,
    MOTILITY_CLASSES,
    add_physical_columns,
    classify_motility,
    compute_track_properties,
    msd_basic_fit,
)
from napari_spotiflow_tracking._track_properties import msd_stats
from napari_spotiflow_tracking._tracking import track_blobs

# Categorical hues for the motility classes — matplotlib's tab: palette, the
# same family scripts/papa_analysis.py draws with. Assigned in fixed order,
# never cycled.
CLASS_COLORS = {
    "constrained": "tab:blue",
    "diffusive": "tab:orange",
    "directed": "tab:green",
    "unclassified": "tab:gray",
}
# Lag index is ordered, not categorical — one hue, light→dark, never a rainbow.
LAG_RAMP = ["#bcd7f5", "#8bb9ec", "#5a9ae2", "#2a78d6", "#17457f"]
MSD_CURVE_COLS = ["track_id", "lag", "msd", "sem", "n_pairs"]
MSD_FIT_COLS = ["track_id", "length", "n_lags", "D", "alpha", "motility_class"]
# Lags supported by fewer than this many tracks are too noisy to average.
MIN_TRACKS_PER_LAG = 3
# Every figure is saved at the same height, and every single-panel figure at the
# same width, so the PNGs drop into a manuscript layout without rescaling.
# Multi-panel figures keep the height and widen by FIG_W_PANEL per panel.
FIG_H = 5.0
FIG_W = 7.0
FIG_W_PANEL = 4.5
# Type is sized for a figure printed at column width, so the axis values and
# labels stay legible after downscaling. Bigger type means fewer tick labels
# fit — _thin_ticks drops alternate ones at save time to make room.
FONT_TICK = 17
FONT_LABEL = 19
FONT_TITLE = 15
FONT_ANNOT = 14          # in-plot annotations and legends
FONT_ANNOT_SMALL = 12    # the same, where a panel is too tight for FONT_ANNOT
plt.rcParams.update({
    "font.size": FONT_TICK,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_ANNOT,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
})
# Output resolution. Figure sizes are in inches, so this alone sets the pixel
# dimensions: 300 dpi gives 2100 x 1500 px per single-panel figure. Physical
# size and type size are unchanged, so _thin_ticks still sees the same layout.
FIG_DPI = 300
# The archetype figure is exported at a fixed full-width size, split evenly over
# its three panels: 6800 x 1500 px.
FIG_W_ARCHETYPES = 6800 / FIG_DPI

# ── segmentation ────────────────────────────────────────────────────


def _detect_all_frames(
    stack: np.ndarray,
    model,
    args: Args,
) -> pd.DataFrame:
    """Run the chosen detector per frame; return time, y, x.

    Thin loop over ``_segmentation.detect_spots`` (Spotiflow) or
    ``_segmentation.detect_spots_log`` (Laplacian of Gaussian), both of which
    work on a single 2D frame. The column name ``time`` matches what
    ``fit_all_frames_gpu`` expects, so the rest of the pipeline does not care
    which detector produced the spots.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {stack.ndim}D")

    rows_t, rows_y, rows_x = [], [], []
    for t in tqdm(range(stack.shape[0]), desc="  detect", leave=False, unit="frame"):
        if args.detector == "log":
            points = detect_spots_log(
                stack[t],
                min_sigma=args.log_min_sigma,
                max_sigma=args.log_max_sigma,
                num_sigma=args.log_num_sigma,
                threshold=args.log_threshold,
            )
        else:
            points, _ = detect_spots(
                stack[t], model,
                prob_thresh=args.prob_thresh, min_distance=args.min_distance,
            )
        if len(points) == 0:
            continue
        rows_t.extend([t] * len(points))
        rows_y.extend(points[:, 0].astype(float).tolist())
        rows_x.extend(points[:, 1].astype(float).tolist())

    return pd.DataFrame({"time": rows_t, "y": rows_y, "x": rows_x})


# ── MSD / motility ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Calibration:
    """Optional physical calibration for the MSD analysis.

    When both ``pixel_size_nm`` and ``frame_interval_s`` are supplied, lengths
    are converted px → µm and lags frames → s, so D comes out in µm²/s^α and
    the fixed localization error ``sigma_nm`` (35 nm in Commun Biol 4, 137)
    is meaningful. Otherwise everything stays in px / frames and the σ offset
    is dropped, because a nanometre figure has no meaning in pixel units.
    """

    pixel_size_nm: float | None = None
    frame_interval_s: float | None = None
    sigma_nm: float = 35.0

    @property
    def is_physical(self) -> bool:
        return self.pixel_size_nm is not None and self.frame_interval_s is not None

    @property
    def length_scale(self) -> float:
        """Multiplier taking pixel coordinates to the working length unit."""
        px = self.pixel_size_nm
        return px / 1000.0 if self.is_physical and px is not None else 1.0

    @property
    def delta(self) -> float:
        """Working time unit per frame."""
        dt = self.frame_interval_s
        return dt if self.is_physical and dt is not None else 1.0

    @property
    def sigma(self) -> float:
        """Localization error in the working length unit (0 when uncalibrated)."""
        return self.sigma_nm / 1000.0 if self.is_physical else 0.0

    @property
    def length_unit(self) -> str:
        return "µm" if self.is_physical else "px"

    @property
    def time_unit(self) -> str:
        return "s" if self.is_physical else "frame"

    @property
    def msd_unit(self) -> str:
        return f"{self.length_unit}²"

    @property
    def d_unit(self) -> str:
        return f"{self.length_unit}²/{self.time_unit}^α"

    def describe(self) -> str:
        """One-line summary. ASCII-only — this is printed to the console, and
        a Windows cp1252 terminal cannot encode 'σ' or 'α'."""
        if not self.is_physical:
            return "uncalibrated (px, frames; sigma=0)"
        return (
            f"{self.pixel_size_nm:g} nm/px, {self.frame_interval_s:g} s/frame, "
            f"sigma={self.sigma_nm:g} nm"
        )


def _empty_msd_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    cal: Calibration,
    max_lag: int,
    min_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        coords_xy = grp[["x", "y"]].to_numpy(dtype=float) * cal.length_scale
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
            "lag": stats.index.to_numpy(dtype=float) * cal.delta,
            "msd": values,
            "sem": stats["sem"].to_numpy(dtype=float),
            "n_pairs": stats["n_pairs"].to_numpy(dtype="int64"),
        }))

        d_fit, alpha = np.nan, np.nan
        try:
            fit = msd_basic_fit(series, delta=cal.delta, sigma=cal.sigma)
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
    """
    if len(curves) == 0:
        return pd.DataFrame(columns=["lag", "msd", "n_tracks"])
    agg = (
        curves.groupby("lag")["msd"]
        .agg(msd="mean", n_tracks="size")
        .reset_index()
        .sort_values("lag")
    )
    return agg[agg["n_tracks"] >= MIN_TRACKS_PER_LAG].reset_index(drop=True)


# ── plotting ────────────────────────────────────────────────────────


def _axis_label(kind: str, cal: Calibration) -> str:
    """Axis label for a per-track quantity, in the working units.

    The uncalibrated strings are the ones these figures have always carried,
    so a run without ``--pixel-size-nm`` looks exactly as it did before the
    calibrated variants existed.
    """
    if not cal.is_physical:
        return {
            "duration": "Track length (frames)",
            "path_length": "Path length (px)",
            "speed": "Mean speed (px / frame)",
            "inst_speed": "Instantaneous speed (px / frame)",
            "cum_disp": "Cumulative displacement (px)",
            "time": "Frame",
        }[kind]
    length, time = cal.length_unit, cal.time_unit
    return {
        "duration": f"Track duration ({time})",
        "path_length": f"Path length ({length})",
        "speed": f"Mean speed ({length}/{time})",
        "inst_speed": f"Instantaneous speed ({length}/{time})",
        "cum_disp": f"Cumulative displacement ({length})",
        "time": f"Time ({time})",
    }[kind]


def _durations(track_props: pd.DataFrame, cal: Calibration) -> np.ndarray:
    """Per-track duration in the working time unit.

    A track spans the intervals *between* its localisations, so an n-frame
    track lasts (n - 1) intervals. Uncalibrated this stays the raw frame
    count, which is what the figure has always shown and what
    ``--min-track-length`` cuts on.
    """
    n = track_props["length"].to_numpy(dtype=float)
    return (n - 1.0) * cal.delta if cal.is_physical else n


def _props_in_working_units(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """A ``track_analysis.csv`` frame with ``_dur`` -- duration in working units.

    The run-level and comparison figures read these tables back off disk, and a
    folder written before ``track_analysis.csv`` carried ``duration_s`` has only
    the pixel columns -- so the duration is recomputed from them rather than the
    figure being left empty. ``length`` is deliberately left as a frame count:
    it is what ``--min-track-length`` cuts on.
    """
    out = add_physical_columns(df, cal.pixel_size_nm, cal.frame_interval_s)
    if out is df:
        out = df.copy()
    out["_dur"] = (out["duration_s"] if cal.is_physical
                   else out["length"].astype(float))
    return out


def _annotate_corner(ax, text: str) -> None:
    """Top-right annotation, stroked so it stays legible over the data."""
    ax.annotate(
        text, xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=FONT_ANNOT, color="black",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
    )


def _plot_length_hist(
    track_props: pd.DataFrame, cal: Calibration, out_path: Path,
    min_length: int | None = None,
) -> None:
    """Track-duration distribution of the analysed tracks.

    ``track_props`` is already filtered to ``--min-track-length``, so this shows
    the population that actually entered the MSD analysis, not every linked
    track. ``min_length`` is passed only so the cut can be stated on the figure
    -- in frames, because that is the unit the flag is given in, with the
    elapsed-time equivalent beside it when the run is calibrated.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    if len(track_props):
        durations = _durations(track_props, cal)
        ax.hist(durations, bins=30, color="tab:blue", zorder=2)
        bits = [f"n = {len(durations):,} analysed tracks"]
        if min_length is not None:
            cut = f"length ≥ {int(min_length)} frames"
            if cal.is_physical:
                cut += f" ({(int(min_length) - 1) * cal.delta:.3f} s)"
            bits.append(cut)
        _annotate_corner(ax, "   ·   ".join(bits))
    ax.set_xlabel(_axis_label("duration", cal), color="black")
    ax.set_ylabel("# tracks", color="black")
    ax.set_title(
        "Track duration distribution" if cal.is_physical
        else "Track length distribution", color="black")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_path)


def _plot_path_length_hist(
    track_props: pd.DataFrame, cal: Calibration, out_path: Path,
) -> None:
    """Distribution of the distance each track actually walked.

    This is the path length -- the summed frame-to-frame steps -- not the net
    displacement, and not the duration that ``length_hist`` shows. At short
    frame intervals it is inflated by localisation noise, since every jitter
    adds to the sum; the net displacement in ``directed_displacement.png`` is
    the transport quantity.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    if len(track_props):
        path = (track_props["total_displacement"].to_numpy(dtype=float)
                * cal.length_scale)
        ax.hist(path, bins=30, color="tab:blue", edgecolor="black", zorder=2)
        _annotate_corner(ax, f"n = {len(path):,} analysed tracks")
    ax.set_xlabel(_axis_label("path_length", cal), color="black")
    ax.set_ylabel("# tracks", color="black")
    ax.set_title("Path length distribution", color="black")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_path)


def _plot_speed_hist(
    track_props: pd.DataFrame, cal: Calibration, out_path: Path,
) -> None:
    """Histogram of per-track mean speed (path length over duration)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    if len(track_props):
        speed = (track_props["mean_speed"].to_numpy(dtype=float)
                 * cal.length_scale / cal.delta)
        ax.hist(speed, bins=30, color="tab:blue", edgecolor="black", zorder=2)
    ax.set_xlabel(_axis_label("speed", cal))
    ax.set_ylabel("# tracks")
    ax.set_title("Track mean-speed distribution")
    fig.tight_layout()
    _save(fig, out_path)


def _top_n_track_ids(tracked_df: pd.DataFrame, max_tracks: int) -> list[int]:
    """Return up to max_tracks particle IDs with the longest tracks."""
    if "particle" not in tracked_df.columns or len(tracked_df) == 0:
        return []
    counts = tracked_df.groupby("particle").size().sort_values(ascending=False)
    return counts.head(max_tracks).index.tolist()


def _plot_length_per_track(
    tracked_df: pd.DataFrame, cal: Calibration, out_path: Path, max_tracks: int,
) -> None:
    """One line per track: cumulative displacement vs elapsed time."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    ids = _top_n_track_ids(tracked_df, max_tracks)
    for tid in ids:
        grp = tracked_df[tracked_df["particle"] == tid].sort_values("frame")
        if len(grp) < 2:
            continue
        coords = grp[["y", "x"]].to_numpy()
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(steps)]) * cal.length_scale
        ax.plot(grp["frame"].to_numpy() * cal.delta, cum,
                alpha=0.5, linewidth=0.7)
    ax.set_xlabel(_axis_label("time", cal))
    ax.set_ylabel(_axis_label("cum_disp", cal))
    ax.set_title(f"Per-track cumulative displacement (top {len(ids)})")
    fig.tight_layout()
    _save(fig, out_path)


def _plot_speed_per_track(
    tracked_df: pd.DataFrame, cal: Calibration, out_path: Path, max_tracks: int,
) -> None:
    """One line per track: frame-to-frame speed vs elapsed time.

    Speed at frame f is the distance from the previous frame to frame f
    (so the first frame of every track has no value and is skipped).
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    ids = _top_n_track_ids(tracked_df, max_tracks)
    for tid in ids:
        grp = tracked_df[tracked_df["particle"] == tid].sort_values("frame")
        if len(grp) < 2:
            continue
        coords = grp[["y", "x"]].to_numpy()
        steps = (np.linalg.norm(np.diff(coords, axis=0), axis=1)
                 * cal.length_scale / cal.delta)
        ax.plot(grp["frame"].to_numpy()[1:] * cal.delta, steps,
                alpha=0.5, linewidth=0.7)
    ax.set_xlabel(_axis_label("time", cal))
    ax.set_ylabel(_axis_label("inst_speed", cal))
    ax.set_title(f"Per-track speed (top {len(ids)})")
    fig.tight_layout()
    _save(fig, out_path)


def _frame_axes(ax) -> None:
    """Box the plot in and push the grid behind the marks.

    Styled to match ``scripts/papa_analysis.py``: matplotlib's default full
    frame and black type, with the grid at alpha 0.3.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black", labelcolor="black")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _thin_ticks(ax) -> None:
    """Blank every other tick label so the larger type still fits.

    Only auto-located linear axes are thinned — a categorical bar axis and a
    log axis carry meaning in every label, so both are left alone. The
    surviving positions are pinned, otherwise ``savefig`` re-runs the locator
    and the blanked labels come back.
    """
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale != "linear":
            continue
        if not isinstance(axis.get_major_locator(), mticker.AutoLocator):
            continue
        lo, hi = sorted(axis.get_view_interval())
        pairs = [
            (loc, text.get_text())
            for loc, text in zip(axis.get_majorticklocs(), axis.get_ticklabels())
            if lo <= loc <= hi
        ]
        if len(pairs) < 4:  # already sparse — thinning would strip the scale
            continue
        axis.set_ticks([loc for loc, _ in pairs])
        axis.set_ticklabels(
            [text if i % 2 == 0 else "" for i, (_, text) in enumerate(pairs)]
        )


def _fit_titles(fig) -> None:
    """Shrink any title too wide for the canvas until it fits.

    A handful of titles carry a calibration string or a normalisation
    constant, and at FONT_TITLE those run off the edge of a fixed-width
    figure — something tight_layout does not correct, since it fits the axes
    and not the text above them. Only the titles that overrun lose a point;
    every other one keeps FONT_TITLE.
    """
    renderer = fig.canvas.get_renderer()
    limit = fig.bbox.width
    for ax in fig.axes:
        title = ax.title
        if not title.get_text().strip():
            continue
        size = title.get_fontsize()
        while size > FONT_ANNOT_SMALL:
            box = title.get_window_extent(renderer=renderer)
            if box.x0 >= 0 and box.x1 <= limit:
                break
            size -= 1
            title.set_fontsize(size)


def _save(fig, out_path: Path) -> None:
    """Thin the tick labels of every panel, then write the figure out."""
    fig.canvas.draw()  # settles the limits, so the tick positions are final
    for ax in fig.axes:
        _thin_ticks(ax)
    _fit_titles(fig)
    fig.savefig(out_path, dpi=FIG_DPI, facecolor="white")
    plt.close(fig)


def _plot_msd_combined(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    cal: Calibration,
    out_path: Path,
) -> None:
    """Log-log MSD: every track faint and colored by class, ensemble mean bold.

    The ensemble mean carries its own ``4·D·t^α + 4·σ²`` fit, annotated with
    the recovered D and α.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)

    class_of = dict(zip(fits["track_id"], fits["motility_class"]))
    drawn = {}
    for tid, grp in curves.groupby("track_id"):
        cls = class_of.get(tid, "unclassified")
        grp = grp.sort_values("lag")
        ax.plot(grp["lag"], grp["msd"],
                color=CLASS_COLORS[cls], alpha=0.22, linewidth=0.8, zorder=1)
        drawn[cls] = drawn.get(cls, 0) + 1

    ens = ensemble_msd(curves)
    fit_label = ""
    if len(ens) >= 2:
        # 2px surface ring so the bold mean stays legible over the thicket.
        ax.plot(ens["lag"], ens["msd"], color="black", linewidth=2.0, zorder=4,
                path_effects=[pe.withStroke(linewidth=4.0, foreground="white")],
                label=f"Ensemble mean (n={int(ens['n_tracks'].iloc[0])})")
        try:
            fit = msd_basic_fit(
                pd.Series(ens["msd"].to_numpy(), index=np.arange(1, len(ens) + 1)),
                delta=cal.delta, sigma=cal.sigma,
            )
            ax.plot(ens["lag"], fit["y_fit"], color="black", linewidth=1.6,
                    linestyle="--", zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
                    label="Fit  4·D·t$^α$ + 4σ²")
            fit_label = (
                f"D = {fit['diffusion_coefficient']:.3g} {cal.d_unit}    "
                f"α = {fit['alpha']:.3g}"
            )
        except Exception:
            fit_label = "ensemble fit did not converge"

    if len(curves):
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(f"Lag time ({cal.time_unit})", color="black")
    ax.set_ylabel(f"MSD ({cal.msd_unit})", color="black")
    ax.set_title(f"MSD — {len(fits)} tracks · {cal.describe()}", color="black")
    if fit_label:
        ax.annotate(fit_label, xy=(0.02, 0.97), xycoords="axes fraction",
                    va="top", ha="left", fontsize=FONT_ANNOT, color="black")

    handles = [
        Line2D([], [], color=CLASS_COLORS[c], linewidth=2.0,
               label=f"{c} ({drawn[c]})")
        for c in (*MOTILITY_CLASSES, "unclassified") if drawn.get(c)
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              frameon=False, fontsize=FONT_ANNOT, loc="lower right",
              labelcolor="black")
    fig.tight_layout()
    _save(fig, out_path)


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


def _stacked_shares(ax, x_labels: list[str], per_item: list[dict]) -> None:
    """Draw 100 %-stacked class bars, least→most motile bottom→top."""
    classes: list[str] = list(MOTILITY_CLASSES)
    if any("unclassified" in d for d in per_item):
        classes.append("unclassified")
    bottoms = np.zeros(len(per_item))
    for cls in classes:
        vals = np.array([
            100.0 * d.get(cls, 0) / max(sum(d.values()), 1) for d in per_item
        ])
        ax.bar(x_labels, vals, bottom=bottoms, color=CLASS_COLORS[cls],
               width=0.62, label=cls, zorder=2)
        # Direct-label any share with room for the text (relief rule: two of
        # these hues sit below 3:1 on white, so colour alone can't carry it).
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 8.0:
                ax.annotate(f"{v:.0f}%", xy=(i, b + v / 2), ha="center",
                            va="center", fontsize=FONT_ANNOT_SMALL, color="white", zorder=3)
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of tracks", color="black")
    ax.grid(axis="x", visible=False)


def _plot_motility_classes(fits: pd.DataFrame, out_path: Path) -> None:
    """Absolute class counts, each bar direct-labelled with n and its share."""
    fig, ax_bar = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax_bar)

    counts = class_counts(fits)
    order, values = list(counts.keys()), list(counts.values())
    total = sum(values)

    ax_bar.bar(order, values, color=[CLASS_COLORS[c] for c in order],
               width=0.6, zorder=2)
    for i, n in enumerate(values):
        ax_bar.annotate(f"{n}\n{100 * n / max(total, 1):.0f}%", xy=(i, n),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT_ANNOT, color="black")
    ax_bar.set_ylabel("# tracks", color="black")
    ax_bar.set_title(f"Motility classes (n={total})", color="black")
    ax_bar.grid(axis="x", visible=False)
    if max(values, default=0) > 0:
        ax_bar.set_ylim(0, max(values) * 1.18)  # headroom for the direct labels

    fig.tight_layout()
    _save(fig, out_path)


def _plot_msd_regimes(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    cal: Calibration,
    out_path: Path,
) -> None:
    """MSD regime map: normalized curves over the three shaded α wedges.

    Each track's MSD is divided by its own first-lag value and the lag axis by
    the first lag, so every curve passes through (1, 1). On log-log axes a
    track's class then depends on slope alone, which is what makes the wedge
    boundaries ``t^0.4`` and ``t^1.2`` an honest partition — over *raw* MSD
    curves they would not be, since vertical position there encodes D, not α.

    Two corrections make the drawn slope match the fitted α. The localization
    floor ``4σ²`` is subtracted, since the raw curve's slope is that of
    ``4·D·t^α + 4σ²``, not of ``t^α``. And the anchor is the *fitted* value at
    the first lag, ``4·D·t₁^α`` — not the measured first point, which for a
    constrained track is mostly localization noise once the offset is removed,
    so dividing by it would fling exactly those tracks into the wrong wedge.
    Tracks whose fit did not converge have no anchor and are omitted.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)

    offset = 4.0 * cal.sigma ** 2
    class_of = dict(zip(fits["track_id"], fits["motility_class"]))
    d_of = dict(zip(fits["track_id"], fits["D"]))
    alpha_of = dict(zip(fits["track_id"], fits["alpha"]))
    drawn, x_max, y_lo, y_hi = {}, 1.0, 1.0, 1.0
    normalized = []
    for tid, grp in curves.groupby("track_id"):
        grp = grp.sort_values("lag")
        lags = grp["lag"].to_numpy(dtype=float)
        vals = grp["msd"].to_numpy(dtype=float) - offset
        d_fit, a_fit = d_of.get(tid, np.nan), alpha_of.get(tid, np.nan)
        lag0 = float(lags[0])
        if lag0 <= 0 or not (np.isfinite(d_fit) and np.isfinite(a_fit)) or d_fit <= 0:
            continue
        anchor = 4.0 * d_fit * lag0 ** a_fit
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        xs = lags / lag0
        ys = np.where(vals > 0, vals / anchor, np.nan)
        if not np.any(np.isfinite(ys)):
            continue
        cls = class_of.get(tid, "unclassified")
        normalized.append((xs, ys, cls))
        drawn[cls] = drawn.get(cls, 0) + 1
        x_max = max(x_max, float(xs[-1]))
        y_lo = min(y_lo, float(np.nanmin(ys)))
        y_hi = max(y_hi, float(np.nanmax(ys)))

    if not normalized:
        # Degenerate axes would draw wedges over a 1→1.001 span; say so instead.
        ax.set_xlabel("Lag time / first lag  (normalized)", color="black")
        ax.set_ylabel("MSD / 4·D·t₁^α", color="black")
        ax.set_title("MSD regime map — 0 tracks", color="black")
        ax.annotate("no tracks with a converged MSD fit", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", va="center",
                    fontsize=FONT_ANNOT, color="black")
        fig.tight_layout()
        _save(fig, out_path)
        return

    # Shade the wedges first, anchored at (1, 1) where every curve starts.
    grid = np.logspace(0, np.log10(max(x_max, 1.001)), 200)
    lo_bound = grid ** ALPHA_CONSTRAINED_MAX
    hi_bound = grid ** ALPHA_DIRECTED_MIN
    # Clamp both limits to where the bulk of the curves live. Degenerate fits
    # (α large, D → 0) give a near-zero anchor and send MSD/anchor to 1e17,
    # while short lags with no signal above the localization floor sit near
    # zero — either extreme alone flattens all three wedges into a sliver.
    all_ys = np.concatenate([ys for _, ys, _ in normalized]) if normalized else None
    if all_ys is not None and np.any(np.isfinite(all_ys)):
        floor = max(float(np.nanpercentile(all_ys, 2)), y_hi * 1e-4)
        floor = min(floor, 0.6)
        ceil = max(float(np.nanpercentile(all_ys, 98)),
                   float(hi_bound[-1])) * 1.6
    else:
        floor, ceil = 0.6, max(y_hi, float(hi_bound[-1])) * 1.6
    ax.fill_between(grid, floor, lo_bound, color=CLASS_COLORS["constrained"],
                    alpha=0.16, linewidth=0, zorder=0)
    ax.fill_between(grid, lo_bound, hi_bound, color=CLASS_COLORS["diffusive"],
                    alpha=0.16, linewidth=0, zorder=0)
    ax.fill_between(grid, hi_bound, ceil, color=CLASS_COLORS["directed"],
                    alpha=0.16, linewidth=0, zorder=0)
    for bound in (lo_bound, hi_bound):
        ax.plot(grid, bound, color="gray", linewidth=1.2, linestyle="--",
                zorder=3)

    for xs, ys, cls in normalized:
        ax.plot(xs, ys, color=CLASS_COLORS[cls], alpha=0.35, linewidth=0.9,
                zorder=2)

    # Region labels ride at the right edge, inside their own wedge.
    x_lab = max(x_max, 1.001) ** 0.62
    for text, y_at in (
        (f"α > {ALPHA_DIRECTED_MIN}  directed", (x_lab ** ALPHA_DIRECTED_MIN) * 2.4),
        (f"{ALPHA_CONSTRAINED_MAX} < α < {ALPHA_DIRECTED_MIN}  diffusive",
         (x_lab ** ((ALPHA_CONSTRAINED_MAX + ALPHA_DIRECTED_MIN) / 2))),
        (f"α < {ALPHA_CONSTRAINED_MAX}  constrained",
         (x_lab ** ALPHA_CONSTRAINED_MAX) * 0.55),
    ):
        ax.annotate(text, xy=(x_lab, y_at), ha="center", va="center",
                    fontsize=FONT_ANNOT, color="black", zorder=4,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    if normalized:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, max(x_max, 1.001))
        ax.set_ylim(floor, ceil)
    ax.set_xlabel("Lag time / first lag  (normalized)", color="black")
    ax.set_ylabel(
        "(MSD − 4σ²) / 4·D·t₁^α" if offset > 0 else "MSD / 4·D·t₁^α",
        color="black",
    )
    ax.set_title(f"MSD regime map — {sum(drawn.values())} tracks", color="black")
    handles = [
        Line2D([], [], color=CLASS_COLORS[c], linewidth=2.0,
               label=f"{c} ({drawn[c]})")
        for c in (*MOTILITY_CLASSES, "unclassified") if drawn.get(c)
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=FONT_ANNOT, loc="upper left",
                  labelcolor="black")
    fig.tight_layout()
    _save(fig, out_path)


def _alpha_criterion(cls: str) -> str:
    """The α rule for a class, worded exactly as ``classify_motility`` applies it.

    Both boundaries fall to ``diffusive``: that function sends ``a < 0.4`` to
    constrained and ``a > 1.2`` to directed, so α of exactly 0.4 or 1.2 is
    diffusive — hence ``≤`` on both sides rather than strict inequalities.
    """
    if cls == "constrained":
        return f"α < {ALPHA_CONSTRAINED_MAX}"
    if cls == "directed":
        return f"α > {ALPHA_DIRECTED_MIN}"
    return f"{ALPHA_CONSTRAINED_MAX} ≤ α ≤ {ALPHA_DIRECTED_MIN}"


def _d_axis_limits(d_values) -> tuple:
    """Log-axis limits for D, clamped against degenerate fits.

    A single non-convergent track can return D ~ 1e-199. On a log axis that
    one point stretches the view over ~180 decades and flattens the entire
    real distribution into a line, so clamp to the bulk instead. Callers
    report whatever is left outside rather than hiding it silently.
    """
    d = np.asarray(d_values, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if not d.size:
        return 1e-3, 1.0
    lo = float(np.percentile(d, 0.5))
    hi = float(np.percentile(d, 99.9))
    if lo <= 0 or not np.isfinite(lo):
        lo = float(d.min())
    return lo / 3.0, hi * 3.0


def _alpha_axis_limits(alpha_values) -> tuple:
    """x-limits for an α axis, clamped against runaway fits.

    A non-convergent fit can return α of 15 where 99.5 % of tracks sit below
    4, which leaves most of the axis blank. Clamp to the bulk, but always keep
    room past the directed cut so that band stays visible even when no track
    reaches it.
    """
    a = np.asarray(alpha_values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return -0.5, ALPHA_DIRECTED_MIN + 1.0
    lo = float(np.percentile(a, 0.5))
    hi = max(float(np.percentile(a, 99.5)), ALPHA_DIRECTED_MIN * 1.4)
    pad = 0.05 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _plot_d_vs_alpha(fits: pd.DataFrame, cal: Calibration, out_path: Path) -> None:
    """Joint distribution of D and α, with marginals and the class cuts."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Explicit margins rather than tight_layout: the shared-axis marginals are
    # not tight_layout-compatible, and a fixed layout keeps this figure the same
    # size as every other one.
    # top is lower than the plain 0.92 this figure used to carry: the corner
    # legend's three two-line entries grow up out of their cell, and at the
    # print type sizes the first entry ran off the top of the canvas.
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05,
                          left=0.17, right=0.97, top=0.86, bottom=0.15)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    # gs[0, 1] is the empty corner of a joint plot — it carries the legend.
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    for a in (ax, ax_top, ax_right):
        _frame_axes(a)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    valid = fits[np.isfinite(fits["alpha"]) & (fits["D"] > 0)]
    for cls in MOTILITY_CLASSES:
        sel = valid[valid["motility_class"] == cls]
        if not len(sel):
            continue
        ax.scatter(sel["alpha"], sel["D"], s=26, color=CLASS_COLORS[cls],
                   edgecolor="white", linewidth=0.6, alpha=0.85, zorder=2,
                   label=f"{cls}\nn = {len(sel)}")
    if len(valid):
        ax.set_yscale("log")
        d_lo, d_hi = _d_axis_limits(valid["D"])
        a_lo, a_hi = _alpha_axis_limits(valid["alpha"])
        ax.set_ylim(d_lo, d_hi)
        ax.set_xlim(a_lo, a_hi)
        # Bin the marginal over the visible range, not the raw data range —
        # otherwise the outliers spend most of the bins off-screen.
        ax_top.hist(valid["alpha"], bins=np.linspace(a_lo, a_hi, 40),
                    color="tab:blue", zorder=2)
        ax_right.hist(valid["D"], bins=np.logspace(
            np.log10(d_lo), np.log10(d_hi), 40),
            orientation="horizontal", color="tab:blue", zorder=2)
        n_off = int((
            (valid["D"] < d_lo) | (valid["D"] > d_hi)
            | (valid["alpha"] < a_lo) | (valid["alpha"] > a_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=FONT_ANNOT_SMALL,
                        color="#52514e")

    for cut in (ALPHA_CONSTRAINED_MAX, ALPHA_DIRECTED_MIN):
        for a in (ax, ax_top):
            # 75 % black rather than full black: these rules sit right next
            # to the legend and read as a competing mark at full strength.
            a.axvline(cut, color="#404040", linestyle="--", linewidth=1.2,
                      zorder=3)
    # One label per α band rather than one per rule: the bands are what the
    # reader needs named, and three labels spread across the axis cannot
    # collide the way two labels beside adjacent rules did. All sit on the
    # bottom axis, running vertically so a narrow band still fits.
    if len(valid):
        x_lo, x_hi = ax.get_xlim()
        span = x_hi - x_lo
        for cls, lo, hi in (
            ("constrained", x_lo, ALPHA_CONSTRAINED_MAX),
            ("diffusive", ALPHA_CONSTRAINED_MAX, ALPHA_DIRECTED_MIN),
            ("directed", ALPHA_DIRECTED_MIN, x_hi),
        ):
            lo, hi = max(lo, x_lo), min(hi, x_hi)
            if hi <= lo or (hi - lo) / span < 0.07:
                continue
            # Anchored just inside the band's left edge, not centred: centred
            # labels drift into whatever the cloud is doing mid-band, while the
            # left edge is a fixed landmark (the axis, or the rule beside it).
            ax.annotate(_alpha_criterion(cls), xy=(lo, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=FONT_ANNOT_SMALL, color="black",
                        rotation=90, va="bottom", ha="left", zorder=5,
                        # A halo regardless: these sit on the bottom axis,
                        # which for the constrained band is inside its cluster.
                        path_effects=[pe.withStroke(linewidth=5.0,
                                                    foreground="white")])

    ax.set_xlabel("Scaling exponent α", color="black")
    ax.set_ylabel(f"D ({cal.d_unit})", color="black")
    ax_top.set_ylabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_right.set_xlabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_top.set_title(f"D vs α — {len(valid)} tracks", color="black")
    if len(valid):
        # The corner cell, not over the data: D rises with α on a calibrated
        # run and falls on an uncalibrated one, so no in-axes corner is
        # reliably empty. The per-class count rides here on a second line
        # rather than inside the plot: placed by α band it would follow the
        # data spread, and on a run where α runs out to 8 with everything
        # packed below 2 the band midpoints land in empty space. The corner is
        # fixed. Bottom-anchored so the taller block grows up into the blank
        # band beside the title instead of down onto the D histogram.
        handles, labels = ax.get_legend_handles_labels()
        ax_corner.legend(handles, labels, loc="lower center", frameon=False,
                         fontsize=FONT_ANNOT_SMALL, labelcolor="black",
                         handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
    _save(fig, out_path)


def _plot_d_vs_alpha_summary(
    per_file: list, cal: Calibration, out_path: Path, *,
    limits: tuple | None = None,
) -> None:
    """Run-level D vs α: every track, plus one marker per experiment.

    ``per_file`` is a list of ``(stem, fits)``. The experiment, not the track,
    is the unit of replication, so pooling tracks alone would weight the
    figure by however many tracks each stack happened to yield — one stack
    contributing 64 %% of the points is not a 64 %% stronger result. Each
    experiment therefore also gets a single marker at its median (α, D), and
    the title reports n as the number of experiments. This is the SuperPlot
    convention of Lord et al., J Cell Biol 219(6):e202001064 (2020).
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05,
                          left=0.17, right=0.97, top=0.87, bottom=0.15)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    for a in (ax, ax_top, ax_right):
        _frame_axes(a)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    frames = [f for _, f in per_file if len(f)]
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    valid = (pooled[np.isfinite(pooled["alpha"]) & (pooled["D"] > 0)]
             if len(pooled) else pooled)

    for cls in MOTILITY_CLASSES:
        sel = valid[valid["motility_class"] == cls] if len(valid) else valid
        if not len(sel):
            continue
        ax.scatter(sel["alpha"], sel["D"], s=14, color=CLASS_COLORS[cls],
                   edgecolor="none", alpha=0.45, zorder=2,
                   label=f"{cls}\nn = {len(sel)}")

    # One marker per experiment at its median — the replicate-level summary.
    n_exp = 0
    for stem, f in per_file:
        good = f[np.isfinite(f["alpha"]) & (f["D"] > 0)] if len(f) else f
        if not len(good):
            continue
        n_exp += 1
        ax.scatter([good["alpha"].median()], [good["D"].median()], s=35,
                   marker="D", color="black", edgecolor="white", linewidth=0.8,
                   alpha=0.75, zorder=6)

    if len(valid):
        ax.set_yscale("log")
        if limits is not None:
            a_lo, a_hi, d_lo, d_hi = limits
        else:
            d_lo, d_hi = _d_axis_limits(valid["D"])
            a_lo, a_hi = _alpha_axis_limits(valid["alpha"])
        ax.set_ylim(d_lo, d_hi)
        ax.set_xlim(a_lo, a_hi)
        # Bin the marginal over the visible range, not the raw data range —
        # otherwise the outliers spend most of the bins off-screen.
        ax_top.hist(valid["alpha"], bins=np.linspace(a_lo, a_hi, 40),
                    color="tab:blue", zorder=2)
        ax_right.hist(valid["D"], bins=np.logspace(
            np.log10(d_lo), np.log10(d_hi), 40),
            orientation="horizontal", color="tab:blue", zorder=2)
        n_off = int((
            (valid["D"] < d_lo) | (valid["D"] > d_hi)
            | (valid["alpha"] < a_lo) | (valid["alpha"] > a_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=FONT_ANNOT_SMALL,
                        color="#52514e")

    for cut in (ALPHA_CONSTRAINED_MAX, ALPHA_DIRECTED_MIN):
        for a in (ax, ax_top):
            a.axvline(cut, color="#404040", linestyle="--", linewidth=1.2,
                      zorder=3)
    if len(valid):
        x_lo, x_hi = ax.get_xlim()
        span = x_hi - x_lo
        for cls, lo, hi in (
            ("constrained", x_lo, ALPHA_CONSTRAINED_MAX),
            ("diffusive", ALPHA_CONSTRAINED_MAX, ALPHA_DIRECTED_MIN),
            ("directed", ALPHA_DIRECTED_MIN, x_hi),
        ):
            lo, hi = max(lo, x_lo), min(hi, x_hi)
            if hi <= lo or (hi - lo) / span < 0.07:
                continue
            ax.annotate(_alpha_criterion(cls), xy=(lo, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=FONT_ANNOT_SMALL, color="black",
                        rotation=90, va="bottom", ha="left", zorder=7,
                        path_effects=[pe.withStroke(linewidth=5.0,
                                                    foreground="white")])

    ax.set_xlabel("Scaling exponent α", color="black")
    ax.set_ylabel(f"D ({cal.d_unit})", color="black")
    ax_top.set_ylabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_right.set_xlabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_top.set_title(
        f"D vs α — n = {n_exp} experiments  ({len(valid):,} tracks)"
        f"\n◆ = per-experiment median",
        color="black", fontsize=FONT_TITLE,
    )
    if len(valid):
        handles, labels = ax.get_legend_handles_labels()
        ax_corner.legend(handles, labels, loc="lower center", frameon=False,
                         fontsize=FONT_ANNOT_SMALL, labelcolor="black",
                         handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
    _save(fig, out_path)


def _pct_limits(values, lo_pct=0.5, hi_pct=99.5, pad=0.05) -> tuple:
    """Percentile limits with padding, so a few outliers cannot own the axis."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    m = pad * (hi - lo)
    return lo - m, hi + m


def _experiment_values(per_file, column, scale):
    """Per-experiment arrays of one track property, in working units.

    Only tracks that entered the MSD analysis are kept, so these summaries
    describe the same population as the per-file figures rather than the full
    linked set (which is dominated by 2-4 frame fragments).
    """
    out = []
    for stem, df in per_file:
        if column not in df.columns or not len(df):
            continue
        keep = df[df["motility_class"] != "unclassified"] if (
            "motility_class" in df.columns) else df
        v = pd.to_numeric(keep[column], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)] * scale
        if v.size:
            out.append((stem, v))
    return out


def _plot_metric_summary(
    per_file, column: str, xlabel: str, scale: float, out_path: Path, *,
    limits: tuple | None = None, y_max: float | None = None,
    density: bool = False, color: str = "tab:blue", title: str | None = None,
) -> None:
    """Run-level distribution of one track property, with replicate medians.

    The histogram is every track pooled; the diamonds are one per experiment.
    The spread of the diamonds — not the width of the histogram — is what
    says whether the experiments agree, because pooling weights each stack by
    however many tracks it happened to yield.

    ``limits`` and ``y_max`` override the axis range this figure would pick
    for itself. ``--compare`` passes the range shared by every group, so the
    per-group figures can be read against each other rather than each one
    silently rescaling to its own data. ``density`` switches the y axis from
    counts to the fraction of that group's tracks, which is the only way a
    group of 12,000 tracks and one of 52,000 can share a y axis.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)

    ylabel = "fraction of tracks" if density else "# tracks"
    groups = _experiment_values(per_file, column, scale)
    if not groups:
        ax.annotate("no tracks", xy=(0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=FONT_ANNOT,
                    color="#52514e")
        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel(ylabel, color="black")
        if limits is not None:
            ax.set_xlim(*limits)
        fig.tight_layout()
        _save(fig, out_path)
        return

    pooled = np.concatenate([v for _, v in groups])
    lo, hi = limits if limits is not None else _pct_limits(pooled)
    weights = (np.full(pooled.size, 1.0 / pooled.size) if density else None)
    ax.hist(pooled, bins=np.linspace(lo, hi, GROUP_BINS), color=color,
            weights=weights, zorder=2)
    ax.set_xlim(lo, hi)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    for _, v in groups:
        ax.scatter([np.median(v)], [0.95], transform=ax.get_xaxis_transform(),
                   s=35, marker="D", color="black", edgecolor="white",
                   linewidth=0.8, alpha=0.75, zorder=5, clip_on=False)

    n_off = int(((pooled < lo) | (pooled > hi)).sum())
    if n_off:
        ax.annotate(f"{n_off} track(s) outside axis range",
                    xy=(0.99, 0.94), xycoords="axes fraction", ha="right",
                    va="top", fontsize=FONT_ANNOT_SMALL, color="#52514e")
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel(ylabel, color="black")
    head = title + "  " if title else ""
    ax.set_title(
        f"{head}n = {len(groups)} experiments  ({len(pooled):,} tracks)"
        f"\n◆ = per-experiment median",
        color="black", fontsize=FONT_TITLE,
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_path)


def _directed_metrics(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
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
    d["_disp"] = net * cal.length_scale
    d["_vel"] = (net / steps) * cal.length_scale / cal.delta
    return d


def _directed_units(cal: Calibration) -> tuple:
    """(run-length, velocity) axis labels for the current calibration."""
    if cal.is_physical:
        return "Run length (µm)", "Velocity (µm/s)"
    return "Run length (px)", "Velocity (px/frame)"


def _plot_directed_hist(
    props: pd.DataFrame, cal: Calibration, column: str, xlabel: str,
    out_path: Path,
) -> None:
    """Per-file distribution of one directed-track property."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)
    d = _directed_metrics(props, cal)
    if len(d):
        v = pd.to_numeric(d[column], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
    else:
        v = np.array([])
    if v.size:
        lo, hi = _pct_limits(v)
        ax.hist(v, bins=np.linspace(lo, hi, 40),
                color=CLASS_COLORS["directed"], zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_title(
            f"Directed tracks — n = {v.size}  (median {np.median(v):.3g})",
            color="black",
        )
    else:
        ax.annotate("no directed tracks", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", va="center",
                    fontsize=FONT_ANNOT, color="#52514e")
        ax.set_title("Directed tracks — n = 0", color="black")
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel("# tracks", color="black")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_path)


def _plot_directed_joint(
    props: pd.DataFrame, cal: Calibration, out_path: Path,
) -> None:
    """Per-file run length against velocity, directed tracks only.

    Carries marginal histograms like the other joint figures, so a reader can
    see each distribution as well as the relationship between them.
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05,
                          left=0.17, right=0.97, top=0.90, bottom=0.15)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    for a in (ax, ax_top, ax_right):
        _frame_axes(a)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    d = _directed_metrics(props, cal)
    xl, yl = _directed_units(cal)
    if len(d):
        ax.scatter(d["_disp"], d["_vel"], s=26,
                   color=CLASS_COLORS["directed"], edgecolor="white",
                   linewidth=0.6, alpha=0.85, zorder=2)
        x_lo, x_hi = _pct_limits(d["_disp"])
        y_lo, y_hi = _pct_limits(d["_vel"])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax_top.hist(d["_disp"], bins=np.linspace(x_lo, x_hi, 40),
                    color="tab:blue", zorder=2)
        ax_right.hist(d["_vel"], bins=np.linspace(y_lo, y_hi, 40),
                      orientation="horizontal", color="tab:blue", zorder=2)
        n_off = int((
            (d["_disp"] < x_lo) | (d["_disp"] > x_hi)
            | (d["_vel"] < y_lo) | (d["_vel"] > y_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=FONT_ANNOT_SMALL, color="#52514e")
        ax_top.set_title(f"Directed tracks — n = {len(d)}", color="black")
    else:
        ax.annotate("no directed tracks", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", va="center",
                    fontsize=FONT_ANNOT, color="#52514e")
        ax_top.set_title("Directed tracks — n = 0", color="black")

    ax.set_xlabel(xl, color="black")
    ax.set_ylabel(yl, color="black")
    ax_top.set_ylabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_right.set_xlabel("# tracks", color="black", fontsize=FONT_ANNOT)
    _save(fig, out_path)


def _plot_joint_summary(
    per_file, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str,
    out_path: Path, color_by_class: bool = True, *,
    limits: tuple | None = None,
) -> None:
    """Run-level joint distribution of two track properties.

    ``per_file`` frames must already carry ``xcol``/``ycol`` in working units.
    One marker per experiment sits at its median, since the experiment and not
    the track is the unit of replication.

    ``limits`` is an ``(x_lo, x_hi, y_lo, y_hi)`` override, used by
    ``--compare`` so every group's figure is drawn on the same axes.
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05,
                          left=0.17, right=0.97, top=0.87, bottom=0.15)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    for a in (ax, ax_top, ax_right):
        _frame_axes(a)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    frames = [(stem, f) for stem, f in per_file if len(f)]
    if frames:
        pooled = pd.concat([f for _, f in frames], ignore_index=True)
    else:
        pooled = pd.DataFrame(columns=[xcol, ycol, "motility_class"])

    if color_by_class and "motility_class" in pooled.columns and len(pooled):
        for cls in MOTILITY_CLASSES:
            sel = pooled[pooled["motility_class"] == cls]
            if not len(sel):
                continue
            ax.scatter(sel[xcol], sel[ycol], s=14, color=CLASS_COLORS[cls],
                       edgecolor="none", alpha=0.45, zorder=2,
                       label=f"{cls}\nn = {len(sel)}")
    elif len(pooled):
        ax.scatter(pooled[xcol], pooled[ycol], s=14,
                   color=CLASS_COLORS["directed"], edgecolor="none",
                   alpha=0.45, zorder=2)
    for _, f in frames:
        ax.scatter([f[xcol].median()], [f[ycol].median()], s=35, marker="D",
                   color="black", edgecolor="white", linewidth=0.8, alpha=0.75,
                   zorder=6)

    if len(pooled):
        if limits is not None:
            x_lo, x_hi, y_lo, y_hi = limits
        else:
            x_lo, x_hi = _pct_limits(pooled[xcol])
            y_lo, y_hi = _pct_limits(pooled[ycol])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax_top.hist(pooled[xcol], bins=np.linspace(x_lo, x_hi, 40),
                    color="tab:blue", zorder=2)
        ax_right.hist(pooled[ycol], bins=np.linspace(y_lo, y_hi, 40),
                      orientation="horizontal", color="tab:blue", zorder=2)
        n_off = int((
            (pooled[xcol] < x_lo) | (pooled[xcol] > x_hi)
            | (pooled[ycol] < y_lo) | (pooled[ycol] > y_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=FONT_ANNOT_SMALL, color="#52514e")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_corner.legend(handles, labels, loc="lower center",
                             frameon=False, fontsize=FONT_ANNOT_SMALL,
                             labelcolor="black", handletextpad=0.4,
                             labelspacing=0.3, borderpad=0.1)
    elif limits is not None:
        ax.set_xlim(*limits[:2])
        ax.set_ylim(*limits[2:])

    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel(ylabel, color="black")
    ax_top.set_ylabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_right.set_xlabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_top.set_title(
        f"{title} — n = {len(frames)} experiments  ({len(pooled):,} tracks)"
        f"\n◆ = per-experiment median",
        color="black", fontsize=FONT_TITLE,
    )
    _save(fig, out_path)


def _length_speed_metrics(per_file, cal: Calibration) -> list:
    """``per_file`` with ``_len``/``_spd`` columns in working units.

    Unclassified tracks are dropped, so this describes the same population as
    every other summary rather than the full linked set.
    """
    spd_scale = cal.length_scale / cal.delta
    frames = []
    for stem, df in per_file:
        if not len(df) or not {"mean_speed", "_dur"} <= set(df.columns):
            continue
        keep = df[df["motility_class"] != "unclassified"].copy()
        if not len(keep):
            continue
        keep["_len"] = keep["_dur"].astype(float)
        keep["_spd"] = keep["mean_speed"].astype(float) * spd_scale
        frames.append((stem, keep))
    return frames


def _length_speed_units(cal: Calibration) -> tuple:
    """(track-length, mean-speed) axis labels for the current calibration."""
    return _axis_label("duration", cal), _axis_label("speed", cal)


def _plot_length_vs_speed_summary(
    per_file, cal: Calibration, out_path: Path, *, limits: tuple | None = None,
) -> None:
    """Run-level track length against mean speed, with replicate medians."""
    xlabel, ylabel = _length_speed_units(cal)
    _plot_joint_summary(
        _length_speed_metrics(per_file, cal), "_len", "_spd", xlabel, ylabel,
        "Length vs speed", out_path, limits=limits)


# ── MSD methods figures (after IJMS 25(16), 8660 — Fig. 1) ──────────


def _representative_track(fits: pd.DataFrame, cls: str) -> int | None:
    """Track whose α is the median of its class — robust to outliers."""
    sel = fits[(fits["motility_class"] == cls) & np.isfinite(fits["alpha"])]
    if not len(sel):
        return None
    order = sel["alpha"].sort_values()
    return int(sel.loc[order.index[len(order) // 2], "track_id"])


def _track_xy(tracked_df: pd.DataFrame, tid: int, cal: Calibration) -> np.ndarray:
    grp = tracked_df[tracked_df["particle"] == tid].sort_values("frame")
    return grp[["x", "y"]].to_numpy(dtype=float) * cal.length_scale


def _fit_curve(fits: pd.DataFrame, tid: int, lags: np.ndarray,
               cal: Calibration) -> np.ndarray | None:
    """``4·D·t^α + 4σ²`` evaluated at ``lags`` for one track."""
    row = fits[fits["track_id"] == tid]
    if not len(row):
        return None
    d_fit = float(row["D"].iloc[0])
    a_fit = float(row["alpha"].iloc[0])
    if not (np.isfinite(d_fit) and np.isfinite(a_fit)):
        return None
    return 4.0 * d_fit * lags ** a_fit + 4.0 * cal.sigma ** 2


def _draw_scale_bar(ax, xy: np.ndarray, cal: Calibration) -> None:
    """Round scale bar sized to roughly a quarter of the trajectory extent."""
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    if span <= 0:
        return
    nice = 10 ** np.floor(np.log10(span / 4.0))
    for mult in (5.0, 2.0, 1.0):
        if nice * mult <= span / 2.0:
            nice *= mult
            break
    x0 = xy[:, 0].min()
    y0 = xy[:, 1].min() - 0.08 * span
    ax.plot([x0, x0 + nice], [y0, y0], color="black", linewidth=2.5,
            solid_capstyle="butt")
    ax.annotate(f"{nice:g} {cal.length_unit}", xy=(x0 + nice / 2, y0),
                xytext=(0, -3), textcoords="offset points", ha="center",
                va="top", fontsize=FONT_ANNOT_SMALL, color="black")


def _plot_msd_construction(
    tracked_df: pd.DataFrame,
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    cal: Calibration,
    out_path: Path,
) -> None:
    """Panels (a) and (b): how an MSD curve is built from one trajectory.

    (a) the trajectory with the displacement chords for Δt = τ…5τ;
    (b) the resulting MSD ± SEM, first five lags colour-matched to (a).
    """
    tid = (_representative_track(fits, "diffusive")
           or _representative_track(fits, "constrained")
           or _representative_track(fits, "directed"))
    fig, (ax_traj, ax_msd) = plt.subplots(1, 2, figsize=(2 * FIG_W_PANEL, FIG_H))
    _frame_axes(ax_traj)
    _frame_axes(ax_msd)

    if tid is None:
        for a, t in ((ax_traj, "Lag construction"), (ax_msd, "MSD ± SEM")):
            a.set_title(t, color="black")
            a.annotate("no tracks with a converged MSD fit", xy=(0.5, 0.5),
                       xycoords="axes fraction", ha="center", va="center",
                       fontsize=FONT_ANNOT, color="black")
        fig.tight_layout()
        _save(fig, out_path)
        return

    # ── (a) trajectory + one chord per lag ───────────────────────
    xy = _track_xy(tracked_df, tid, cal)
    ax_traj.plot(xy[:, 0], xy[:, 1], color="lightgray", linewidth=1.2, zorder=1)
    ax_traj.scatter(xy[:, 0], xy[:, 1], s=14, color="black", zorder=2)
    n_lags = min(5, len(xy) - 1)
    for k in range(1, n_lags + 1):
        ax_traj.annotate(
            "", xy=tuple(xy[k]), xytext=tuple(xy[0]),
            arrowprops=dict(arrowstyle="->", color=LAG_RAMP[k - 1],
                            linewidth=2.0, shrinkA=0, shrinkB=0),
            zorder=3,
        )
    handles = [Line2D([], [], color=LAG_RAMP[k - 1], linewidth=2.0,
                      label=f"Δt = {'' if k == 1 else k}τ")
               for k in range(1, n_lags + 1)]
    ax_traj.legend(handles=handles, frameon=False, fontsize=FONT_ANNOT_SMALL,
                   loc="best", labelcolor="black", title="steps from start")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_xlabel(f"x ({cal.length_unit})", color="black")
    ax_traj.set_ylabel(f"y ({cal.length_unit})", color="black")
    ax_traj.set_title(f"Displacements at Δt = τ…{n_lags}τ  (track {tid})",
                      color="black")

    # ── (b) the MSD those displacements average to ───────────────
    grp = curves[curves["track_id"] == tid].sort_values("lag")
    lags = grp["lag"].to_numpy(dtype=float)
    vals = grp["msd"].to_numpy(dtype=float)
    sems = grp["sem"].to_numpy(dtype=float)
    ax_msd.errorbar(lags, vals, yerr=np.nan_to_num(sems), fmt="o", markersize=4,
                    color="black", ecolor="gray", elinewidth=1.0,
                    capsize=2, zorder=2, label="MSD ± SEM")
    for k in range(1, min(n_lags, len(lags)) + 1):
        ax_msd.scatter(lags[k - 1], vals[k - 1], s=54, color=LAG_RAMP[k - 1],
                       edgecolor="white", linewidth=0.8, zorder=3)
    fit_y = _fit_curve(fits, tid, lags, cal)
    if fit_y is not None:
        a_fit = float(fits.loc[fits["track_id"] == tid, "alpha"].iloc[0])
        ax_msd.plot(lags, fit_y, color="black", linestyle="--", linewidth=1.6,
                    zorder=4, label=f"fit  4·D·t$^α$ + 4σ²  (α={a_fit:.2f})")
    ax_msd.set_xlabel(f"Lag time τ ({cal.time_unit})", color="black")
    ax_msd.set_ylabel(f"MSD ({cal.msd_unit})", color="black")
    ax_msd.set_title("Averaging equal-lag displacements", color="black")
    ax_msd.legend(frameon=False, fontsize=FONT_ANNOT_SMALL, loc="upper left",
                  labelcolor="black")

    fig.tight_layout()
    _save(fig, out_path)


def _plot_msd_archetypes(
    tracked_df: pd.DataFrame,
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    cal: Calibration,
    out_path: Path,
) -> None:
    """Panels (c)–(e): the characteristic MSD shape of each class.

    One representative track per class — MSD ± SEM with its fit, a dashed
    pure-Brownian ``4·D·τ`` reference so the departure from linear is visible,
    and the trajectory itself as an inset with a scale bar.
    """
    panels = [
        ("Brownian", "diffusive"),
        ("Drifted", "directed"),
        ("Confined", "constrained"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_ARCHETYPES, FIG_H))
    for ax, (title, cls) in zip(axes, panels):
        _frame_axes(ax)
        ax.set_xlabel(f"Lag time τ ({cal.time_unit})", color="black")
        ax.set_ylabel(f"MSD ({cal.msd_unit})", color="black")
        tid = _representative_track(fits, cls)
        if tid is None:
            ax.set_title(f"{title} — {cls}", color="black")
            ax.annotate(f"no {cls} tracks", xy=(0.5, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        fontsize=FONT_ANNOT, color="black")
            continue

        grp = curves[curves["track_id"] == tid].sort_values("lag")
        lags = grp["lag"].to_numpy(dtype=float)
        vals = grp["msd"].to_numpy(dtype=float)
        sems = np.nan_to_num(grp["sem"].to_numpy(dtype=float))
        ax.errorbar(lags, vals, yerr=sems, fmt="o", markersize=3.5,
                    color=CLASS_COLORS[cls], ecolor="gray", elinewidth=1.0,
                    capsize=2, zorder=3)
        fit_y = _fit_curve(fits, tid, lags, cal)
        a_fit = float(fits.loc[fits["track_id"] == tid, "alpha"].iloc[0])
        if fit_y is not None:
            ax.plot(lags, fit_y, color="black", linewidth=1.6, zorder=4)
        # Headroom for the inset, scaled to the data — not to the Brownian
        # reference, which for a plateauing track runs far above every point.
        y_top = float(np.nanmax(vals + sems)) * 1.5 or 1.0
        # Pure-Brownian reference through the first point: MSD ∝ τ.
        if len(lags) and lags[0] > 0 and vals[0] > 0:
            ax.plot(lags, vals[0] * lags / lags[0], color="gray",
                    linestyle="--", linewidth=1.2, zorder=2,
                    label="Brownian  4·D·τ")
            ax.legend(frameon=False, fontsize=FONT_ANNOT_SMALL, loc="lower right",
                      labelcolor="black")
        ax.set_ylim(0, y_top)
        ax.set_title(f"{title} — {cls} (track {tid}, α={a_fit:.2f})",
                     color="black", fontsize=FONT_TITLE - 1)

        # Trajectory inset, upper-left — reliably clear of a rising MSD curve.
        inset = ax.inset_axes((0.07, 0.56, 0.36, 0.40))
        xy = _track_xy(tracked_df, tid, cal)
        inset.plot(xy[:, 0], xy[:, 1], color=CLASS_COLORS[cls], linewidth=0.9)
        _draw_scale_bar(inset, xy, cal)
        inset.set_aspect("equal", adjustable="datalim")
        inset.margins(x=0.12, y=0.22)  # room for the scale bar and its label
        inset.set_xticks([])
        inset.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            inset.spines[side].set_color("black")
        inset.set_facecolor("#f0f0f0")

    # Explicit margins, not tight_layout: tight_layout sizes each panel to its
    # own tick-label width, which leaves the three panels slightly unequal.
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.16,
                        wspace=0.22)
    _save(fig, out_path)


def _plot_msd_shape_comparison(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    cal: Calibration,
    out_path: Path,
) -> None:
    """Panel (f): the three class-mean MSD *shapes*, on linear axes.

    Both axes are normalized to the common maximum lag, so every curve runs
    from the origin to (1, 1) and only its curvature differs — concave for
    subdiffusive, straight for Brownian, convex for superdiffusive. Without
    that, a directed population whose MSD is orders of magnitude larger than a
    confined one flattens the other curves onto the axis and the comparison
    the panel exists to make is lost. Absolute magnitudes live in
    ``msd_combined.png``; this panel is about shape alone.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)

    class_of = dict(zip(fits["track_id"], fits["motility_class"]))
    labelled = curves.assign(motility_class=curves["track_id"].map(class_of))

    per_class = {}
    for cls in MOTILITY_CLASSES:
        sub = labelled[labelled["motility_class"] == cls]
        if not len(sub):
            continue
        agg = (sub.groupby("lag")["msd"]
               .agg(mean="mean", n="size").reset_index().sort_values("lag"))
        agg = agg[agg["n"] >= MIN_TRACKS_PER_LAG]
        if len(agg) >= 2:
            per_class[cls] = agg

    # A shared x-axis needs a lag range every class actually reaches.
    common_max = min((float(a["lag"].max()) for a in per_class.values()),
                     default=0.0)
    drew = False
    for cls, agg in per_class.items():
        agg = agg[agg["lag"] <= common_max]
        if len(agg) < 2:
            continue
        norm = float(agg["mean"].iloc[-1])
        if norm <= 0:
            continue
        alphas = fits[fits["motility_class"] == cls]["alpha"]
        xs = (agg["lag"] / common_max).to_numpy(dtype=float)
        ys = (agg["mean"] / norm).to_numpy(dtype=float)
        ax.plot(xs, ys, color=CLASS_COLORS[cls], linewidth=2.2, zorder=3)
        drew = True
        # Direct labels beat a legend here: three long labels won't fit on one
        # row, and the curves are well separated at mid-range.
        i = int(np.argmin(np.abs(xs - 0.55)))
        ax.annotate(
            f"{cls}  (n={int((fits['motility_class'] == cls).sum())}, "
            f"median α={alphas.median():.2f})",
            xy=(xs[i], ys[i]), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=FONT_ANNOT, color="black", zorder=5,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )

    # The normalization belongs on the figure, but at this type size it only
    # fits in the title — an axis label long enough to carry it runs off both
    # ends of a column-width figure.
    ax.set_xlabel("Lag time τ / τ$_{max}$", color="black")
    ax.set_ylabel("Mean MSD, normalized to τ$_{max}$", color="black")
    ax.set_title(
        f"Class-mean MSD shapes — shape only, normalized at "
        f"τ$_{{max}}$ = {common_max:.3g} {cal.time_unit}",
        color="black",
    )
    if not drew:
        ax.annotate("no class had enough tracks to average", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", va="center",
                    fontsize=FONT_ANNOT, color="black")
    fig.tight_layout()
    _save(fig, out_path)


def _legend_below_labels(fig, ax, leg) -> None:
    """Raise the axes so the rotated x labels and ``leg`` both fit under it.

    ``tight_layout`` does not account for a legend anchored outside the axes,
    so any fixed offset collides as soon as the labels get long — a nested
    experiment path ("ctrl/cell01") is far taller at 30 degrees than a bare
    stem. Measure both instead of guessing, and leave the figure height alone
    so the PNG still drops into a manuscript layout unscaled.
    """
    fig.canvas.draw()  # settles the text extents
    r = fig.canvas.get_renderer()
    fig_h = fig.get_window_extent(r).height
    pad = 6 * fig.dpi / 72.0 / fig_h  # 6 pt of air, as a figure fraction
    labels_h = max(
        (t.get_window_extent(r).height for t in ax.get_xticklabels()),
        default=0.0,
    ) / fig_h
    bottom = leg.get_window_extent(r).height / fig_h + labels_h + 2 * pad
    pos = ax.get_position()
    if bottom < pos.y1:
        ax.set_position([pos.x0, bottom, pos.width, pos.y1 - bottom])


def _plot_run_summary(rows: list[dict], out_path: Path) -> None:
    """One 100 %-stacked class bar per input file, for cross-condition reading."""
    usable = [r for r in rows if sum(r["counts"].values()) > 0]
    if not usable:
        return
    # Wider per bar than the bars need: at this type size it is the rotated
    # file stems along the x-axis that decide whether the figure is readable.
    width = max(7.0, 1.15 * len(usable) + 3.0)
    fig, ax = plt.subplots(figsize=(width, FIG_H))
    _frame_axes(ax)
    labels = [r["stem"] for r in usable]
    _stacked_shares(ax, labels, [r["counts"] for r in usable])
    # pad clears the per-bar n= annotations sitting on the 100 % line.
    ax.set_title("Motility class share per file", color="black", pad=20)
    for i, r in enumerate(usable):
        ax.annotate(f"n={sum(r['counts'].values())}", xy=(i, 100),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=FONT_ANNOT_SMALL, color="black")
    # Anchored to the figure's bottom edge, not the axes': how far the axes
    # sit above it depends on the label height, which _legend_below_labels
    # measures once everything is laid out.
    leg = ax.legend(frameon=False, fontsize=FONT_ANNOT,
                    ncol=len(MOTILITY_CLASSES) + 1,
                    loc="lower center", bbox_to_anchor=(0.5, 0.0),
                    bbox_transform=fig.transFigure, labelcolor="black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _legend_below_labels(fig, ax, leg)
    _save(fig, out_path)


# ── per-file pipeline ───────────────────────────────────────────────


def process_file(tif_path: Path, out_dir: Path, model, args: Args) -> dict:
    """Run preprocess → segment → fit → mask → track → analyze → write.

    Returns a log row: {stem, n_spots, n_tracks, status, error}.
    """
    stem = tif_path.stem
    file_out = out_dir / stem

    if args.skip_existing and file_out.exists():
        return {"stem": stem, "n_spots": 0, "n_tracks": 0,
                "status": "SKIP", "error": "output exists"}

    file_out.mkdir(parents=True, exist_ok=True)

    stack = tifffile.imread(str(tif_path))
    if stack.ndim != 3:
        return {"stem": stem, "n_spots": 0, "n_tracks": 0,
                "status": "SKIP", "error": f"ndim={stack.ndim} (expected 3)"}

    # ── Preprocess ───────────────────────────────────────────────
    pre = stack
    if not args.no_walk:
        pre = walking_average(pre, window=args.walk_window)
    if not args.no_submean:
        # Subtract the (rounded) scalar mean of the whole stack — same as the
        # plugin's "Subtract → Mean" calculation, with dtype-aware clipping.
        stack_mean = int(round(float(np.mean(pre))))
        pre = apply_calculation(pre, "subtract", stack_mean)
    if not args.no_blur:
        pre = gaussian_blur(pre, args.gauss_sigma)

    # save preprocessed stack for debugging (optional)
    print(f"Saving preprocessed stack... {pre.shape}, {pre.dtype}")
    tifffile.imwrite(str(file_out / "preprocessed.tif"), pre, bigtiff=True)
    # ── Segment ──────────────────────────────────────────────────
    spots = _detect_all_frames(pre, model, args)
    if len(spots) == 0:
        # Empty mask, empty CSVs, no plots.
        empty_mask = np.zeros(pre.shape, dtype=np.int32)
        tifffile.imwrite(str(file_out / "mask.tif"), empty_mask, bigtiff=True)
        empty_cols_spots = ["frame", "y", "x", "amp", "x_fit", "y_fit",
                            "sigma_x", "sigma_y", "bg",
                            "area", "mean_intensity", "max_intensity", "spot_radius"]
        pd.DataFrame(columns=empty_cols_spots).to_csv(file_out / "spots.csv", index=False)
        pd.DataFrame(columns=empty_cols_spots + ["particle"]).to_csv(file_out / "tracks.csv", index=False)
        pd.DataFrame(columns=["track_id", "length", "total_displacement",
                              "net_displacement", "mean_speed",
                              "mean_intensity", "max_intensity", "intensity_std",
                              "D", "alpha", "motility_class"]
                     ).to_csv(file_out / "track_analysis.csv", index=False)
        empty_curves, empty_fits = _empty_msd_frames()
        empty_curves.to_csv(file_out / "msd_curves.csv", index=False)
        empty_fits.to_csv(file_out / "msd_fits.csv", index=False)
        return {"stem": stem, "n_spots": 0, "n_tracks": 0,
                "status": "OK", "error": ""}

    # ── Fit + mask ───────────────────────────────────────────────
    print(f"Fitting {len(spots)} spots on GPU...")
    fitted = fit_all_frames_gpu(pre, spots, win=args.fit_win, device=args.device)
    print("Building spot mask...")
    mask, props = make_gpu_spot_mask(
        pre.shape, fitted, intensity_stack=pre, max_radius=args.max_radius,
    )
    print(f"Found {len(props)} valid fitted spots (of {len(spots)} detected).")
    # spots.csv columns: rename `time` → `frame`
    spots_out = props.rename(columns={"time": "frame"})
    spots_out.to_csv(file_out / "spots.csv", index=False)
    tifffile.imwrite(str(file_out / "mask.tif"), mask, bigtiff=True)

    # ── Track ────────────────────────────────────────────────────
    tracker_in = spots_out[["frame", "y", "x"]].copy()
    tracked = track_blobs(tracker_in, search_range=args.search_range, memory=args.memory)
    # Re-attach the fit/intensity columns for downstream consumers.
    tracked = tracked.merge(
        spots_out.drop(columns=["frame", "y", "x"], errors="ignore"),
        left_index=True, right_index=True, how="left",
    )
    tracked.to_csv(file_out / "tracks.csv", index=False)

    counts = analyze_folder(file_out, tracked, pre, args)

    return {"stem": stem, "n_spots": int(len(spots_out)),
            "n_tracks": int(tracked["particle"].nunique()),
            "status": "OK", "error": "", "counts": counts}


# ── analysis (post-tracking) ────────────────────────────────────────


def analyze_folder(
    out_dir: Path,
    tracked: pd.DataFrame,
    image: np.ndarray | None,
    args: Args,
) -> dict:
    """Write the per-track, MSD and figure outputs for one already-linked stack.

    Split out of ``process_file`` so ``--analysis-only`` can re-run it over a
    folder that already holds ``tracks.csv``, at a different
    ``--min-track-length`` or calibration, without redetecting or relinking.
    ``image`` is the preprocessed stack, used only for intensity sampling; pass
    ``None`` to write the geometry columns alone. Returns the motility class
    counts.
    """
    track_props = compute_track_properties(tracked, image=image)

    # ── MSD + motility classification ────────────────────────────
    cal = Calibration(
        pixel_size_nm=args.pixel_size_nm,
        frame_interval_s=args.frame_interval_s,
        sigma_nm=args.sigma_nm,
    )
    print(f"MSD analysis — {cal.describe()}")
    msd_curves, msd_fits = compute_msd_analysis(
        tracked, cal, max_lag=args.msd_max_lag, min_length=args.min_track_length,
    )
    msd_curves.to_csv(out_dir / "msd_curves.csv", index=False)
    msd_fits.to_csv(out_dir / "msd_fits.csv", index=False)
    if len(msd_fits):
        shares = msd_fits["motility_class"].value_counts()
        print("  " + ", ".join(f"{k}={v}" for k, v in shares.items()))
    else:
        print(f"  no track reached the {max(args.min_track_length, 3)}-frame "
              f"minimum — MSD outputs are empty")

    # Carry D / alpha / class onto the per-track table too.
    track_props = track_props.merge(
        msd_fits[["track_id", "D", "alpha", "motility_class"]],
        on="track_id", how="left",
    )
    track_props["motility_class"] = track_props["motility_class"].fillna("unclassified")
    # Calibrated copies of the geometry columns, beside the pixel ones rather
    # than over them: --summary-only and --compare read this table back and do
    # their own scaling, and `length` in frames is what --min-track-length cuts.
    track_props = add_physical_columns(
        track_props, args.pixel_size_nm, args.frame_interval_s)
    track_props.to_csv(out_dir / "track_analysis.csv", index=False)

    # ── Plots (filter applies to plots only) ─────────────────────
    plot_props = track_props[track_props["length"] >= args.min_track_length]
    plot_tracked = tracked[tracked["particle"].isin(plot_props["track_id"])]
    _plot_length_hist(plot_props, cal, out_dir / "length_hist.png",
                      args.min_track_length)
    _plot_path_length_hist(plot_props, cal, out_dir / "path_length_hist.png")
    _plot_speed_hist(plot_props, cal, out_dir / "speed_hist.png")
    _plot_length_per_track(plot_tracked, cal, out_dir / "length_per_track.png",
                           args.max_tracks_plot)
    _plot_speed_per_track(plot_tracked, cal, out_dir / "speed_per_track.png",
                          args.max_tracks_plot)
    _plot_msd_combined(msd_curves, msd_fits, cal, out_dir / "msd_combined.png")
    _plot_msd_regimes(msd_curves, msd_fits, cal, out_dir / "msd_regimes.png")
    _plot_motility_classes(msd_fits, out_dir / "motility_classes.png")
    _plot_d_vs_alpha(msd_fits, cal, out_dir / "d_vs_alpha.png")
    _plot_msd_construction(tracked, msd_curves, msd_fits, cal,
                           out_dir / "msd_construction.png")
    _plot_msd_archetypes(tracked, msd_curves, msd_fits, cal,
                         out_dir / "msd_archetypes.png")
    _plot_msd_shape_comparison(msd_curves, msd_fits, cal,
                               out_dir / "msd_shape_comparison.png")

    # Directed tracks only: run length and velocity are transport quantities
    # and are meaningless for a constrained or purely diffusive track.
    disp_label, vel_label = _directed_units(cal)
    _plot_directed_hist(track_props, cal, "_disp", disp_label,
                        out_dir / "directed_displacement.png")
    _plot_directed_hist(track_props, cal, "_vel", vel_label,
                        out_dir / "directed_velocity.png")
    _plot_directed_joint(track_props, cal,
                         out_dir / "directed_displacement_vs_velocity.png")

    return class_counts(msd_fits)


# ── run-level summary ───────────────────────────────────────────────


def discover_processed(
    root: Path, marker: str = "msd_fits.csv",
) -> list[tuple[str, Path]]:
    """Find every already-processed experiment folder under ``root``.

    An experiment is any directory holding ``marker`` — ``msd_fits.csv`` for a
    summary, ``tracks.csv`` for a re-analysis, both written per stack by
    ``process_file``. Search is recursive, so a root of previous run outputs
    (``root/exp1/stack_a/``) works as well as a flat one (``root/stack_a/``).
    Returns ``(label, folder)`` pairs sorted by label, where the label is the
    folder's path relative to ``root`` — so a direct child is labelled by its
    own name, and a nested one as ``exp1/stack_a``.
    """
    folders = sorted({hit.parent for hit in root.rglob(marker)} - {root})
    return [(f.relative_to(root).as_posix(), f) for f in folders]


#: Anything outside this set is replaced in a summary filename prefix.
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _summary_path(root: Path, name: str) -> Path:
    """Path for one root-level summary output, prefixed with the root's name.

    These files get copied out of their run folder and compared side by side,
    where a dozen identically named ``motility_summary.png`` are useless. The
    prefix is sanitised, since run folders carry spaces and ``=``; a root with
    no usable name (a drive root) falls back to the bare filename.
    """
    stem = _UNSAFE_IN_NAME.sub("_", root.resolve().name).strip("._-")
    return root / (f"{stem}_{name}" if stem else name)


def _at_least(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Drop tracks shorter than ``min_length`` frames.

    Both ``msd_fits.csv`` and ``track_analysis.csv`` carry a ``length`` column,
    so the run-level figures can be re-cut at a new minimum without recomputing
    anything. Raising the minimum works; lowering it past what the original run
    used cannot bring back tracks the MSD pass never fitted — that needs
    ``--analysis-only``.
    """
    if "length" not in df.columns:
        return df
    return df[df["length"] >= min_length]


def summary_rows_from_disk(root: Path, min_length: int) -> list[dict]:
    """Build ``write_run_summary`` rows by reading each msd_fits.csv back."""
    rows = []
    for label, folder in discover_processed(root):
        try:
            fits = pd.read_csv(folder / "msd_fits.csv")
        except Exception as exc:
            print(f"WARN: unreadable {folder / 'msd_fits.csv'}: {exc}",
                  file=sys.stderr)
            continue
        rows.append({"stem": label,
                     "counts": class_counts(_at_least(fits, min_length))})
    return rows


def write_run_summary(root: Path, summary_rows: list[dict], args: Args) -> None:
    """Write the across-experiment summary CSV and figures into ``root``.

    Every input is re-read from disk (``<root>/<stem>/msd_fits.csv`` and
    ``track_analysis.csv``) rather than held in memory, so this works the same
    for a fresh run, a ``--skip-existing`` run, and ``--summary-only``. Both
    tables are cut at ``--min-track-length``, so these figures agree with the
    per-file plots — and ``--summary-only`` can re-cut them at a new minimum.
    """
    if not any(sum(r["counts"].values()) for r in summary_rows):
        # Every experiment came back empty. The figures below all bail out on
        # empty input, which would leave whatever a previous run wrote sitting
        # there looking current, so say plainly what happened.
        print(f"WARN: no track survived --min-track-length "
              f"{args.min_track_length} in any of the {len(summary_rows)} "
              f"experiment(s) — summary figures left unchanged",
              file=sys.stderr)
    summary = pd.DataFrame([
        {"stem": r["stem"], **r["counts"],
         "n_tracks": sum(r["counts"].values())}
        for r in summary_rows
    ]).fillna(0)
    summary.to_csv(_summary_path(root, "motility_summary.csv"), index=False)
    _plot_run_summary(summary_rows, _summary_path(root, "motility_summary.png"))

    # Run-level D vs α. Read back from disk rather than held in memory, so
    # a --skip-existing run still summarises every stack in the folder.
    per_file = []
    for r in summary_rows:
        fits_path = root / r["stem"] / "msd_fits.csv"
        if fits_path.exists():
            per_file.append(
                (r["stem"], _at_least(pd.read_csv(fits_path),
                                      args.min_track_length)))
    cal = Calibration(pixel_size_nm=args.pixel_size_nm,
                      frame_interval_s=args.frame_interval_s,
                      sigma_nm=args.sigma_nm)
    per_file = [(stem, df) for stem, df in per_file if len(df)]
    if per_file:
        _plot_d_vs_alpha_summary(
            per_file, cal, _summary_path(root, "d_vs_alpha_summary.png"))

    # Length / speed summaries come from track_analysis.csv, which carries
    # the geometry as well as the class label.
    per_track = []
    for r in summary_rows:
        props_path = root / r["stem"] / "track_analysis.csv"
        if props_path.exists():
            per_track.append(
                (r["stem"], _props_in_working_units(
                    _at_least(pd.read_csv(props_path), args.min_track_length),
                    cal)))
    per_track = [(stem, df) for stem, df in per_track if len(df)]
    if per_track:
        _plot_metric_summary(
            per_track, "_dur", _axis_label("duration", cal),
            1.0, _summary_path(root, "length_summary.png"))
        _plot_metric_summary(
            per_track, "mean_speed", _axis_label("speed", cal),
            cal.length_scale / cal.delta,
            _summary_path(root, "speed_summary.png"))
        _plot_length_vs_speed_summary(
            per_track, cal,
            _summary_path(root, "length_vs_speed_summary.png"))

        # Directed-only run length and velocity, same replicate treatment.
        per_directed = [(stem, _directed_metrics(df, cal))
                        for stem, df in per_track]
        per_directed = [(stem, d) for stem, d in per_directed if len(d)]
        if per_directed:
            disp_label, vel_label = _directed_units(cal)
            _plot_metric_summary(
                per_directed, "_disp", disp_label, 1.0,
                _summary_path(root, "directed_displacement_summary.png"))
            _plot_metric_summary(
                per_directed, "_vel", vel_label, 1.0,
                _summary_path(root, "directed_velocity_summary.png"))
            _plot_joint_summary(
                per_directed, "_disp", "_vel", disp_label, vel_label,
                "Directed run length vs velocity",
                _summary_path(
                    root, "directed_displacement_vs_velocity_summary.png"),
                color_by_class=False)
    print(f"Motility summary across {len(summary_rows)} file(s): "
          f"{_summary_path(root, 'motility_summary.png')}", file=sys.stderr)


# ── group comparison (--compare) ────────────────────────────────────


#: Categorical hues for comparison groups, assigned in this fixed order and
#: never re-ordered when a group is added or dropped. Chosen by running the
#: six colour checks over the *all-pairs* pairlist rather than the adjacent
#: one: four distributions overlaid in a single axes put every pair side by
#: side, so it is not enough for neighbours in the legend to separate. This
#: set clears it with worst-pair CVD ΔE 16.2 (target ≥ 8) and normal-vision
#: ΔE 19.6 (floor ≥ 15) on white. It is also deliberately disjoint from
#: CLASS_COLORS' blue/orange anchor, so a group hue never reads as a motility
#: class. Yellow and magenta sit below 3:1 contrast on white, which is why
#: every figure drawn with them carries a legend and direct labels — colour
#: alone is not allowed to identify a group.
GROUP_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7",
                "#2a78d6", "#e34948", "#eb6834", "#1baf7a"]
#: Groups past this many stop clearing the all-pairs gate above.
GROUP_COLORS_SAFE = 4
#: Bin edges per comparison histogram. Shared by every group, so the bin width
#: is identical and raw counts can be compared bar for bar.
GROUP_BINS = 45
#: Fill opacity for an overlaid group histogram. The outline stays fully
#: opaque, so where two groups overlap the fills tint each other but each
#: group's own profile is still traceable.
GROUP_FILL_ALPHA = 0.30
#: Points drawn per group in a combined joint figure. Every group is cut to
#: the same number and the draw order is then shuffled, because neither
#: plotting all of them nor plotting them group by group is honest: 95,000
#: overlapping markers saturate, and whichever group is drawn last paints
#: over the rest, so the biggest — or merely the last — group looks dominant.
#: The marginal histograms beside the panel are the unsampled distributions.
GROUP_SCATTER_MAX = 1500
#: Headroom above the tallest bar, so the shared y limit does not clip it.
GROUP_Y_HEADROOM = 1.05
#: Extra headroom on the overlay figure alone, which reserves a band above
#: the curves for the per-experiment diamonds, the direct labels and the
#: legend. Without it the tallest group's label is clipped by the top spine
#: and its diamonds land on its own peak.
GROUP_OVERLAY_HEADROOM = 1.32
#: Where --compare writes, relative to the parent it is pointed at.
COMPARE_DIRNAME = "comparison"


def discover_groups(parent: Path, marker: str = "msd_fits.csv") -> list:
    """Immediate sub-directories of ``parent`` that hold processed results.

    One group per child directory — the DE/EK/PL/WT layout, where each child
    is its own completed bulk run. The marker is searched recursively, so a
    child laid out as ``DE/DE01/<stack>/`` and one laid out as
    ``PL/<stack>/`` are both recognised; :func:`discover_processed` then
    enumerates the experiments *inside* a group exactly as it does for a
    single-root run. This run's own output folder is skipped, so re-running
    ``--compare`` cannot turn a previous comparison into a fifth group.
    Returns ``(name, folder)`` pairs sorted by name.
    """
    if not parent.is_dir():
        return []
    out = []
    for child in sorted(p for p in parent.iterdir() if p.is_dir()):
        if child.name == COMPARE_DIRNAME:
            continue
        if next(child.rglob(marker), None) is not None:
            out.append((child.name, child))
    return out


def _group_colors(names: list, order: list | None = None) -> dict:
    """Hue per group name, assigned in :data:`GROUP_COLORS` order.

    ``order`` names the full set the assignment was made over, so filtering
    the groups actually drawn does not repaint the survivors — colour has to
    follow the group, not its rank in whatever subset a figure shows.
    """
    slots = {name: GROUP_COLORS[i % len(GROUP_COLORS)]
             for i, name in enumerate(names)}
    return slots if order is None else {n: slots[n] for n in order}


def _union_limits(limits) -> tuple:
    """The span covering every ``(lo, hi)`` given — the widest, not the mean.

    This is what makes the per-group figures comparable: each group still
    picks its own percentile cut, and the axis is stretched to hold all of
    them, so no group's tail is silently clipped by another group's range.
    """
    pairs = [(float(lo), float(hi)) for lo, hi in limits]
    if not pairs:
        return 0.0, 1.0
    return min(lo for lo, _ in pairs), max(hi for _, hi in pairs)


def _shared_hist_axes(group_values: dict, n_bins: int = GROUP_BINS) -> tuple:
    """Axis range, bin edges and y ceilings shared by every group.

    Returns ``(lo, hi, edges, y_counts, y_density)``. ``y_counts`` is the
    tallest bin any group puts in those shared bins; ``y_density`` the same
    as a fraction of that group's own tracks, which is the only y axis a
    12,000-track group and a 52,000-track one can honestly share.
    """
    arrays = {}
    for name, values in group_values.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            arrays[name] = v
    lo, hi = _union_limits([_pct_limits(v) for v in arrays.values()])
    edges = np.linspace(lo, hi, n_bins)
    if not arrays:
        return lo, hi, edges, 1.0, 1.0
    counts = {n: np.histogram(v, bins=edges)[0] for n, v in arrays.items()}
    y_counts = max(int(c.max()) for c in counts.values())
    y_density = max(float(c.max()) / arrays[n].size for n, c in counts.items())
    return (lo, hi, edges,
            y_counts * GROUP_Y_HEADROOM, y_density * GROUP_Y_HEADROOM)


def _pooled_values(per_file) -> np.ndarray:
    """Every experiment's values in one array, or an empty one."""
    arrays = [v for _, v in per_file]
    return np.concatenate(arrays) if arrays else np.array([])


def _group_metric(per_group: dict, column: str, scale: float) -> dict:
    """``{group: [(stem, values)]}`` for one metric, in working units."""
    out = {}
    for name, per_file in per_group.items():
        values = _experiment_values(per_file, column, scale)
        if values:
            out[name] = values
    return out


def _plot_group_overlay(
    metric: dict, xlabel: str, out_path: Path, *, colors: dict,
    density: bool = False,
) -> None:
    """Every group's distribution of one metric, overlaid on shared axes.

    Outlines rather than filled bars, since four filled histograms hide each
    other. Each group is named twice — in the legend with its track count, and
    directly at its own peak — because two of the group hues sit below 3:1
    contrast on white, and identity may never rest on colour alone.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _frame_axes(ax)

    names = [n for n in colors if n in metric]
    pooled = {n: _pooled_values(metric[n]) for n in names}
    lo, hi, edges, y_counts, y_density = _shared_hist_axes(pooled)
    y_top = (y_density if density else y_counts) * GROUP_OVERLAY_HEADROOM
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, y_top)

    placed: list = []
    n_off = 0
    for i, name in enumerate(names):
        v = pooled[name]
        if not v.size:
            continue
        weights = np.full(v.size, 1.0 / v.size) if density else None
        counts, _ = np.histogram(v, bins=edges,
                                 weights=weights)
        ax.hist(v, bins=edges, weights=weights, histtype="stepfilled",
                linewidth=2.0, facecolor=to_rgba(colors[name],
                                                 GROUP_FILL_ALPHA),
                edgecolor=colors[name], zorder=3,
                label=f"{name}  n = {v.size:,}")
        # One row of experiment medians per group, in the reserved top band.
        for _, values in metric[name]:
            ax.scatter([np.median(values)], [0.99 - 0.03 * i],
                       transform=ax.get_xaxis_transform(), s=30, marker="D",
                       color=colors[name], edgecolor="white", linewidth=0.8,
                       zorder=5, clip_on=False)
        # Direct label at this group's own peak — the relief the low-contrast
        # hues require, and it survives a greyscale print. Two groups that
        # peak in the same place would print on top of each other, so a label
        # is lifted clear of any already placed near the same spot.
        peak = int(np.argmax(counts))
        lx, ly = 0.5 * (edges[peak] + edges[peak + 1]), counts[peak]
        dy = 6
        for px, py in placed:
            if abs(lx - px) < 0.06 * (hi - lo) and abs(ly - py) < 0.10 * y_top:
                dy += 20
        placed.append((lx, ly))
        ax.annotate(name, xy=(lx, ly), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    va="bottom", fontsize=FONT_ANNOT, color=colors[name],
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=4.0,
                                                foreground="white")])
        n_off += int(((v < lo) | (v > hi)).sum())

    if n_off:
        ax.annotate(f"{n_off} track(s) outside axis range",
                    xy=(0.99, 0.99), xycoords="axes fraction", ha="right",
                    va="top", fontsize=FONT_ANNOT_SMALL, color="#52514e")
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel("fraction of tracks" if density else "# tracks",
                  color="black")
    n_exp = sum(len(metric[n]) for n in names)
    ax.set_title(
        f"{len(names)} groups, {n_exp} experiments"
        f"\n◆ = per-experiment median",
        color="black", fontsize=FONT_TITLE,
    )
    ax.grid(axis="x", visible=False)
    # Anchored below the diamond band, so the two never overlap.
    ax.legend(frameon=False, fontsize=FONT_ANNOT, labelcolor="black",
              loc="upper right", bbox_to_anchor=(1.0, 0.87))
    fig.tight_layout()
    _save(fig, out_path)


def _plot_group_joint(
    per_group: dict, xcol: str, ycol: str, xlabel: str, ylabel: str,
    title: str, out_path: Path, *, colors: dict, limits: tuple,
    log_y: bool = False,
) -> None:
    """Two metrics against each other, every group in one axes.

    Coloured by group rather than by motility class — the comparison here is
    between conditions, and a second meaning for the class hues in the same
    figure would be unreadable. Per-track points are faint; the solid diamond
    per experiment is the replicate-level result.
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05,
                          left=0.17, right=0.97, top=0.87, bottom=0.15)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    for a in (ax, ax_top, ax_right):
        _frame_axes(a)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    x_lo, x_hi, y_lo, y_hi = limits
    x_edges = np.linspace(x_lo, x_hi, 40)
    y_edges = (np.logspace(np.log10(y_lo), np.log10(y_hi), 40) if log_y
               else np.linspace(y_lo, y_hi, 40))

    rng = np.random.default_rng(0)
    n_exp = n_track = 0
    cloud_x, cloud_y, cloud_c = [], [], []
    capped = False
    for name in (n for n in colors if n in per_group):
        frames = [(stem, f) for stem, f in per_group[name] if len(f)]
        if not frames:
            continue
        pooled = pd.concat([f for _, f in frames], ignore_index=True)
        n_exp += len(frames)
        n_track += len(pooled)
        x = pd.to_numeric(pooled[xcol], errors="coerce").to_numpy(float)
        y = pd.to_numeric(pooled[ycol], errors="coerce").to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if x.size > GROUP_SCATTER_MAX:
            capped = True
            pick = rng.choice(x.size, GROUP_SCATTER_MAX, replace=False)
            x, y = x[pick], y[pick]
        cloud_x.append(x)
        cloud_y.append(y)
        cloud_c.append(np.full(x.size, colors[name]))
        for _, f in frames:
            ax.scatter([f[xcol].median()], [f[ycol].median()], s=48,
                       marker="D", color=colors[name], edgecolor="white",
                       linewidth=1.0, zorder=6)
        fill = to_rgba(colors[name], GROUP_FILL_ALPHA)
        ax_top.hist(pooled[xcol], bins=x_edges, histtype="stepfilled",
                    linewidth=1.8, facecolor=fill, edgecolor=colors[name],
                    zorder=3)
        ax_right.hist(pooled[ycol], bins=y_edges, histtype="stepfilled",
                      linewidth=1.8, orientation="horizontal",
                      facecolor=fill, edgecolor=colors[name], zorder=3)
        # Legend proxy, so the label is not sized by the drawn marker.
        ax.plot([], [], marker="D", linestyle="none", color=colors[name],
                markeredgecolor="white", markersize=8,
                label=f"{name}  n = {len(pooled):,}")

    # One shuffled scatter for every group at once: drawing them group by
    # group would let the last one paint over the others.
    if cloud_x:
        x = np.concatenate(cloud_x)
        y = np.concatenate(cloud_y)
        c = np.concatenate(cloud_c)
        order = rng.permutation(x.size)
        ax.scatter(x[order], y[order], s=10, c=c[order], edgecolor="none",
                   alpha=0.30, zorder=2)
        if capped:
            ax.annotate(f"cloud subsampled to {GROUP_SCATTER_MAX:,}/group; "
                        f"marginals show all tracks",
                        xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=FONT_ANNOT_SMALL, color="#52514e")

    if log_y:
        ax.set_yscale("log")
        ax_right.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel, color="black")
    ax.set_ylabel(ylabel, color="black")
    ax_top.set_ylabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_right.set_xlabel("# tracks", color="black", fontsize=FONT_ANNOT)
    ax_top.set_title(
        f"{title} — {n_exp} experiments  ({n_track:,} tracks)"
        f"\n◆ = per-experiment median",
        color="black", fontsize=FONT_TITLE,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax_corner.legend(handles, labels, loc="lower center", frameon=False,
                         fontsize=FONT_ANNOT_SMALL, labelcolor="black",
                         handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
    _save(fig, out_path)


def _plot_group_motility(counts_by_group: dict, out_path: Path) -> None:
    """One 100 %-stacked class bar per group.

    Class colours are kept here — in this figure colour means motility class,
    the same as in every per-run figure, and the group is the x axis.
    """
    usable = [(n, c) for n, c in counts_by_group.items() if sum(c.values())]
    if not usable:
        return
    width = max(FIG_W, 1.4 * len(usable) + 3.0)
    fig, ax = plt.subplots(figsize=(width, FIG_H))
    _frame_axes(ax)
    _stacked_shares(ax, [n for n, _ in usable], [c for _, c in usable])
    ax.set_title("Motility class share per group", color="black", pad=20)
    for i, (_, c) in enumerate(usable):
        ax.annotate(f"n={sum(c.values()):,}", xy=(i, 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FONT_ANNOT_SMALL, color="black")
    leg = ax.legend(frameon=False, fontsize=FONT_ANNOT,
                    ncol=len(MOTILITY_CLASSES) + 1, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
                    labelcolor="black")
    fig.tight_layout()
    _legend_below_labels(fig, ax, leg)
    _save(fig, out_path)


#: One comparison metric: the frame it comes from, its column, and how the
#: axis label and unit scale are derived from the calibration.
_GROUP_METRICS = (
    ("length", "props", "_dur",
     lambda cal: _axis_label("duration", cal),
     lambda cal: 1.0),
    ("speed", "props", "mean_speed",
     lambda cal: _axis_label("speed", cal),
     lambda cal: cal.length_scale / cal.delta),
    ("directed_displacement", "directed", "_disp",
     lambda cal: _directed_units(cal)[0], lambda cal: 1.0),
    ("directed_velocity", "directed", "_vel",
     lambda cal: _directed_units(cal)[1], lambda cal: 1.0),
)


def _read_group_frames(
    folder: Path, name: str, min_length: int, cal: Calibration,
) -> tuple:
    """``(fits, props, counts)`` for one group, read back off disk.

    Each is a list of ``(stem, frame)`` over the group's experiments, cut at
    ``min_length`` exactly as :func:`write_run_summary` cuts them — so a
    group's comparison figure describes the same tracks as its own run-level
    figure.
    """
    fits, props, counts = [], [], {}
    for label, exp in discover_processed(folder):
        fits_path = exp / "msd_fits.csv"
        try:
            f = _at_least(pd.read_csv(fits_path), min_length)
        except Exception as exc:
            print(f"WARN: unreadable {fits_path}: {exc}", file=sys.stderr)
            continue
        counts[label] = class_counts(f)
        if len(f):
            fits.append((f"{name}/{label}", f))
        props_path = exp / "track_analysis.csv"
        if props_path.exists():
            p = _props_in_working_units(
                _at_least(pd.read_csv(props_path), min_length), cal)
            if len(p):
                props.append((f"{name}/{label}", p))
    return fits, props, counts


def write_group_comparison(parent: Path, args: Args) -> Path:
    """Write the cross-group comparison figures under ``parent``.

    Each child directory of ``parent`` is one group. Every metric is drawn
    three ways: overlaid across groups in counts, overlaid again as a fraction
    of each group's tracks, and once per group on the *shared* axis range — so
    a group's own figure can be laid beside another's and read directly, which
    it cannot when each figure silently scales to its own data.

    Returns the output directory. Exits with a message if ``parent`` holds no
    group; that is a mistyped path far more often than an empty run.
    """
    groups = discover_groups(parent)
    if not groups:
        raise SystemExit(
            f"ERROR: no group found under {parent} — --compare expects a "
            f"parent whose sub-directories are each a completed run "
            f"(e.g. Root/DE, Root/WT). Point --summary-only at a single run "
            f"instead.")
    names = [n for n, _ in groups]
    if len(names) > GROUP_COLORS_SAFE:
        print(f"WARN: {len(names)} groups — the group palette is only "
              f"validated for {GROUP_COLORS_SAFE} overlaid distributions. "
              f"Past that, hues start to collide for colour-blind readers; "
              f"prefer comparing {GROUP_COLORS_SAFE} at a time.",
              file=sys.stderr)
    colors = _group_colors(names)
    cal = Calibration(pixel_size_nm=args.pixel_size_nm,
                      frame_interval_s=args.frame_interval_s,
                      sigma_nm=args.sigma_nm)

    out_dir = parent / COMPARE_DIRNAME
    harmonised = out_dir / "harmonised"
    harmonised.mkdir(parents=True, exist_ok=True)
    prefix = _UNSAFE_IN_NAME.sub("_", parent.resolve().name).strip("._-")
    combined = (lambda name: out_dir / (f"{prefix}_combined_{name}"
                                        if prefix else f"combined_{name}"))

    per_fits, per_props, per_directed, per_counts, rows = {}, {}, {}, {}, []
    for name, folder in groups:
        fits, props, counts = _read_group_frames(
            folder, name, args.min_track_length, cal)
        per_fits[name] = fits
        per_props[name] = props
        per_directed[name] = [(stem, _directed_metrics(df, cal))
                              for stem, df in props]
        per_directed[name] = [(s, d) for s, d in per_directed[name] if len(d)]
        per_counts[name] = counts
        for label, c in counts.items():
            rows.append({"group": name, "stem": label, **c,
                         "n_tracks": sum(c.values())})

    pd.DataFrame(rows).fillna(0).to_csv(
        out_dir / (f"{prefix}_comparison_summary.csv" if prefix
                   else "comparison_summary.csv"), index=False)

    # ── 1-D metrics: overlay in counts and density, then each group on the
    #    shared range so the per-group figures are comparable at a glance.
    sources = {"props": per_props, "directed": per_directed}
    for key, source, column, label_of, scale_of in _GROUP_METRICS:
        metric = _group_metric(sources[source], column, scale_of(cal))
        if not metric:
            continue
        xlabel = label_of(cal)
        lo, hi, _, y_counts, y_density = _shared_hist_axes(
            {n: _pooled_values(v) for n, v in metric.items()})
        _plot_group_overlay(metric, xlabel, combined(f"{key}_summary.png"),
                            colors=colors)
        _plot_group_overlay(metric, xlabel,
                            combined(f"{key}_summary_density.png"),
                            colors=colors, density=True)
        for name in metric:
            _plot_metric_summary(
                sources[source][name], column, xlabel, scale_of(cal),
                harmonised / f"{name}_{key}_summary.png",
                limits=(lo, hi), y_max=y_counts, color=colors[name],
                title=name)
            _plot_metric_summary(
                sources[source][name], column, xlabel, scale_of(cal),
                harmonised / f"{name}_{key}_summary_density.png",
                limits=(lo, hi), y_max=y_density, density=True,
                color=colors[name], title=name)

    _write_group_joints(per_fits, per_props, per_directed, cal, colors,
                        combined, harmonised)
    _plot_group_motility(
        {n: _sum_counts(c) for n, c in per_counts.items()},
        combined("motility_summary.png"))

    print(f"Group comparison across {len(names)} group(s) "
          f"({', '.join(names)}): {out_dir}", file=sys.stderr)
    return out_dir


def _sum_counts(counts_by_stem: dict) -> dict:
    """Class counts for a whole group, summed over its experiments."""
    total: dict = {}
    for counts in counts_by_stem.values():
        for cls, n in counts.items():
            total[cls] = total.get(cls, 0) + n
    return total


def _joint_limits(per_group: dict, xcol: str, ycol: str) -> tuple | None:
    """``(x_lo, x_hi, y_lo, y_hi)`` covering every group, or None if empty."""
    spans = []
    for per_file in per_group.values():
        frames = [f for _, f in per_file if len(f)]
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        spans.append((_pct_limits(pooled[xcol]), _pct_limits(pooled[ycol])))
    if not spans:
        return None
    x = _union_limits([s[0] for s in spans])
    y = _union_limits([s[1] for s in spans])
    return (*x, *y)


def _write_group_joints(
    per_fits: dict, per_props: dict, per_directed: dict, cal: Calibration,
    colors: dict, combined, harmonised: Path,
) -> None:
    """The three joint figures: combined by group, then per group on one axis."""
    # D vs α. Limits come from the same helpers the single-run figure uses,
    # so the α cut lines still land where the classification puts them.
    spans = []
    for per_file in per_fits.values():
        frames = [f for _, f in per_file if len(f)]
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        valid = pooled[np.isfinite(pooled["alpha"]) & (pooled["D"] > 0)]
        if len(valid):
            spans.append((_alpha_axis_limits(valid["alpha"]),
                          _d_axis_limits(valid["D"])))
    if spans:
        da_limits = (*_union_limits([s[0] for s in spans]),
                     *_union_limits([s[1] for s in spans]))
        _plot_group_joint(
            per_fits, "alpha", "D", "Scaling exponent α", f"D ({cal.d_unit})",
            "D vs α", combined("d_vs_alpha_summary.png"), colors=colors,
            limits=da_limits, log_y=True)
        for name, per_file in per_fits.items():
            if per_file:
                _plot_d_vs_alpha_summary(
                    per_file, cal,
                    harmonised / f"{name}_d_vs_alpha_summary.png",
                    limits=da_limits)

    per_ls = {n: _length_speed_metrics(p, cal) for n, p in per_props.items()}
    xlabel, ylabel = _length_speed_units(cal)
    ls_limits = _joint_limits(per_ls, "_len", "_spd")
    if ls_limits:
        _plot_group_joint(
            per_ls, "_len", "_spd", xlabel, ylabel, "Length vs speed",
            combined("length_vs_speed_summary.png"), colors=colors,
            limits=ls_limits)
        for name, per_file in per_props.items():
            if per_file:
                _plot_length_vs_speed_summary(
                    per_file, cal,
                    harmonised / f"{name}_length_vs_speed_summary.png",
                    limits=ls_limits)

    disp_label, vel_label = _directed_units(cal)
    dv_limits = _joint_limits(per_directed, "_disp", "_vel")
    if dv_limits:
        name_png = "directed_displacement_vs_velocity_summary.png"
        _plot_group_joint(
            per_directed, "_disp", "_vel", disp_label, vel_label,
            "Directed run length vs velocity", combined(name_png),
            colors=colors, limits=dv_limits)
        for name, per_file in per_directed.items():
            if per_file:
                _plot_joint_summary(
                    per_file, "_disp", "_vel", disp_label, vel_label,
                    "Directed run length vs velocity",
                    harmonised / f"{name}_{name_png}",
                    color_by_class=False, limits=dv_limits)


# ── CLI ─────────────────────────────────────────────────────────────


@dataclass
class Args:
    input_dir: Path | None
    output_dir: Path
    input_file: Path | None
    walk_window: int
    no_walk: bool
    gauss_sigma: float
    no_blur: bool
    no_submean: bool
    detector: str
    prob_thresh: float
    min_distance: int
    log_min_sigma: float
    log_max_sigma: float
    log_num_sigma: int
    log_threshold: float
    device: str
    fit_win: int
    max_radius: float
    search_range: float
    memory: int
    min_track_length: int
    max_tracks_plot: int
    skip_existing: bool
    summary_only: bool
    analysis_only: bool
    compare: Path | None
    pixel_size_nm: float | None
    frame_interval_s: float | None
    sigma_nm: float
    msd_max_lag: int


def parse_args(argv=None) -> Args:
    p = argparse.ArgumentParser(
        description="Bulk preprocess + segment + track + analyze TIFF stacks.",
    )
    p.add_argument("input_dir", type=Path, nargs="?", default=None,
                   help="Directory of .tif/.tiff stacks (T,Y,X). "
                        "Omit when using --in, --summary-only or "
                        "--analysis-only.")
    p.add_argument("output_dir", type=Path, nargs="?", default=None,
                   help="Output directory (created if missing). With "
                        "--summary-only or --analysis-only this is the "
                        "already-processed root that is reworked in place. "
                        "Omit when using --compare.")
    # dest is required: `in` is a keyword, so args.in would be unreachable.
    p.add_argument("--in", dest="input_file", type=Path, default=None,
                   metavar="FILE",
                   help="Analyse this single .tif/.tiff stack instead of a "
                        "directory. Mutually exclusive with input_dir.")

    p.add_argument("--walk-window", type=int, default=4,
                   help="Walking-average window (odd; coerced if even). Default: 4.")
    p.add_argument("--no-walk", action="store_true", help="Skip walking average.")
    p.add_argument("--gauss-sigma", type=float, default=1.0,
                   help="Gaussian-blur sigma (per frame, Y/X only). Default: 1.0.")
    p.add_argument("--no-blur", action="store_true", help="Skip Gaussian blur.")
    p.add_argument("--no-submean", action="store_true", help="Skip stack-mean subtraction.")

    p.add_argument("--detector", choices=["spotiflow", "log"], default="spotiflow",
                   help="Spot detector: Spotiflow (deep learning) or a "
                        "classical Laplacian of Gaussian. Default: spotiflow.")
    p.add_argument("--prob-thresh", type=float, default=0.5,
                   help="Spotiflow probability threshold. Ignored with "
                        "--detector log. Default: 0.5.")
    p.add_argument("--min-distance", type=int, default=2,
                   help="Spotiflow min-distance NMS. Ignored with "
                        "--detector log. Default: 2.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="Compute device (cuda falls back to cpu). Default: cuda.")

    log_grp = p.add_argument_group(
        "LoG detection",
        "Only used with --detector log. Defaults match the Spot Detection "
        "widget's LoG panel.",
    )
    log_grp.add_argument("--log-min-sigma", type=float, default=2.0,
                         help="Smallest blob sigma, in px. Default: 2.0.")
    log_grp.add_argument("--log-max-sigma", type=float, default=10.0,
                         help="Largest blob sigma, in px. Default: 10.0.")
    log_grp.add_argument("--log-num-sigma", type=int, default=10,
                         help="Number of sigmas tried between min and max. "
                              "Default: 10.")
    log_grp.add_argument("--log-threshold", type=float, default=0.09,
                         help="Absolute LoG response threshold on the "
                              "intensity-rescaled frame. Default: 0.09.")

    p.add_argument("--fit-win", type=int, default=2,
                   help="Half-window for GPU Gaussian fit. Default: 2.")
    p.add_argument("--max-radius", type=float, default=2.0,
                   help="Cap for spot mask radius. Default: 2.0.")

    p.add_argument("--search-range", type=float, default=5.0,
                   help="trackpy search_range. Default: 5.0.")
    p.add_argument("--memory", type=int, default=3, help="trackpy memory. Default: 3.")

    p.add_argument("--min-track-length", type=int, default=5,
                   help="Min track length included in plots (CSVs unfiltered). Default: 5.")
    p.add_argument("--max-tracks-plot", type=int, default=200,
                   help="Cap on tracks drawn in per-track plots. Default: 200.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip an input if its output sub-folder already exists.")
    p.add_argument("--summary-only", action="store_true",
                   help="Do not process anything. Treat output_dir as a root "
                        "of already-processed experiments and (re)write only "
                        "the run-level summary CSV and figures there.")
    p.add_argument("--compare", type=Path, default=None, metavar="PARENT",
                   help="Do not process anything. Treat each sub-directory of "
                        "PARENT as one condition (e.g. Root/DE, Root/WT) and "
                        "write cross-condition figures into PARENT/comparison: "
                        "every condition overlaid, and each condition redrawn "
                        "on the axis range they share. Takes no input_dir or "
                        "output_dir.")
    p.add_argument("--analysis-only", action="store_true",
                   help="Do not detect or track. Treat output_dir as a root of "
                        "folders holding tracks.csv and redo the analysis over "
                        "each — useful for trying a new --min-track-length, "
                        "--msd-max-lag or calibration. Run --summary-only "
                        "afterwards to refresh the run-level figures.")

    msd_grp = p.add_argument_group(
        "MSD / motility",
        "Give BOTH --pixel-size-nm and --frame-interval-s to fit in physical "
        "units (D in µm²/s^α) with the --sigma-nm localization-error offset. "
        "Otherwise the MSD stays in px²/frame^α and σ is dropped.",
    )
    msd_grp.add_argument("--pixel-size-nm", type=float, default=None,
                         help="Pixel size in nm. Default: unset (pixel units).")
    msd_grp.add_argument("--frame-interval-s", type=float, default=None,
                         help="Frame interval in seconds. Default: unset (frame units).")
    msd_grp.add_argument("--sigma-nm", type=float, default=35.0,
                         help="Fixed localization error for the 4σ² offset, in nm. "
                              "Only used when calibrated. Default: 35.0.")
    msd_grp.add_argument("--msd-max-lag", type=int, default=25,
                         help="Maximum MSD lag time in frames. Default: 25.")

    ns = p.parse_args(argv)
    # Modes that take no input_dir (--in FILE, --summary-only, --analysis-only)
    # leave argparse a single positional, which it binds to the first optional
    # positional -- input_dir. Shift it to output_dir, where the user meant it;
    # otherwise those modes could never be given an output directory at all.
    if (ns.output_dir is None and ns.input_dir is not None
            and (ns.input_file is not None or ns.summary_only
                 or ns.analysis_only)):
        ns.input_dir, ns.output_dir = None, ns.input_dir
    if ns.summary_only and ns.analysis_only:
        p.error("give at most one of --summary-only or --analysis-only "
                "(run them in turn: analysis first, then summary)")
    if ns.compare is not None:
        if ns.summary_only or ns.analysis_only:
            p.error("--compare is its own mode: it reads runs that "
                    "--summary-only has already written, so run them in turn")
        if (ns.input_dir is not None or ns.input_file is not None
                or ns.output_dir is not None):
            p.error("--compare takes no input or output directory: it writes "
                    "into a 'comparison' folder inside PARENT")
    elif ns.output_dir is None:
        p.error("the following arguments are required: output_dir")
    elif ns.summary_only or ns.analysis_only:
        flag = "--summary-only" if ns.summary_only else "--analysis-only"
        if ns.input_dir is not None or ns.input_file is not None:
            p.error(f"{flag} takes no input: it reworks output_dir in place")
    elif (ns.input_dir is None) == (ns.input_file is None):
        p.error("give exactly one of input_dir or --in FILE")
    if (ns.pixel_size_nm is None) != (ns.frame_interval_s is None):
        p.error("--pixel-size-nm and --frame-interval-s must be given together "
                "(or neither, for pixel/frame units)")
    return Args(
        input_dir=ns.input_dir, output_dir=ns.output_dir,
        input_file=ns.input_file,
        walk_window=ns.walk_window, no_walk=ns.no_walk,
        gauss_sigma=ns.gauss_sigma, no_blur=ns.no_blur, no_submean=ns.no_submean,
        detector=ns.detector,
        prob_thresh=ns.prob_thresh, min_distance=ns.min_distance,
        log_min_sigma=ns.log_min_sigma, log_max_sigma=ns.log_max_sigma,
        log_num_sigma=ns.log_num_sigma, log_threshold=ns.log_threshold,
        device=ns.device,
        fit_win=ns.fit_win, max_radius=ns.max_radius,
        search_range=ns.search_range, memory=ns.memory,
        min_track_length=ns.min_track_length,
        max_tracks_plot=ns.max_tracks_plot,
        skip_existing=ns.skip_existing, summary_only=ns.summary_only,
        analysis_only=ns.analysis_only, compare=ns.compare,
        pixel_size_nm=ns.pixel_size_nm, frame_interval_s=ns.frame_interval_s,
        sigma_nm=ns.sigma_nm, msd_max_lag=ns.msd_max_lag,
    )


def run_analysis_only(args: Args) -> int:
    """Redo the analysis over every folder under output_dir holding tracks.csv.

    Detection and linking are the expensive, unchanged part of the pipeline, so
    a new --min-track-length or calibration only needs this tail rerun. Each
    folder is independent: one failure is logged and the rest continue.
    """
    if not args.output_dir.is_dir():
        print(f"ERROR: output_dir is not a directory: {args.output_dir}",
              file=sys.stderr)
        return 2
    folders = discover_processed(args.output_dir, marker="tracks.csv")
    if not folders:
        print(f"ERROR: no tracked experiment (tracks.csv) found under "
              f"{args.output_dir}", file=sys.stderr)
        return 2

    log_path = args.output_dir / "run.log"
    n_ok = n_fail = 0
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"# analysis-only\tmin_track_length={args.min_track_length}"
                    f"\tmsd_max_lag={args.msd_max_lag}\n")
        for label, folder in tqdm(folders, desc="folders", unit="folder"):
            try:
                tracked = pd.read_csv(folder / "tracks.csv")
                # The preprocessed stack is what the original run sampled
                # intensities from; without it the geometry columns still
                # rewrite, but track_analysis.csv loses its intensity ones.
                pre_path = folder / "preprocessed.tif"
                image = None
                if pre_path.exists():
                    image = tifffile.imread(str(pre_path))
                else:
                    print(f"WARN: {label}: no preprocessed.tif — writing "
                          f"track_analysis.csv without intensity columns",
                          file=sys.stderr)
                counts = analyze_folder(folder, tracked, image, args)
            except Exception:
                tb = traceback.format_exc().replace("\n", " | ")
                log_f.write(f"{label}\tstatus=FAIL\t{tb}\n")
                log_f.flush()
                print(f"FAIL: {label} — see run.log", file=sys.stderr)
                n_fail += 1
                continue
            log_f.write(f"{label}\tn_tracks={sum(counts.values())}"
                        f"\tstatus=OK\t\n")
            log_f.flush()
            n_ok += 1

    print(f"Re-analysed {n_ok} folder(s), {n_fail} failed. "
          f"Run --summary-only on {args.output_dir} to refresh the run-level "
          f"figures.", file=sys.stderr)
    return 0 if n_fail == 0 else 1


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.compare is not None:
        if not args.compare.is_dir():
            print(f"ERROR: --compare is not a directory: {args.compare}",
                  file=sys.stderr)
            return 2
        write_group_comparison(args.compare, args)
        return 0

    if args.summary_only:
        if not args.output_dir.is_dir():
            print(f"ERROR: output_dir is not a directory: {args.output_dir}",
                  file=sys.stderr)
            return 2
        rows = summary_rows_from_disk(args.output_dir, args.min_track_length)
        if not rows:
            print(f"ERROR: no processed experiment (msd_fits.csv) found under "
                  f"{args.output_dir}", file=sys.stderr)
            return 2
        write_run_summary(args.output_dir, rows, args)
        return 0

    if args.analysis_only:
        return run_analysis_only(args)

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"ERROR: --in is not a file: {args.input_file}",
                  file=sys.stderr)
            return 2
        if args.input_file.suffix.lower() not in (".tif", ".tiff"):
            print(f"ERROR: --in must be a .tif/.tiff stack: {args.input_file}",
                  file=sys.stderr)
            return 2
        inputs = [args.input_file]
    else:
        if not args.input_dir.is_dir():
            print(f"ERROR: input_dir is not a directory: {args.input_dir}",
                  file=sys.stderr)
            return 2
        inputs = sorted(
            list(args.input_dir.glob("*.tif")) + list(args.input_dir.glob("*.tiff"))
        )
        if not inputs:
            print(f"ERROR: no .tif/.tiff files found in {args.input_dir}",
                  file=sys.stderr)
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(use_gpu=(args.device == "cuda"))
    if device != args.device:
        print(f"WARN: requested device={args.device}, falling back to {device}",
              file=sys.stderr)
    # Use the resolved device everywhere downstream (model load + GPU fit).
    args.device = device

    if args.detector == "log":
        # Classical detector — no network, so nothing to load. The GPU
        # Gaussian fit downstream still uses the resolved device.
        model = None
        print(f"Detector: LoG (sigma {args.log_min_sigma}-{args.log_max_sigma} "
              f"in {args.log_num_sigma} steps, threshold {args.log_threshold})",
              file=sys.stderr)
    else:
        print(f"Loading Spotiflow 'general' on {device}...", file=sys.stderr)
        model = load_model("general", device=device)

    log_path = args.output_dir / "run.log"
    n_ok = n_fail = n_skip = 0
    summary_rows: list[dict] = []
    if args.detector == "log":
        detector_note = (
            f"detector=log\tmin_sigma={args.log_min_sigma}"
            f"\tmax_sigma={args.log_max_sigma}"
            f"\tnum_sigma={args.log_num_sigma}"
            f"\tthreshold={args.log_threshold}"
        )
    else:
        detector_note = (
            f"detector=spotiflow\tprob_thresh={args.prob_thresh}"
            f"\tmin_distance={args.min_distance}"
        )
    with log_path.open("a", encoding="utf-8") as log_f:
        # Header line: the run is only reproducible if the log says which
        # detector produced the spots.
        log_f.write(f"# {detector_note}\n")
        for tif in tqdm(inputs, desc="files", unit="file"):
            try:
                row = process_file(tif, args.output_dir, model, args)
            except Exception:
                tb = traceback.format_exc()
                row = {"stem": tif.stem, "n_spots": 0, "n_tracks": 0,
                       "status": "FAIL", "error": tb.replace("\n", " | ")}
                print(f"FAIL: {tif.name} — see run.log", file=sys.stderr)

            if row.get("counts"):
                summary_rows.append({"stem": row["stem"], "counts": row["counts"]})

            log_line = (
                f"{row['stem']}\tn_spots={row['n_spots']}"
                f"\tn_tracks={row['n_tracks']}"
                f"\tstatus={row['status']}\t{row['error']}\n"
            )
            log_f.write(log_line)
            log_f.flush()

            if row["status"] == "OK":
                n_ok += 1
            elif row["status"] == "SKIP":
                n_skip += 1
            else:
                n_fail += 1

    # A single --in stack is not a set of replicates, so there is nothing to
    # summarise across; its per-file outputs stand on their own.
    if summary_rows and args.input_file is None:
        write_run_summary(args.output_dir, summary_rows, args)

    print(
        f"Done: {n_ok} OK, {n_skip} skipped, {n_fail} failed. "
        f"Log: {log_path}",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
