"""Group-level summaries: pool a group's movies from their per-movie CSVs.

Ported from ``scripts/bulk_track_analysis.py``: ``_read_group_frames``
(L2532) and ``summary_rows_from_disk`` (L2068) -> :func:`read_movie_frames`;
the motility bars of ``write_run_summary`` (``_plot_run_summary`` L1811 and
``_stacked_shares`` L700) -> :func:`plot_motility_summary`;
``_plot_d_vs_alpha_summary`` (L1040) -> :func:`plot_d_vs_alpha_summary`;
``_plot_metric_summary`` (L1191) + ``_experiment_values`` (L1171) ->
:func:`plot_metric_summary`; ``_plot_joint_summary`` (L1379) ->
:func:`plot_joint_summary`; ``_plot_length_vs_speed_summary`` (L1497) +
``_length_speed_metrics`` (L1472) -> :func:`plot_length_vs_speed_summary`;
``_at_least`` (L2054) -> :func:`_at_least`.

Movies are read straight off disk (``<movie>/msd/msd_fits.csv`` and
``track_analysis.csv``) rather than kept in memory, so this also covers
movies skipped via ``skip_existing``. A movie missing either file is left
out entirely, mirroring ``summary_rows_from_disk``'s skip of an unreadable
``msd_fits.csv``.

Class colours are :data:`theme.MOTILITY_COLORS`, matching the source's
per-class colouring; a plain, non-class hue (``"tab:blue"`` in the source)
is :data:`theme.NEUTRAL`. The one single-hue scatter that is still
implicitly about directed tracks alone (the ``color_by_class=False`` branch
of the joint plot) keeps :data:`theme.MOTILITY_COLORS`'s ``"directed"``
entry, matching the equivalent movie-level figure
(:func:`figures_movie.plot_directed_joint`).

Every function here follows the batch pipeline's output contract: empty
input writes nothing, rather than the source's "no data" placeholder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ...._track_properties import (
    ALPHA_CONSTRAINED_MAX,
    ALPHA_DIRECTED_MIN,
    MOTILITY_CLASSES,
)
from ... import theme
from . import compute
from .figures_movie import (
    FIG_H,
    FIG_W,
    _FONT_ANNOT,
    _FONT_ANNOT_SMALL,
    _alpha_axis_limits,
    _alpha_criterion,
    _d_axis_limits,
    _finish,
    _grid_behind,
    _pct_limits,
)
from .units import Units

log = logging.getLogger("napari_spotiflow_tracking.batch")

#: Bin edges per pooled histogram — shared across the four metric summaries.
GROUP_BINS = 45


def _at_least(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Drop tracks shorter than ``min_length`` frames.

    Both ``msd_fits.csv`` and ``track_analysis.csv`` carry a ``length``
    column, so the group figures can be re-cut at a new minimum without
    recomputing anything.
    """
    if "length" not in df.columns:
        return df
    return df[df["length"] >= min_length]


def _duration_column(props: pd.DataFrame, units: Units) -> pd.Series:
    """Per-track duration in the working time unit, preferring the on-disk column.

    ``track_analysis.csv`` already carries ``duration_s`` when the run that
    wrote it was calibrated (:func:`_track_properties.add_physical_columns`)
    -- reuse it rather than recompute. A calibrated *summary* run reading an
    older or uncalibrated movie's table -- e.g. under ``skip_existing`` --
    has no such column, so it is derived here with the same formula
    (``add_physical_columns``, ``bulk:_props_in_working_units`` L1780-1794):
    an n-frame track spans (n - 1) intervals.
    """
    if units.is_physical:
        if "duration_s" in props.columns:
            return props["duration_s"]
        return (props["length"].astype(float) - 1.0) * units.delta
    return props["length"].astype(float)


def read_movie_frames(movie_dirs, min_length: int, units: Units) -> list:
    """``(label, fits, props, directed)`` for every movie with a ``msd_fits.csv``.

    Reads ``<movie_dir>/msd/msd_fits.csv`` and ``track_analysis.csv`` back
    off disk and cuts both at ``min_length``, so a group summary describes
    the same tracks as its own per-movie figures. ``label`` is the movie
    folder's name.

    A movie with no ``msd_fits.csv`` (failed, or run with ``tables: none``)
    is left out entirely. A movie with ``msd_fits.csv`` but no
    ``track_analysis.csv`` (e.g. run with ``tables: [msd_fits]``) is still
    included, with ``props`` and ``directed`` as ``None`` -- it still
    contributes to the fits-only summaries (``motility_summary``,
    ``d_vs_alpha_summary``), matching ``bulk:_read_group_frames``, which
    reads fits and props independently. An unreadable CSV (truncated,
    locked) is logged and treated the same as missing, rather than failing
    the whole group, matching ``bulk:summary_rows_from_disk`` /
    ``bulk:_read_group_frames``.
    """
    frames = []
    for movie_dir in movie_dirs:
        movie_dir = Path(movie_dir)
        fits_path = movie_dir / "msd" / "msd_fits.csv"
        if not fits_path.exists():
            continue
        try:
            fits = _at_least(pd.read_csv(fits_path), min_length)
        except Exception as exc:
            log.warning("%s: could not read %s (%s: %s); movie left out of the "
                       "group summary", movie_dir.name, fits_path,
                       type(exc).__name__, exc)
            continue

        props, directed = None, None
        props_path = movie_dir / "msd" / "track_analysis.csv"
        if props_path.exists():
            try:
                props = _at_least(pd.read_csv(props_path), min_length).copy()
            except Exception as exc:
                log.warning("%s: could not read %s (%s: %s); movie kept for "
                           "fits-only summaries", movie_dir.name, props_path,
                           type(exc).__name__, exc)
                props = None
            if props is not None:
                props["_dur"] = _duration_column(props, units)
                directed = compute.directed_metrics(props, units)
        frames.append((movie_dir.name, fits, props, directed))
    return frames


def motility_summary_table(per_movie: list) -> pd.DataFrame:
    """One row per movie: its motility class counts, plus ``n_tracks``."""
    rows = [
        {"movie": label, **compute.class_counts(fits),
         "n_tracks": int(len(fits))}
        for label, fits, _, _ in per_movie
    ]
    return pd.DataFrame(rows).fillna(0)


def group_fits_table(per_movie: list) -> pd.DataFrame:
    """Every movie's ``msd_fits`` rows, concatenated with a ``movie`` column.

    ``per_movie`` is never empty here -- ``summarize_group`` already returns
    before calling this when :func:`read_movie_frames` yields nothing, and
    every entry it does yield carries a (possibly empty) ``fits`` frame.
    """
    frames = [fits.assign(movie=label) for label, fits, _, _ in per_movie]
    return pd.concat(frames, ignore_index=True)


def _stacked_shares(ax, x_labels: list, per_item: list) -> None:
    """Draw 100%-stacked class bars, least->most motile bottom->top."""
    classes = list(MOTILITY_CLASSES)
    if any("unclassified" in d for d in per_item):
        classes.append("unclassified")
    bottoms = np.zeros(len(per_item))
    for cls in classes:
        vals = np.array([
            100.0 * d.get(cls, 0) / max(sum(d.values()), 1) for d in per_item
        ])
        ax.bar(x_labels, vals, bottom=bottoms, color=theme.MOTILITY_COLORS[cls],
               width=0.62, label=cls, zorder=2)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 8.0:
                ax.annotate(f"{v:.0f}%", xy=(i, b + v / 2), ha="center",
                            va="center", fontsize=_FONT_ANNOT_SMALL,
                            color="white", zorder=3)
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of tracks", color="black")
    ax.grid(axis="x", visible=False)


def _legend_below_labels(fig, ax, leg) -> None:
    """Raise the axes so the rotated x labels and ``leg`` both fit under it.

    ``tight_layout`` does not account for a legend anchored outside the
    axes, so any fixed offset collides as soon as the labels get long — a
    nested movie label is taller at 30 degrees than a bare one. Measure both
    instead of guessing, and leave the figure height alone so the PNG still
    drops into a manuscript layout unscaled.
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


def plot_motility_summary(per_movie: list, stem: Path, formats, dpi: int) -> None:
    """One 100%-stacked class bar per movie, for cross-movie reading."""
    usable = []
    for label, fits, _, _ in per_movie:
        counts = compute.class_counts(fits)
        if sum(counts.values()) > 0:
            usable.append((label, counts))
    if not usable:
        return
    labels = [label for label, _ in usable]
    counts = [c for _, c in usable]
    width = max(FIG_W, 1.15 * len(usable) + 3.0)
    with theme.style():
        fig = Figure(figsize=(width, FIG_H))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        _grid_behind(ax)
        _stacked_shares(ax, labels, counts)
        # pad clears the per-bar n= annotations sitting on the 100% line.
        ax.set_title("Motility class share per movie", color="black", pad=20)
        for i, c in enumerate(counts):
            ax.annotate(f"n={sum(c.values())}", xy=(i, 100),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=_FONT_ANNOT_SMALL,
                        color="black")
        leg = ax.legend(fontsize=_FONT_ANNOT, ncol=len(MOTILITY_CLASSES) + 1,
                        loc="lower center", bbox_to_anchor=(0.5, 0.0),
                        bbox_transform=fig.transFigure, labelcolor="black")
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        fig.tight_layout()
        _legend_below_labels(fig, ax, leg)
        _finish(fig, stem, formats, dpi)


def plot_d_vs_alpha_summary(per_movie: list, units: Units, stem: Path,
                            formats, dpi: int) -> None:
    """Group-level D vs alpha: every track, plus one marker per movie.

    The movie, not the track, is the unit of replication — see
    :func:`figures_movie.plot_msd_combined`-adjacent commentary on
    ``per_file`` in the source. Each movie also gets a single marker at its
    median (alpha, D); the title reports n as the number of movies. This is
    the SuperPlot convention of Lord et al., J Cell Biol 219(6):e202001064
    (2020).
    """
    per_file = [(label, fits) for label, fits, _, _ in per_movie if len(fits)]
    if not per_file:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              wspace=0.05, hspace=0.05,
                              left=0.17, right=0.97, top=0.87, bottom=0.15)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_corner = fig.add_subplot(gs[0, 1])
        ax_corner.axis("off")
        for a in (ax, ax_top, ax_right):
            _grid_behind(a)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)

        pooled = pd.concat([f for _, f in per_file], ignore_index=True)
        valid = pooled[np.isfinite(pooled["alpha"]) & (pooled["D"] > 0)]

        for cls in MOTILITY_CLASSES:
            sel = valid[valid["motility_class"] == cls]
            if not len(sel):
                continue
            ax.scatter(sel["alpha"], sel["D"], s=14, color=theme.MOTILITY_COLORS[cls],
                       edgecolor="none", alpha=0.45, zorder=2,
                       label=f"{cls}\nn = {len(sel)}")

        n_movies = 0
        for _, f in per_file:
            good = f[np.isfinite(f["alpha"]) & (f["D"] > 0)]
            if not len(good):
                continue
            n_movies += 1
            ax.scatter([good["alpha"].median()], [good["D"].median()], s=35,
                       marker="D", color="black", edgecolor="white", linewidth=0.8,
                       alpha=0.75, zorder=6)

        if len(valid):
            ax.set_yscale("log")
            d_lo, d_hi = _d_axis_limits(valid["D"])
            a_lo, a_hi = _alpha_axis_limits(valid["alpha"])
            ax.set_ylim(d_lo, d_hi)
            ax.set_xlim(a_lo, a_hi)
            ax_top.hist(valid["alpha"], bins=np.linspace(a_lo, a_hi, 40),
                        color=theme.NEUTRAL, zorder=2)
            ax_right.hist(valid["D"], bins=np.logspace(
                np.log10(d_lo), np.log10(d_hi), 40),
                orientation="horizontal", color=theme.NEUTRAL, zorder=2)
            n_off = int((
                (valid["D"] < d_lo) | (valid["D"] > d_hi)
                | (valid["alpha"] < a_lo) | (valid["alpha"] > a_hi)
            ).sum())
            if n_off:
                ax.annotate(f"{n_off} track(s) outside axis range",
                            xy=(0.99, 0.01), xycoords="axes fraction",
                            ha="right", va="bottom", fontsize=_FONT_ANNOT_SMALL,
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
                            fontsize=_FONT_ANNOT_SMALL, color="black",
                            rotation=90, va="bottom", ha="left", zorder=7,
                            path_effects=[pe.withStroke(linewidth=5.0,
                                                        foreground="white")])

        ax.set_xlabel(theme.axis_label("alpha", units.calibration), color="black")
        ax.set_ylabel(theme.axis_label("diffusion", units.calibration), color="black")
        ax_top.set_ylabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_right.set_xlabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_top.set_title(
            f"D vs α — n = {n_movies} movies  ({len(valid):,} tracks)"
            f"\n◆ = per-movie median",
            color="black",
        )
        if len(valid):
            handles, labels = ax.get_legend_handles_labels()
            ax_corner.legend(handles, labels, loc="lower center",
                             fontsize=_FONT_ANNOT_SMALL, labelcolor="black",
                             handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
        _finish(fig, stem, formats, dpi)


def _experiment_values(frames: list, column: str, scale: float) -> list:
    """Per-movie arrays of one track property, in working units.

    Unclassified tracks are dropped, so this describes the same population
    as :func:`plot_d_vs_alpha_summary` and the other summaries rather than
    the full (pre-MSD) linked set.
    """
    out = []
    for label, df in frames:
        if column not in df.columns or not len(df):
            continue
        keep = (df[df["motility_class"] != "unclassified"]
               if "motility_class" in df.columns else df)
        v = pd.to_numeric(keep[column], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)] * scale
        if v.size:
            out.append((label, v))
    return out


def plot_metric_summary(frames: list, column: str, xlabel: str, scale: float,
                        stem: Path, formats, dpi: int) -> None:
    """Group-level distribution of one track property, with per-movie medians.

    The histogram is every track pooled; the diamonds are one per movie. The
    spread of the diamonds — not the width of the histogram — is what says
    whether the movies agree, because pooling weights each movie by however
    many tracks it happened to yield.
    """
    groups = _experiment_values(frames, column, scale)
    if not groups:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        pooled = np.concatenate([v for _, v in groups])
        lo, hi = _pct_limits(pooled)
        ax.hist(pooled, bins=np.linspace(lo, hi, GROUP_BINS), color=theme.NEUTRAL,
               zorder=2)
        ax.set_xlim(lo, hi)

        for _, v in groups:
            ax.scatter([np.median(v)], [0.95], transform=ax.get_xaxis_transform(),
                      s=35, marker="D", color="black", edgecolor="white",
                      linewidth=0.8, alpha=0.75, zorder=5, clip_on=False)

        n_off = int(((pooled < lo) | (pooled > hi)).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.94), xycoords="axes fraction", ha="right",
                        va="top", fontsize=_FONT_ANNOT_SMALL, color="#52514e")
        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel("# tracks", color="black")
        ax.set_title(
            f"n = {len(groups)} movies  ({len(pooled):,} tracks)"
            f"\n◆ = per-movie median",
            color="black",
        )
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_joint_summary(frames: list, xcol: str, ycol: str, xlabel: str, ylabel: str,
                       title: str, stem: Path, formats, dpi: int,
                       color_by_class: bool = True) -> None:
    """Group-level joint distribution of two track properties.

    ``frames`` must already carry ``xcol``/``ycol`` in working units. One
    marker per movie sits at its median, since the movie and not the track
    is the unit of replication.
    """
    non_empty = [(label, f) for label, f in frames if len(f)]
    if not non_empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              wspace=0.05, hspace=0.05,
                              left=0.17, right=0.97, top=0.87, bottom=0.15)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_corner = fig.add_subplot(gs[0, 1])
        ax_corner.axis("off")
        for a in (ax, ax_top, ax_right):
            _grid_behind(a)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)

        pooled = pd.concat([f for _, f in non_empty], ignore_index=True)

        if color_by_class and "motility_class" in pooled.columns:
            for cls in MOTILITY_CLASSES:
                sel = pooled[pooled["motility_class"] == cls]
                if not len(sel):
                    continue
                ax.scatter(sel[xcol], sel[ycol], s=14, color=theme.MOTILITY_COLORS[cls],
                          edgecolor="none", alpha=0.45, zorder=2,
                          label=f"{cls}\nn = {len(sel)}")
        else:
            ax.scatter(pooled[xcol], pooled[ycol], s=14,
                      color=theme.MOTILITY_COLORS["directed"], edgecolor="none",
                      alpha=0.45, zorder=2)
        for _, f in non_empty:
            ax.scatter([f[xcol].median()], [f[ycol].median()], s=35, marker="D",
                      color="black", edgecolor="white", linewidth=0.8, alpha=0.75,
                      zorder=6)

        x_lo, x_hi = _pct_limits(pooled[xcol])
        y_lo, y_hi = _pct_limits(pooled[ycol])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax_top.hist(pooled[xcol], bins=np.linspace(x_lo, x_hi, 40),
                   color=theme.NEUTRAL, zorder=2)
        ax_right.hist(pooled[ycol], bins=np.linspace(y_lo, y_hi, 40),
                     orientation="horizontal", color=theme.NEUTRAL, zorder=2)
        n_off = int((
            (pooled[xcol] < x_lo) | (pooled[xcol] > x_hi)
            | (pooled[ycol] < y_lo) | (pooled[ycol] > y_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=_FONT_ANNOT_SMALL, color="#52514e")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_corner.legend(handles, labels, loc="lower center",
                             fontsize=_FONT_ANNOT_SMALL, labelcolor="black",
                             handletextpad=0.4, labelspacing=0.3, borderpad=0.1)

        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel(ylabel, color="black")
        ax_top.set_ylabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_right.set_xlabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_top.set_title(
            f"{title} — n = {len(non_empty)} movies  ({len(pooled):,} tracks)"
            f"\n◆ = per-movie median",
            color="black",
        )
        _finish(fig, stem, formats, dpi)


def _length_speed_metrics(frames: list, units: Units) -> list:
    """``frames`` (label, props) with ``_len``/``_spd`` columns in working units.

    Unclassified tracks are dropped, so this describes the same population
    as every other summary rather than the full linked set.
    """
    spd_scale = units.length_scale / units.delta
    out = []
    for label, df in frames:
        if not len(df) or not {"mean_speed", "_dur"} <= set(df.columns):
            continue
        keep = df[df["motility_class"] != "unclassified"].copy()
        if not len(keep):
            continue
        keep["_len"] = keep["_dur"].astype(float)
        keep["_spd"] = keep["mean_speed"].astype(float) * spd_scale
        out.append((label, keep))
    return out


def plot_length_vs_speed_summary(per_movie: list, units: Units, stem: Path,
                                 formats, dpi: int) -> None:
    """Group-level track length against mean speed, with per-movie medians.

    Movies with no ``track_analysis.csv`` (``props is None``, see
    :func:`read_movie_frames`) are skipped -- there is nothing to plot for
    them here.
    """
    frames = _length_speed_metrics(
        [(label, props) for label, _, props, _ in per_movie if props is not None],
        units)
    plot_joint_summary(
        frames, "_len", "_spd",
        theme.axis_label("duration", units.calibration),
        f"Mean speed ({theme.unit('speed', units.calibration)})",
        "Length vs speed", stem, formats, dpi)
