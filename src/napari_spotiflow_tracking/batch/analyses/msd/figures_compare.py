"""Cross-group comparison figures: every group overlaid, then redrawn alone
on the axis range they share.

Ported from ``scripts/bulk_track_analysis.py``: ``_shared_hist_axes``
(L2261) -> :func:`shared_hist_axes`; ``union_limits`` (L2248) ->
:func:`union_limits`; ``pooled_values`` (L2286) -> :func:`pooled_values`;
``_group_metric`` (L2292) -> :func:`group_metric`; ``_plot_group_overlay``
(L2302) -> :func:`plot_group_overlay`; ``_plot_group_joint`` (L2383) ->
:func:`plot_group_joint`; ``_plot_group_motility`` (L2489) ->
:func:`plot_group_motility`; ``_joint_limits`` (L2665) -> :func:`joint_limits`.

The per-group "harmonised" figures reuse ``figures_group``'s single-group
plot functions with the ``limits``/``y_max``/``density``/``color``/``title``
keyword-only arguments re-added there for exactly this purpose (see that
module) -- this module supplies the axis range and colour, ``MsdAnalysis.
compare_groups`` (``analysis.py``) drives which metric goes where.

Group colours are never assigned here: :func:`MsdAnalysis.compare_groups`
takes them from :func:`theme.group_colors` over every configured input, so a
group that produced nothing still leaves its neighbours' colours alone. Every
function here only draws a group present in the ``colors`` mapping it is
given, and iterates in ``colors`` order, so the group palette --- not
whatever subset a particular figure happens to have data for --- decides hue
and left-to-right order.

Every function follows the batch pipeline's output contract: empty input
writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from ...._track_properties import MOTILITY_CLASSES
from ... import theme
from . import figures_group
from .figures_movie import (
    FIG_H,
    FIG_W,
    _FONT_ANNOT,
    _FONT_ANNOT_SMALL,
    _finish,
    _grid_behind,
    _pct_limits,
)

#: Bin edges per comparison histogram. Shared by every group, so the bin
#: width is identical and raw counts can be compared bar for bar.
GROUP_BINS = 45
#: Fill opacity for an overlaid group histogram. The outline stays fully
#: opaque, so where two groups overlap the fills tint each other but each
#: group's own profile is still traceable.
GROUP_FILL_ALPHA = 0.30
#: Points drawn per group in a combined joint figure. Every group is cut to
#: the same number and the draw order is then shuffled, because neither
#: plotting all of them nor plotting them group by group is honest: many
#: overlapping markers saturate, and whichever group is drawn last paints
#: over the rest, so the biggest -- or merely the last -- group looks
#: dominant. The marginal histograms beside the panel are the unsampled
#: distributions.
GROUP_SCATTER_MAX = 1500
#: Headroom above the tallest bar, so the shared y limit does not clip it.
GROUP_Y_HEADROOM = 1.05
#: Extra headroom on the overlay figure alone, which reserves a band above
#: the curves for the per-movie diamonds, the direct labels and the legend.
GROUP_OVERLAY_HEADROOM = 1.32
#: Groups past this many stop clearing the all-pairs gate -- see the
#: ``log.warning`` in ``MsdAnalysis.compare_groups``.
GROUP_COLORS_SAFE = 4


def union_limits(limits) -> tuple:
    """The span covering every ``(lo, hi)`` given -- the widest, not the mean.

    This is what makes the per-group figures comparable: each group still
    picks its own percentile cut, and the axis is stretched to hold all of
    them, so no group's tail is silently clipped by another group's range.
    """
    pairs = [(float(lo), float(hi)) for lo, hi in limits]
    if not pairs:
        return 0.0, 1.0
    return min(lo for lo, _ in pairs), max(hi for _, hi in pairs)


def pooled_values(per_movie_values: list) -> np.ndarray:
    """Every movie's values in one array, or an empty one."""
    arrays = [v for _, v in per_movie_values]
    return np.concatenate(arrays) if arrays else np.array([])


def shared_hist_axes(group_values: Mapping[str, np.ndarray],
                     n_bins: int = GROUP_BINS) -> tuple:
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
    lo, hi = union_limits(_pct_limits(v) for v in arrays.values())
    edges = np.linspace(lo, hi, n_bins)
    if not arrays:
        return lo, hi, edges, 1.0, 1.0
    counts = {n: np.histogram(v, bins=edges)[0] for n, v in arrays.items()}
    y_counts = max(int(c.max()) for c in counts.values())
    y_density = max(float(c.max()) / arrays[n].size for n, c in counts.items())
    return (lo, hi, edges,
            y_counts * GROUP_Y_HEADROOM, y_density * GROUP_Y_HEADROOM)


def group_metric(per_group: Mapping[str, list], column: str, scale: float) -> dict:
    """``{group: [(movie, values)]}`` for one metric, in working units.

    ``per_group`` maps a group name to a list of ``(movie_label, frame)``
    pairs (e.g. the props or directed frames a group's movies produced).
    Groups with nothing usable for ``column`` are left out.
    """
    out = {}
    for name, frames in per_group.items():
        values = figures_group._experiment_values(frames, column, scale)
        if values:
            out[name] = values
    return out


def joint_limits(per_group: Mapping[str, list], xcol: str, ycol: str) -> Optional[tuple]:
    """``(x_lo, x_hi, y_lo, y_hi)`` covering every group, or ``None`` if empty."""
    spans = []
    for frames in per_group.values():
        non_empty = [f for _, f in frames if len(f)]
        if not non_empty:
            continue
        pooled = pd.concat(non_empty, ignore_index=True)
        spans.append((_pct_limits(pooled[xcol]), _pct_limits(pooled[ycol])))
    if not spans:
        return None
    x = union_limits(s[0] for s in spans)
    y = union_limits(s[1] for s in spans)
    return (*x, *y)


def plot_group_overlay(
    metric: Mapping[str, list], xlabel: str, stem: Path, formats: Sequence[str],
    dpi: int, *, colors: Mapping[str, str], density: bool = False,
) -> None:
    """Every group's distribution of one metric, overlaid on shared axes.

    Outlines rather than filled bars, since several filled histograms hide
    each other. Each group is named twice -- in the legend with its track
    count, and directly at its own peak -- because colour alone may never be
    what identifies a group (some group hues sit close together for
    colour-vision-deficient readers).
    """
    names = [n for n in colors if n in metric]
    if not names:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        pooled = {n: pooled_values(metric[n]) for n in names}
        lo, hi, edges, y_counts, y_density = shared_hist_axes(pooled)
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
            counts, _ = np.histogram(v, bins=edges, weights=weights)
            ax.hist(v, bins=edges, weights=weights, histtype="stepfilled",
                    linewidth=2.0, facecolor=to_rgba(colors[name], GROUP_FILL_ALPHA),
                    edgecolor=colors[name], zorder=3,
                    label=f"{name}  n = {v.size:,}")
            # One row of per-movie medians per group, in the reserved top band.
            for _, values in metric[name]:
                ax.scatter([np.median(values)], [0.99 - 0.03 * i],
                           transform=ax.get_xaxis_transform(), s=30, marker="D",
                           color=colors[name], edgecolor="white", linewidth=0.8,
                           zorder=5, clip_on=False)
            # Direct label at this group's own peak. Two groups that peak in
            # the same place would print on top of each other, so a label is
            # lifted clear of any already placed near the same spot.
            peak = int(np.argmax(counts))
            lx, ly = 0.5 * (edges[peak] + edges[peak + 1]), counts[peak]
            dy = 6
            for px, py in placed:
                if abs(lx - px) < 0.06 * (hi - lo) and abs(ly - py) < 0.10 * y_top:
                    dy += 20
            placed.append((lx, ly))
            ax.annotate(name, xy=(lx, ly), xytext=(0, dy),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=_FONT_ANNOT, color=colors[name], zorder=6,
                        path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
            n_off += int(((v < lo) | (v > hi)).sum())

        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.99), xycoords="axes fraction", ha="right",
                        va="top", fontsize=_FONT_ANNOT_SMALL, color="#52514e")
        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel("fraction of tracks" if density else "# tracks", color="black")
        n_movies = sum(len(metric[n]) for n in names)
        ax.set_title(
            f"{len(names)} groups, {n_movies} movies"
            f"\n◆ = per-movie median",
            color="black",
        )
        ax.grid(axis="x", visible=False)
        # Anchored below the diamond band, so the two never overlap.
        ax.legend(fontsize=_FONT_ANNOT, labelcolor="black",
                  loc="upper right", bbox_to_anchor=(1.0, 0.87))
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_group_joint(
    per_group: Mapping[str, list], xcol: str, ycol: str, xlabel: str, ylabel: str,
    title: str, stem: Path, formats: Sequence[str], dpi: int, *,
    colors: Mapping[str, str], limits: tuple, log_y: bool = False,
) -> None:
    """Two metrics against each other, every group in one axes.

    Coloured by group rather than by motility class -- the comparison here is
    between conditions. Per-movie points are faint; the solid diamond per
    movie is the replicate-level result.
    """
    names = [n for n in colors if n in per_group
            and any(len(f) for _, f in per_group[n])]
    if not names:
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

        x_lo, x_hi, y_lo, y_hi = limits
        x_edges = np.linspace(x_lo, x_hi, 40)
        y_edges = (np.logspace(np.log10(y_lo), np.log10(y_hi), 40) if log_y
                   else np.linspace(y_lo, y_hi, 40))

        rng = np.random.default_rng(0)
        n_movies = n_track = 0
        cloud_x, cloud_y, cloud_c = [], [], []
        capped = False
        for name in names:
            frames = [(label, f) for label, f in per_group[name] if len(f)]
            pooled = pd.concat([f for _, f in frames], ignore_index=True)
            n_movies += len(frames)
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
                            va="bottom", fontsize=_FONT_ANNOT_SMALL, color="#52514e")

        if log_y:
            ax.set_yscale("log")
            ax_right.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel(ylabel, color="black")
        ax_top.set_ylabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_right.set_xlabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_top.set_title(
            f"{title} — {n_movies} movies  ({n_track:,} tracks)"
            f"\n◆ = per-movie median",
            color="black",
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_corner.legend(handles, labels, loc="lower center",
                             fontsize=_FONT_ANNOT_SMALL, labelcolor="black",
                             handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
        _finish(fig, stem, formats, dpi)


def plot_group_motility(counts_by_group: Mapping[str, dict], stem: Path,
                        formats: Sequence[str], dpi: int) -> None:
    """One 100%-stacked class bar per group.

    Class colours are kept here -- in this figure colour means motility
    class, the same as in every per-movie figure, and the group is the x
    axis.
    """
    usable = [(n, c) for n, c in counts_by_group.items() if sum(c.values())]
    if not usable:
        return
    labels = [n for n, _ in usable]
    counts = [c for _, c in usable]
    width = max(FIG_W, 1.4 * len(usable) + 3.0)
    with theme.style():
        fig = Figure(figsize=(width, FIG_H))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        _grid_behind(ax)
        figures_group._stacked_shares(ax, labels, counts)
        ax.set_title("Motility class share per group", color="black", pad=20)
        for i, c in enumerate(counts):
            ax.annotate(f"n={sum(c.values()):,}", xy=(i, 100), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=_FONT_ANNOT_SMALL, color="black")
        leg = ax.legend(fontsize=_FONT_ANNOT, ncol=len(MOTILITY_CLASSES) + 1,
                        loc="lower center", bbox_to_anchor=(0.5, 0.0),
                        bbox_transform=fig.transFigure, labelcolor="black")
        fig.tight_layout()
        figures_group._legend_below_labels(fig, ax, leg)
        _finish(fig, stem, formats, dpi)
