"""Core per-movie MSD figures, plus per-track distribution and directed-motion figures.

Ported from ``scripts/bulk_track_analysis.py``: ``_plot_msd_combined`` (L616),
``_plot_motility_classes`` (L724), ``_plot_msd_regimes`` (L749),
``_plot_d_vs_alpha`` (L931), ``_plot_length_hist`` (L406),
``_plot_path_length_hist`` (L440), ``_plot_speed_hist`` (L466),
``_plot_length_per_track`` (L491), ``_plot_speed_per_track`` (L514),
``_plot_directed_hist`` (L1289), ``_plot_directed_joint`` (L1323),
``_plot_msd_construction`` (L1556), ``_plot_msd_archetypes`` (L1634) and
``_plot_msd_shape_comparison`` (L1710) — the last three following Figure 1 of
Schirripa Spagnolo & Luin, *Int. J. Mol. Sci.* 25, 8660 (2024) — plus the
helpers they depend on (``_frame_axes``, ``_thin_ticks``, ``_fit_titles``,
``_alpha_criterion``, ``_d_axis_limits``, ``_alpha_axis_limits``,
``_annotate_corner``, ``_pct_limits``, ``_durations``, ``_top_n_track_ids``,
``_representative_track``, ``_track_xy``, ``_fit_curve``, ``_draw_scale_bar``).
Spine visibility and font sizes are no longer set per figure —
``theme.style()`` provides them — so panels now show the top/right-hidden
frame every other batch figure uses instead of the source script's full box.
Class colours come from :data:`theme.MOTILITY_COLORS` rather than the old
``tab:``/``CLASS_COLORS`` palette; plain (unclassed) histograms and marginals
use :data:`theme.NEUTRAL`; the per-track line plots (many raw, unclassed
traces) use :data:`theme.RAW_TRACE` rather than matplotlib's default colour
cycle.

Unlike the source script, every function here returns without writing
anything when its input is empty, per the batch pipeline's output contract —
the source always drew a (possibly blank or "no data") figure. A function
whose *inputs* are non-empty but which cannot find a representative track for
one panel (e.g. no track of a given motility class) still draws that panel's
"no data" placeholder, matching the source — only a wholly empty input skips
the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ...._track_properties import (
    ALPHA_CONSTRAINED_MAX,
    ALPHA_DIRECTED_MIN,
    MOTILITY_CLASSES,
    msd_basic_fit,
)
from ... import theme
from ...outputs import save_figure
from . import compute
from .units import Units

# Every figure is saved at the same height, and every single-panel figure at
# the same width, so the PNGs drop into a manuscript layout without
# rescaling. A multi-panel figure would widen by FIG_W_PANEL per panel.
FIG_H = 5.0
FIG_W = 7.0
FIG_W_PANEL = 4.5
# The archetype figure is exported at a fixed full-width size, split evenly
# over its three panels: 6800 x 1500 px at the source script's 300 dpi.
FIG_W_ARCHETYPES = 6800 / 300
# Lag index is ordered, not categorical — one hue, light->dark, never a rainbow.
LAG_RAMP = ["#bcd7f5", "#8bb9ec", "#5a9ae2", "#2a78d6", "#17457f"]
# Lags supported by fewer than this many tracks are too noisy to average
# (mirrors compute.MIN_TRACKS_PER_LAG, which ensemble_msd already applies).
MIN_TRACKS_PER_LAG = 3

# theme.style() sets font.size/axes.labelsize/legend.fontsize etc. for the
# rc-driven elements (ticks, axis labels, titles, legend text). These two
# cover the individual in-plot annotations and small marginal-axis labels
# that sit smaller than font.size and aren't reachable through an rcParam.
_FONT_ANNOT = 14
_FONT_ANNOT_SMALL = 12


def _grid_behind(ax) -> None:
    """Put a faint grid behind the marks. Spine visibility is theme.style()'s job."""
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
    constant, and at the rc title size those can run off the edge of a
    fixed-width figure — something ``tight_layout`` does not correct, since
    it fits the axes and not the text above them. Only the titles that
    overrun lose a point; every other one keeps the rc size.
    """
    renderer = fig.canvas.get_renderer()
    limit = fig.bbox.width
    for ax in fig.axes:
        title = ax.title
        if not title.get_text().strip():
            continue
        size = title.get_fontsize()
        while size > _FONT_ANNOT_SMALL:
            box = title.get_window_extent(renderer=renderer)
            if box.x0 >= 0 and box.x1 <= limit:
                break
            size -= 1
            title.set_fontsize(size)


def _finish(fig, stem: Path, formats, dpi: int) -> None:
    """Thin tick labels, fit titles to width, then write the figure out."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()  # settles the limits, so the tick positions are final
    for ax in fig.axes:
        _thin_ticks(ax)
    _fit_titles(fig)
    save_figure(fig, stem, formats, dpi)


def _alpha_criterion(cls: str) -> str:
    """The alpha rule for a class, worded exactly as ``classify_motility`` applies it.

    Both boundaries fall to ``diffusive``: that function sends ``a < 0.4`` to
    constrained and ``a > 1.2`` to directed, so alpha of exactly 0.4 or 1.2 is
    diffusive — hence "<=" on both sides rather than strict inequalities.
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
    """x-limits for an alpha axis, clamped against runaway fits.

    A non-convergent fit can return alpha of 15 where 99.5 % of tracks sit
    below 4, which leaves most of the axis blank. Clamp to the bulk, but
    always keep room past the directed cut so that band stays visible even
    when no track reaches it.
    """
    a = np.asarray(alpha_values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return -0.5, ALPHA_DIRECTED_MIN + 1.0
    lo = float(np.percentile(a, 0.5))
    hi = max(float(np.percentile(a, 99.5)), ALPHA_DIRECTED_MIN * 1.4)
    pad = 0.05 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _annotate_corner(ax, text: str) -> None:
    """Top-right annotation, stroked so it stays legible over the data."""
    ax.annotate(
        text, xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=_FONT_ANNOT, color="black",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
    )


def _pct_limits(values, lo_pct: float = 0.5, hi_pct: float = 99.5, pad: float = 0.05) -> tuple:
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


def _durations(table: pd.DataFrame, units: Units) -> np.ndarray:
    """Per-track duration in the working time unit.

    A track spans the intervals *between* its localizations, so an n-frame
    track lasts (n - 1) intervals. Uncalibrated this stays the raw frame
    count, which is what ``min_length`` cuts on.
    """
    n = table["length"].to_numpy(dtype=float)
    return (n - 1.0) * units.delta if units.is_physical else n


def _top_n_track_ids(tracked: pd.DataFrame, max_tracks: int) -> list:
    """Up to ``max_tracks`` particle IDs, longest tracks first."""
    if "particle" not in tracked.columns or len(tracked) == 0:
        return []
    counts = tracked.groupby("particle").size().sort_values(ascending=False)
    return counts.head(max_tracks).index.tolist()


def plot_msd_combined(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Log-log MSD: every track faint and colored by class, ensemble mean bold.

    The ensemble mean carries its own ``4*D*t^alpha + 4*sigma^2`` fit,
    annotated with the recovered D and alpha.
    """
    if curves.empty or fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        class_of = dict(zip(fits["track_id"], fits["motility_class"]))
        drawn: dict = {}
        for tid, grp in curves.groupby("track_id"):
            cls = class_of.get(tid, "unclassified")
            grp = grp.sort_values("lag")
            ax.plot(grp["lag"], grp["msd"], color=theme.MOTILITY_COLORS[cls],
                    alpha=0.22, linewidth=0.8, zorder=1)
            drawn[cls] = drawn.get(cls, 0) + 1

        ens = compute.ensemble_msd(curves)
        fit_label = ""
        if len(ens) >= 2:
            # 2px surface ring so the bold mean stays legible over the thicket.
            ax.plot(ens["lag"], ens["msd"], color="black", linewidth=2.0, zorder=4,
                    path_effects=[pe.withStroke(linewidth=4.0, foreground="white")],
                    label=f"Ensemble mean (n={int(ens['n_tracks'].iloc[0])})")
            try:
                fit = msd_basic_fit(
                    pd.Series(ens["msd"].to_numpy(), index=np.arange(1, len(ens) + 1)),
                    delta=units.delta, sigma=units.sigma,
                )
                ax.plot(ens["lag"], fit["y_fit"], color="black", linewidth=1.6,
                        linestyle="--", zorder=5,
                        path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
                        label="Fit  4·D·t$^α$ + 4σ²")
                fit_label = (
                    f"D = {fit['diffusion_coefficient']:.3g} {units.d_unit}    "
                    f"α = {fit['alpha']:.3g}"
                )
            except Exception:
                fit_label = "ensemble fit did not converge"

        if len(curves):
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(theme.axis_label("lag", units.calibration), color="black")
        ax.set_ylabel(theme.axis_label("msd", units.calibration), color="black")
        ax.set_title(f"MSD — {len(fits)} tracks · {units.describe()}", color="black")
        if fit_label:
            ax.annotate(fit_label, xy=(0.02, 0.97), xycoords="axes fraction",
                        va="top", ha="left", fontsize=_FONT_ANNOT, color="black")

        handles = [
            Line2D([], [], color=theme.MOTILITY_COLORS[c], linewidth=2.0,
                   label=f"{c} ({drawn[c]})")
            for c in (*MOTILITY_CLASSES, "unclassified") if drawn.get(c)
        ]
        ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
                  loc="lower right", labelcolor="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_motility_classes(
    fits: pd.DataFrame,
    stem: Path,
    formats,
    dpi: int,
    *,
    _capture: dict | None = None,
) -> None:
    """Absolute class counts, each bar direct-labelled with n and its share.

    ``_capture`` is a test hook: when given, it is filled with
    ``{"colors": [...]}`` — the per-bar colours in draw order — since they
    are otherwise only inspectable by parsing the saved image.
    """
    if fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax_bar = fig.subplots()
        _grid_behind(ax_bar)

        counts = compute.class_counts(fits)
        order, values = list(counts.keys()), list(counts.values())
        total = sum(values)
        colors = [theme.MOTILITY_COLORS[c] for c in order]
        if _capture is not None:
            _capture["colors"] = colors

        ax_bar.bar(order, values, color=colors, width=0.6, zorder=2)
        for i, n in enumerate(values):
            ax_bar.annotate(f"{n}\n{100 * n / max(total, 1):.0f}%", xy=(i, n),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=_FONT_ANNOT,
                            color="black")
        ax_bar.set_ylabel("# tracks", color="black")
        ax_bar.set_title(f"Motility classes (n={total})", color="black")
        ax_bar.grid(axis="x", visible=False)
        if max(values, default=0) > 0:
            ax_bar.set_ylim(0, max(values) * 1.18)  # headroom for the direct labels

        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_msd_regimes(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """MSD regime map: normalized curves over the three shaded alpha wedges.

    Each track's MSD is divided by its own first-lag value and the lag axis
    by the first lag, so every curve passes through (1, 1). On log-log axes
    a track's class then depends on slope alone, which is what makes the
    wedge boundaries ``t^0.4`` and ``t^1.2`` an honest partition — over *raw*
    MSD curves they would not be, since vertical position there encodes D,
    not alpha.

    Two corrections make the drawn slope match the fitted alpha. The
    localization floor ``4*sigma^2`` is subtracted, since the raw curve's
    slope is that of ``4*D*t^alpha + 4*sigma^2``, not of ``t^alpha``. And the
    anchor is the *fitted* value at the first lag, ``4*D*t1^alpha`` — not the
    measured first point, which for a constrained track is mostly
    localization noise once the offset is removed, so dividing by it would
    fling exactly those tracks into the wrong wedge. Tracks whose fit did not
    converge have no anchor and are omitted.
    """
    if curves.empty or fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        offset = 4.0 * units.sigma ** 2
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
            # Degenerate axes would draw wedges over a 1->1.001 span; say so instead.
            ax.set_xlabel("Lag time / first lag  (normalized)", color="black")
            ax.set_ylabel("MSD / 4·D·t₁^α", color="black")
            ax.set_title("MSD regime map — 0 tracks", color="black")
            ax.annotate("no tracks with a converged MSD fit", xy=(0.5, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        fontsize=_FONT_ANNOT, color="black")
            fig.tight_layout()
            _finish(fig, stem, formats, dpi)
            return

        # Shade the wedges first, anchored at (1, 1) where every curve starts.
        grid = np.logspace(0, np.log10(max(x_max, 1.001)), 200)
        lo_bound = grid ** ALPHA_CONSTRAINED_MAX
        hi_bound = grid ** ALPHA_DIRECTED_MIN
        # Clamp both limits to where the bulk of the curves live. Degenerate
        # fits (alpha large, D -> 0) give a near-zero anchor and send
        # MSD/anchor to 1e17, while short lags with no signal above the
        # localization floor sit near zero — either extreme alone flattens
        # all three wedges into a sliver.
        all_ys = np.concatenate([ys for _, ys, _ in normalized]) if normalized else None
        if all_ys is not None and np.any(np.isfinite(all_ys)):
            floor = max(float(np.nanpercentile(all_ys, 2)), y_hi * 1e-4)
            floor = min(floor, 0.6)
            ceil = max(float(np.nanpercentile(all_ys, 98)),
                       float(hi_bound[-1])) * 1.6
        else:
            floor, ceil = 0.6, max(y_hi, float(hi_bound[-1])) * 1.6
        ax.fill_between(grid, floor, lo_bound, color=theme.MOTILITY_COLORS["constrained"],
                        alpha=0.16, linewidth=0, zorder=0)
        ax.fill_between(grid, lo_bound, hi_bound, color=theme.MOTILITY_COLORS["diffusive"],
                        alpha=0.16, linewidth=0, zorder=0)
        ax.fill_between(grid, hi_bound, ceil, color=theme.MOTILITY_COLORS["directed"],
                        alpha=0.16, linewidth=0, zorder=0)
        for bound in (lo_bound, hi_bound):
            ax.plot(grid, bound, color="gray", linewidth=1.2, linestyle="--",
                    zorder=3)

        for xs, ys, cls in normalized:
            ax.plot(xs, ys, color=theme.MOTILITY_COLORS[cls], alpha=0.35,
                    linewidth=0.9, zorder=2)

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
                        fontsize=_FONT_ANNOT, color="black", zorder=4,
                        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

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
            Line2D([], [], color=theme.MOTILITY_COLORS[c], linewidth=2.0,
                   label=f"{c} ({drawn[c]})")
            for c in (*MOTILITY_CLASSES, "unclassified") if drawn.get(c)
        ]
        if handles:
            ax.legend(handles=handles, loc="upper left", labelcolor="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_d_vs_alpha(
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Joint distribution of D and alpha, with marginals and the class cuts."""
    if fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        # Explicit margins rather than tight_layout: the shared-axis
        # marginals are not tight_layout-compatible, and a fixed layout keeps
        # this figure the same size as every other one. top is lower than a
        # plain 0.92: the corner legend's three two-line entries grow up out
        # of their cell, and at print type sizes the first entry ran off the
        # top of the canvas.
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
            _grid_behind(a)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)

        valid = fits[np.isfinite(fits["alpha"]) & (fits["D"] > 0)]
        for cls in MOTILITY_CLASSES:
            sel = valid[valid["motility_class"] == cls]
            if not len(sel):
                continue
            ax.scatter(sel["alpha"], sel["D"], s=26, color=theme.MOTILITY_COLORS[cls],
                       edgecolor="white", linewidth=0.6, alpha=0.85, zorder=2,
                       label=f"{cls}\nn = {len(sel)}")
        if len(valid):
            ax.set_yscale("log")
            d_lo, d_hi = _d_axis_limits(valid["D"])
            a_lo, a_hi = _alpha_axis_limits(valid["alpha"])
            ax.set_ylim(d_lo, d_hi)
            ax.set_xlim(a_lo, a_hi)
            # Bin the marginal over the visible range, not the raw data
            # range — otherwise the outliers spend most of the bins off-screen.
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
                # 75% black rather than full black: these rules sit right
                # next to the legend and read as a competing mark at full
                # strength.
                a.axvline(cut, color="#404040", linestyle="--", linewidth=1.2,
                          zorder=3)
        # One label per alpha band rather than one per rule: the bands are
        # what the reader needs named, and three labels spread across the
        # axis cannot collide the way two labels beside adjacent rules did.
        # All sit on the bottom axis, running vertically so a narrow band
        # still fits.
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
                # Anchored just inside the band's left edge, not centred:
                # centred labels drift into whatever the cloud is doing
                # mid-band, while the left edge is a fixed landmark (the
                # axis, or the rule beside it).
                ax.annotate(_alpha_criterion(cls), xy=(lo, 0.02),
                            xycoords=("data", "axes fraction"),
                            xytext=(8, 0), textcoords="offset points",
                            fontsize=_FONT_ANNOT_SMALL, color="black",
                            rotation=90, va="bottom", ha="left", zorder=5,
                            # A halo regardless: these sit on the bottom
                            # axis, which for the constrained band is inside
                            # its cluster.
                            path_effects=[pe.withStroke(linewidth=5.0,
                                                        foreground="white")])

        ax.set_xlabel(theme.axis_label("alpha", units.calibration), color="black")
        ax.set_ylabel(theme.axis_label("diffusion", units.calibration), color="black")
        ax_top.set_ylabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_right.set_xlabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_top.set_title(f"D vs α — {len(valid)} tracks", color="black")
        if len(valid):
            # The corner cell, not over the data: D rises with alpha on a
            # calibrated run and falls on an uncalibrated one, so no
            # in-axes corner is reliably empty. The per-class count rides
            # here on a second line rather than inside the plot: placed by
            # alpha band it would follow the data spread, and on a run
            # where alpha runs out to 8 with everything packed below 2 the
            # band midpoints land in empty space. The corner is fixed.
            # Bottom-anchored so the taller block grows up into the blank
            # band beside the title instead of down onto the D histogram.
            handles, labels = ax.get_legend_handles_labels()
            ax_corner.legend(handles, labels, loc="lower center",
                             fontsize=_FONT_ANNOT_SMALL, labelcolor="black",
                             handletextpad=0.4, labelspacing=0.3, borderpad=0.1)
        _finish(fig, stem, formats, dpi)


def plot_length_hist(
    table: pd.DataFrame,
    units: Units,
    min_length: int,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Track-duration distribution of the analysed (already length-filtered) tracks.

    ``min_length`` is stated on the figure so the cut is legible without the
    config — in frames, since that is the unit it is given in, with the
    elapsed-time equivalent beside it when the run is calibrated.
    """
    if table.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        durations = _durations(table, units)
        ax.hist(durations, bins=30, color=theme.NEUTRAL, zorder=2)
        bits = [f"n = {len(durations):,} analysed tracks",
                f"length ≥ {int(min_length)} frames"]
        if units.is_physical:
            bits[-1] += f" ({(int(min_length) - 1) * units.delta:.3f} {units.time_unit})"
        _annotate_corner(ax, "   ·   ".join(bits))

        ax.set_xlabel(theme.axis_label("duration", units.calibration), color="black")
        ax.set_ylabel("# tracks", color="black")
        ax.set_title(
            "Track duration distribution" if units.is_physical
            else "Track length distribution", color="black")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_path_length_hist(
    table: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Distribution of the distance each track actually walked.

    This is the path length — the summed frame-to-frame steps — not the net
    displacement, and not the duration ``length_hist`` shows. At short frame
    intervals it is inflated by localization noise, since every jitter adds
    to the sum; the net displacement in ``directed_displacement`` is the
    transport quantity.
    """
    if table.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        path = table["total_displacement"].to_numpy(dtype=float) * units.length_scale
        ax.hist(path, bins=30, color=theme.NEUTRAL, edgecolor="black", zorder=2)
        _annotate_corner(ax, f"n = {len(path):,} analysed tracks")

        ax.set_xlabel(theme.axis_label("path_length", units.calibration), color="black")
        ax.set_ylabel("# tracks", color="black")
        ax.set_title("Path length distribution", color="black")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_speed_hist(
    table: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Histogram of per-track mean speed (path length over duration)."""
    if table.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        speed = table["mean_speed"].to_numpy(dtype=float) * units.length_scale / units.delta
        ax.hist(speed, bins=30, color=theme.NEUTRAL, edgecolor="black", zorder=2)

        ax.set_xlabel(f"Mean speed ({theme.unit('speed', units.calibration)})", color="black")
        ax.set_ylabel("# tracks", color="black")
        ax.set_title("Track mean-speed distribution", color="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_length_per_track(
    tracked: pd.DataFrame,
    units: Units,
    max_tracks: int,
    stem: Path,
    formats,
    dpi: int,
    *,
    _capture: dict | None = None,
) -> None:
    """One line per track: cumulative displacement vs elapsed time.

    ``cum_disp`` has no matching :func:`theme.axis_label` kind — it is a
    running sum over time, not the track's net (start-to-end) displacement —
    so its label keeps the source's noun, taking only the unit from
    :func:`theme.unit`.

    ``_capture`` is a test hook: when given, it is filled with
    ``{"n_lines": ...}`` — the number of track lines actually drawn — since
    ``max_tracks`` being honoured is otherwise only checkable by parsing the
    saved image.
    """
    if tracked.empty:
        return
    ids = _top_n_track_ids(tracked, max_tracks)
    if not ids:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        n_lines = 0
        for tid in ids:
            grp = tracked[tracked["particle"] == tid].sort_values("frame")
            if len(grp) < 2:
                continue
            coords = grp[["y", "x"]].to_numpy()
            steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(steps)]) * units.length_scale
            ax.plot(grp["frame"].to_numpy() * units.delta, cum,
                    color=theme.RAW_TRACE, alpha=0.5, linewidth=0.7)
            n_lines += 1
        if _capture is not None:
            _capture["n_lines"] = n_lines

        ax.set_xlabel(theme.axis_label("time", units.calibration), color="black")
        ax.set_ylabel(f"Cumulative displacement ({theme.unit('length', units.calibration)})",
                      color="black")
        ax.set_title(f"Per-track cumulative displacement (top {len(ids)})", color="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_speed_per_track(
    tracked: pd.DataFrame,
    units: Units,
    max_tracks: int,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """One line per track: frame-to-frame speed vs elapsed time.

    Speed at frame f is the distance from the previous frame to frame f (so
    the first frame of every track has no value and is skipped).
    """
    if tracked.empty:
        return
    ids = _top_n_track_ids(tracked, max_tracks)
    if not ids:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        for tid in ids:
            grp = tracked[tracked["particle"] == tid].sort_values("frame")
            if len(grp) < 2:
                continue
            coords = grp[["y", "x"]].to_numpy()
            steps = (np.linalg.norm(np.diff(coords, axis=0), axis=1)
                     * units.length_scale / units.delta)
            ax.plot(grp["frame"].to_numpy()[1:] * units.delta, steps,
                    color=theme.RAW_TRACE, alpha=0.5, linewidth=0.7)

        ax.set_xlabel(theme.axis_label("time", units.calibration), color="black")
        ax.set_ylabel(f"Instantaneous speed ({theme.unit('speed', units.calibration)})",
                      color="black")
        ax.set_title(f"Per-track speed (top {len(ids)})", color="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_directed_hist(
    directed: pd.DataFrame,
    units: Units,
    column: str,
    xlabel: str,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Distribution of one directed-track property (run length or velocity).

    ``directed`` is :func:`compute.directed_metrics`'s output — already
    restricted to directed tracks — and ``column`` is ``"_disp"`` (net
    displacement, the run length) or ``"_vel"`` (that displacement over the
    track's duration).
    """
    if directed.empty:
        return
    v = pd.to_numeric(directed[column], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        # Empty input writes nothing — the caller already skips this call
        # when `directed` is empty; this covers the column-specific case.
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

        lo, hi = _pct_limits(v)
        ax.hist(v, bins=np.linspace(lo, hi, 40),
                color=theme.MOTILITY_COLORS["directed"], zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_title(
            f"Directed tracks — n = {v.size}  (median {np.median(v):.3g})",
            color="black")

        ax.set_xlabel(xlabel, color="black")
        ax.set_ylabel("# tracks", color="black")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_directed_joint(
    directed: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Run length against velocity, directed tracks only, with marginals."""
    if directed.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              wspace=0.05, hspace=0.05,
                              left=0.17, right=0.97, top=0.90, bottom=0.15)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        for a in (ax, ax_top, ax_right):
            _grid_behind(a)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)

        ax.scatter(directed["_disp"], directed["_vel"], s=26,
                   color=theme.MOTILITY_COLORS["directed"], edgecolor="white",
                   linewidth=0.6, alpha=0.85, zorder=2)
        x_lo, x_hi = _pct_limits(directed["_disp"])
        y_lo, y_hi = _pct_limits(directed["_vel"])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax_top.hist(directed["_disp"], bins=np.linspace(x_lo, x_hi, 40),
                    color=theme.NEUTRAL, zorder=2)
        ax_right.hist(directed["_vel"], bins=np.linspace(y_lo, y_hi, 40),
                      orientation="horizontal", color=theme.NEUTRAL, zorder=2)
        n_off = int((
            (directed["_disp"] < x_lo) | (directed["_disp"] > x_hi)
            | (directed["_vel"] < y_lo) | (directed["_vel"] > y_hi)
        ).sum())
        if n_off:
            ax.annotate(f"{n_off} track(s) outside axis range",
                        xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=_FONT_ANNOT_SMALL, color="#52514e")
        ax_top.set_title(f"Directed tracks — n = {len(directed)}", color="black")

        ax.set_xlabel(f"Run length ({theme.unit('length', units.calibration)})", color="black")
        ax.set_ylabel(f"Velocity ({theme.unit('speed', units.calibration)})", color="black")
        ax_top.set_ylabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        ax_right.set_xlabel("# tracks", color="black", fontsize=_FONT_ANNOT)
        _finish(fig, stem, formats, dpi)


# ── MSD methods figures (after IJMS 25(16), 8660 — Fig. 1) ──────────


def _representative_track(fits: pd.DataFrame, cls: str) -> int | None:
    """Track whose α is the median of its class — robust to outliers."""
    sel = fits[(fits["motility_class"] == cls) & np.isfinite(fits["alpha"])]
    if not len(sel):
        return None
    order = sel["alpha"].sort_values()
    return int(sel.loc[order.index[len(order) // 2], "track_id"])


def _track_xy(tracked: pd.DataFrame, tid: int, units: Units) -> np.ndarray:
    grp = tracked[tracked["particle"] == tid].sort_values("frame")
    return grp[["x", "y"]].to_numpy(dtype=float) * units.length_scale


def _fit_curve(fits: pd.DataFrame, tid: int, lags: np.ndarray,
                units: Units) -> np.ndarray | None:
    """``4·D·t^α + 4σ²`` evaluated at ``lags`` for one track."""
    row = fits[fits["track_id"] == tid]
    if not len(row):
        return None
    d_fit = float(row["D"].iloc[0])
    a_fit = float(row["alpha"].iloc[0])
    if not (np.isfinite(d_fit) and np.isfinite(a_fit)):
        return None
    return 4.0 * d_fit * lags ** a_fit + 4.0 * units.sigma ** 2


def _draw_scale_bar(ax, xy: np.ndarray, units: Units) -> None:
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
    ax.annotate(f"{nice:g} {units.length_unit}", xy=(x0 + nice / 2, y0),
                xytext=(0, -3), textcoords="offset points", ha="center",
                va="top", fontsize=_FONT_ANNOT_SMALL, color="black")


def plot_msd_construction(
    tracked: pd.DataFrame,
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Panels (a) and (b): how an MSD curve is built from one trajectory.

    (a) the trajectory with the displacement chords for Δt = τ…5τ;
    (b) the resulting MSD ± SEM, first five lags colour-matched to (a).
    """
    if tracked.empty or curves.empty or fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(2 * FIG_W_PANEL, FIG_H))
        ax_traj, ax_msd = fig.subplots(1, 2)
        _grid_behind(ax_traj)
        _grid_behind(ax_msd)

        tid = (_representative_track(fits, "diffusive")
               or _representative_track(fits, "constrained")
               or _representative_track(fits, "directed"))
        if tid is None:
            for a, t in ((ax_traj, "Lag construction"), (ax_msd, "MSD ± SEM")):
                a.set_title(t, color="black")
                a.annotate("no tracks with a converged MSD fit", xy=(0.5, 0.5),
                           xycoords="axes fraction", ha="center", va="center",
                           fontsize=_FONT_ANNOT, color="black")
            fig.tight_layout()
            _finish(fig, stem, formats, dpi)
            return

        # ── (a) trajectory + one chord per lag ───────────────────────
        xy = _track_xy(tracked, tid, units)
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
        ax_traj.legend(handles=handles, frameon=False, fontsize=_FONT_ANNOT_SMALL,
                       loc="best", labelcolor="black", title="steps from start")
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.set_xlabel(f"x ({theme.unit('length', units.calibration)})", color="black")
        ax_traj.set_ylabel(f"y ({theme.unit('length', units.calibration)})", color="black")
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
        fit_y = _fit_curve(fits, tid, lags, units)
        if fit_y is not None:
            a_fit = float(fits.loc[fits["track_id"] == tid, "alpha"].iloc[0])
            ax_msd.plot(lags, fit_y, color="black", linestyle="--", linewidth=1.6,
                        zorder=4, label=f"fit  4·D·t$^α$ + 4σ²  (α={a_fit:.2f})")
        ax_msd.set_xlabel(f"Lag time τ ({theme.unit('time', units.calibration)})", color="black")
        ax_msd.set_ylabel(theme.axis_label("msd", units.calibration), color="black")
        ax_msd.set_title("Averaging equal-lag displacements", color="black")
        ax_msd.legend(frameon=False, fontsize=_FONT_ANNOT_SMALL, loc="upper left",
                      labelcolor="black")

        fig.tight_layout()
        _finish(fig, stem, formats, dpi)


def plot_msd_archetypes(
    tracked: pd.DataFrame,
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Panels (c)–(e): the characteristic MSD shape of each class.

    One representative track per class — MSD ± SEM with its fit, a dashed
    pure-Brownian ``4·D·τ`` reference so the departure from linear is visible,
    and the trajectory itself as an inset with a scale bar.
    """
    if tracked.empty or curves.empty or fits.empty:
        return
    panels = [
        ("Brownian", "diffusive"),
        ("Drifted", "directed"),
        ("Confined", "constrained"),
    ]
    with theme.style():
        fig = Figure(figsize=(FIG_W_ARCHETYPES, FIG_H))
        axes = fig.subplots(1, 3)
        for ax, (title, cls) in zip(axes, panels):
            _grid_behind(ax)
            ax.set_xlabel(f"Lag time τ ({theme.unit('time', units.calibration)})",
                          color="black")
            ax.set_ylabel(theme.axis_label("msd", units.calibration), color="black")
            tid = _representative_track(fits, cls)
            if tid is None:
                ax.set_title(f"{title} — {cls}", color="black")
                ax.annotate(f"no {cls} tracks", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center", va="center",
                            fontsize=_FONT_ANNOT, color="black")
                continue

            grp = curves[curves["track_id"] == tid].sort_values("lag")
            lags = grp["lag"].to_numpy(dtype=float)
            vals = grp["msd"].to_numpy(dtype=float)
            sems = np.nan_to_num(grp["sem"].to_numpy(dtype=float))
            ax.errorbar(lags, vals, yerr=sems, fmt="o", markersize=3.5,
                        color=theme.MOTILITY_COLORS[cls], ecolor="gray", elinewidth=1.0,
                        capsize=2, zorder=3)
            fit_y = _fit_curve(fits, tid, lags, units)
            a_fit = float(fits.loc[fits["track_id"] == tid, "alpha"].iloc[0])
            if fit_y is not None:
                ax.plot(lags, fit_y, color="black", linewidth=1.6, zorder=4)
            # Headroom for the inset, scaled to the data — not to the
            # Brownian reference, which for a plateauing track runs far
            # above every point.
            y_top = float(np.nanmax(vals + sems)) * 1.5 or 1.0
            # Pure-Brownian reference through the first point: MSD ∝ τ.
            if len(lags) and lags[0] > 0 and vals[0] > 0:
                ax.plot(lags, vals[0] * lags / lags[0], color="gray",
                        linestyle="--", linewidth=1.2, zorder=2,
                        label="Brownian  4·D·τ")
                ax.legend(frameon=False, fontsize=_FONT_ANNOT_SMALL, loc="lower right",
                          labelcolor="black")
            ax.set_ylim(0, y_top)
            # One point below the rc title size (theme.STYLE["axes.titlesize"]
            # == 15): this title also carries the track id and alpha.
            ax.set_title(f"{title} — {cls} (track {tid}, α={a_fit:.2f})",
                         color="black", fontsize=_FONT_ANNOT)

            # Trajectory inset, upper-left — reliably clear of a rising MSD curve.
            inset = ax.inset_axes((0.07, 0.56, 0.36, 0.40))
            xy = _track_xy(tracked, tid, units)
            inset.plot(xy[:, 0], xy[:, 1], color=theme.MOTILITY_COLORS[cls], linewidth=0.9)
            _draw_scale_bar(inset, xy, units)
            inset.set_aspect("equal", adjustable="datalim")
            inset.margins(x=0.12, y=0.22)  # room for the scale bar and its label
            inset.set_xticks([])
            inset.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                inset.spines[side].set_color("black")
            inset.set_facecolor("#f0f0f0")

        # Explicit margins, not tight_layout: tight_layout sizes each panel to
        # its own tick-label width, which leaves the three panels slightly
        # unequal.
        fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.16,
                            wspace=0.22)
        _finish(fig, stem, formats, dpi)


def plot_msd_shape_comparison(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    units: Units,
    stem: Path,
    formats,
    dpi: int,
) -> None:
    """Panel (f): the three class-mean MSD *shapes*, on linear axes.

    Both axes are normalized to the common maximum lag, so every curve runs
    from the origin to (1, 1) and only its curvature differs — concave for
    subdiffusive, straight for Brownian, convex for superdiffusive. Without
    that, a directed population whose MSD is orders of magnitude larger than
    a confined one flattens the other curves onto the axis and the comparison
    the panel exists to make is lost. Absolute magnitudes live in
    ``msd_combined.png``; this panel is about shape alone.
    """
    if curves.empty or fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        _grid_behind(ax)

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
            ax.plot(xs, ys, color=theme.MOTILITY_COLORS[cls], linewidth=2.2, zorder=3)
            drew = True
            # Direct labels beat a legend here: three long labels won't fit
            # on one row, and the curves are well separated at mid-range.
            i = int(np.argmin(np.abs(xs - 0.55)))
            ax.annotate(
                f"{cls}  (n={int((fits['motility_class'] == cls).sum())}, "
                f"median α={alphas.median():.2f})",
                xy=(xs[i], ys[i]), xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=_FONT_ANNOT, color="black", zorder=5,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
            )

        # The normalization belongs on the figure, but at this type size it
        # only fits in the title — an axis label long enough to carry it runs
        # off both ends of a column-width figure.
        ax.set_xlabel("Lag time τ / τ$_{max}$", color="black")
        ax.set_ylabel("Mean MSD, normalized to τ$_{max}$", color="black")
        ax.set_title(
            f"Class-mean MSD shapes — shape only, normalized at "
            f"τ$_{{max}}$ = {common_max:.3g} {units.time_unit}",
            color="black",
        )
        if not drew:
            ax.annotate("no class had enough tracks to average", xy=(0.5, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        fontsize=_FONT_ANNOT, color="black")
        fig.tight_layout()
        _finish(fig, stem, formats, dpi)
