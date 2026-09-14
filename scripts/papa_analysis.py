"""PAPA microscopy analysis pipeline.

Runs Spotiflow spot detection on every frame of a TIFF stack, saves
segmentation masks, a blob CSV, and a set of graphs with cycle-group
separators. Derived from the input stem, the graphs are:

    _combine_graph.png             particles per frame
    _combine_graph_norm.png        the same, rescaled 0-1
    _combine_ratio_graph.png       particles per frame + per-cycle G2/G1 ratio
    _combine_ratio_graph_norm.png  the same, rescaled 0-1
    _ratio_total.png               per-cycle counts with the ratio overlaid
    _ratio_g1_g2.png               per-cycle G1/G2 ratio bars
    _ratio_g2_g1.png               per-cycle G2/G1 ratio bars
    _info.txt                      how this run was configured

Figure titles name only what the figure shows. The acquisition pattern, the
ratio window, the normalisation range and the per-cycle frame ranges all go in
``_info.txt``, so a figure can be traced back without being cluttered.

The ``_norm`` pair rescales the plotted trace to 0-1 as
``(n - min) / (max - min)``, taking the min and max over the cycle frames — the
excluded lead-in does not set the scale. Ratios are computed from the raw
counts throughout, so each ``_norm`` figure carries exactly the ratio line of
its unnormalised twin.

Reuses the napari-spotiflow-tracking plugin's detection and PAPA-analysis
functions directly (no napari / Qt / UI). Install the plugin into the same
environment first (e.g. ``pip install -e .``).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from napari_spotiflow_tracking._papa_analysis import (
    build_cycle_info,
    compute_per_cycle_summary,
    lead_in_span,
    normalize_min_max,
    ratio_window_length,
)
from napari_spotiflow_tracking._preprocessing import _get_device
from napari_spotiflow_tracking._segmentation import (
    create_spot_mask,
    detect_spots,
    load_model,
)

# ── figure style ────────────────────────────────────────────────────
# Matches bulk_track_analysis.py so the PNGs from both scripts drop into the
# same manuscript layout without rescaling: one height for every figure, one
# width for every single-panel figure.
FIG_H = 5.0
FIG_W = 7.0
FIG_W_PANEL = 4.5
# The combined graph is a time series thousands of frames long — at FIG_W the
# per-cycle peaks merge into each other, so it gets the width of a three-panel
# figure. Everything else here is single-panel.
FIG_W_TRACE = FIG_W + 2 * FIG_W_PANEL
# Type is sized for a figure printed at column width, so values stay legible
# after downscaling.
FONT_TICK = 15
FONT_LABEL = 17
FONT_TITLE = 13
FONT_ANNOT = 12          # in-plot annotations and legends
FONT_ANNOT_SMALL = 10    # the same, where a panel is too tight for FONT_ANNOT
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
# Figure sizes are in inches, so this alone sets the pixel dimensions.
FIG_DPI = 300


def parse_args():
    p = argparse.ArgumentParser(description="PAPA microscopy analysis pipeline.")
    p.add_argument("input_tiff", type=str, help="Path to input TIFF stack.")
    p.add_argument("group1_size", type=int, help="Number of frames in group 1.")
    p.add_argument("group2_size", type=int, help="Number of frames in group 2.")
    p.add_argument("--skip_frames", type=int, default=1,
                   help="Number of skip frames between groups (default: 1).")
    p.add_argument("--ratio_frame_fraction", type=float, default=1.0,
                   help="Fraction (0-1] of each group's frames used for the "
                        "ratio, taken from the start of the block. Affects the "
                        "ratio only; the trace and the total counts always use "
                        "every frame (default: 1.0, the whole block).")
    p.add_argument("--spotiflow_model", type=str, default="general",
                   help="Spotiflow pretrained model name (default: general).")
    p.add_argument("--prob_thresh", type=float, default=None,
                   help="Spotiflow probability threshold (default: model default).")
    p.add_argument("--min_distance", type=int, default=1,
                   help="Minimum distance between spots (default: 1).")
    p.add_argument("--spot_radius", type=int, default=1,
                   help="Radius for spot masks in pixels (default: 1).")
    p.add_argument("--label_group1", type=str, default="DR",
                   help="Legend label for group 1 separator (default: DR).")
    p.add_argument("--label_group2", type=str, default="PAPA",
                   help="Legend label for group 2 separator (default: PAPA).")
    p.add_argument("--raw", action="store_true", default=False,
                   help="Crop right half of image before analysis and save as _crop.tif.")
    args = p.parse_args()

    # Validate the cycle-pattern arguments up front: build_cycle_info would
    # reject the same bad input, but only after model loading, per-frame
    # detection, and mask/blob-CSV writing have already run.
    if args.group1_size < 1:
        p.error("group1_size must be >= 1")
    if args.group2_size < 1:
        p.error("group2_size must be >= 1")
    if args.skip_frames < 0:
        p.error("skip_frames must be >= 0")
    # Zero is rejected rather than clamped: a window of no frames would make
    # every ratio NaN, which reads as a bug rather than as a setting.
    if not 0 < args.ratio_frame_fraction <= 1:
        p.error("ratio_frame_fraction must be > 0 and <= 1")

    return args


# Titles stay to the point: the acquisition pattern, the ratio window and the
# normalisation range are recorded once in the run's info file instead of being
# repeated across every figure.
NORM_YLABEL = "Particles (normalised 0-1)"


def _normalize_for_plot(particles_per_frame, cycle_info):
    """Min-max the trace to 0-1, scaled over the cycle frames.

    The lead-in is excluded from the analysis, so it does not set the scale
    either. It is still drawn, riding along on the cycles' scale — which puts
    it outside 0-1 if it ever runs higher than the cycles do.

    Returns ``(trace, (min, max))``, or ``(None, (min, max))`` when the trace is
    flat or empty and there is no range to scale against.
    """
    in_cycles = particles_per_frame
    if cycle_info:
        in_cycles = pd.concat([
            particles_per_frame[
                (particles_per_frame["frame"] >= info["start_frame"])
                & (particles_per_frame["frame"] <= info["end_frame"])
            ]
            for info in cycle_info
        ])

    return normalize_min_max(particles_per_frame, range_frames=in_cycles)


def write_info_file(save_path, args, max_time, cycle_info,
                    particles_per_frame):
    """Record everything the figures no longer carry in their titles.

    The graphs are titled for reading; the acquisition pattern, the ratio
    window, the normalisation range and the per-cycle frame ranges are written
    here once so a figure can always be traced back to how it was made.
    """
    g1, g2 = args.group1_size, args.group2_size
    frac = args.ratio_frame_fraction
    n_cycles = max((c["cycle"] for c in cycle_info), default=0)
    lead_in = lead_in_span(max_time, g1)
    _, (norm_lo, norm_hi) = _normalize_for_plot(particles_per_frame, cycle_info)

    lines = [
        "PAPA analysis — run info",
        "=" * 60,
        "",
        f"Input                 : {args.input_tiff}",
        f"Frames                : {max_time}",
        "",
        "Detection",
        f"  Spotiflow model     : {args.spotiflow_model}",
        f"  prob_thresh         : "
        f"{'model default' if args.prob_thresh is None else args.prob_thresh}",
        f"  min_distance        : {args.min_distance}",
        f"  spot radius (mask)  : {args.spot_radius}",
        "",
        "Cycle pattern",
        f"  {args.label_group1} (group 1)        : {g1} frames",
        f"  {args.label_group2} (group 2)      : {g2} frames",
        f"  skip between blocks : {args.skip_frames} frame(s)",
        f"  lead-in (excluded)  : "
        + (f"frames {lead_in[0]}-{lead_in[1]}" if lead_in else "none"),
        f"  complete cycles     : {n_cycles}",
        "",
        "Ratio",
        f"  frame fraction      : {frac:g}",
        f"  counted per block   : first "
        f"{ratio_window_length(g1, frac)} of {g1} ({args.label_group1}), "
        f"first {ratio_window_length(g2, frac)} of {g2} ({args.label_group2})",
        "  note                : ratios use the raw counts, so the normalised",
        "                        figures carry the same ratio line",
        "",
        "Normalisation (the _norm figures)",
        "  method              : (n - min) / (max - min)",
        "  range taken over    : the cycle frames; the lead-in is drawn but",
        "                        does not set the scale",
        f"  range               : {norm_lo:g} - {norm_hi:g} particles",
        "",
        "Cycles",
    ]
    if cycle_info:
        for info in cycle_info:
            label = args.label_group1 if info["group"] == 1 else args.label_group2
            lines.append(f"  cycle {info['cycle']:<3} {label:<6} "
                         f"frames {info['start_frame']}-{info['end_frame']}")
    else:
        frames_per_cycle = g1 + args.skip_frames + g1 + args.skip_frames + g2
        lines.append(f"  none — {max_time} frames is short of the "
                     f"{frames_per_cycle} needed for the lead-in plus one cycle")

    stem = Path(args.input_tiff).with_suffix("").name
    lines += [
        "",
        "Figures",
        f"  {stem}_combine_graph.png",
        "      Particles per frame.",
        f"  {stem}_combine_graph_norm.png",
        "      The same, rescaled 0-1.",
        f"  {stem}_combine_ratio_graph.png",
        f"      Particles per frame with the per-cycle "
        f"{args.label_group2}/{args.label_group1} ratio overlaid.",
        f"  {stem}_combine_ratio_graph_norm.png",
        "      The same, rescaled 0-1.",
        f"  {stem}_ratio_total.png",
        "      Per-cycle counts with the ratio overlaid.",
        f"  {stem}_ratio_g1_g2.png",
        f"      Per-cycle {args.label_group1}/{args.label_group2} ratio bars.",
        f"  {stem}_ratio_g2_g1.png",
        f"      Per-cycle {args.label_group2}/{args.label_group1} ratio bars.",
        "",
    ]

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved run info to {save_path}")


def _draw_trace_and_bands(ax, particles_per_frame, cycle_info, max_time,
                          group1_size, label_group1, label_group2):
    """Draw the particle trace over the shaded lead-in / group / cycle bands.

    Shared by the plain combined graph and the ratio-overlay variant, so the
    two always shade the acquisition identically. Sets only the x limits — the
    caller owns the title, labels, legend and y axis.
    """
    group_colors = ["tab:blue", "tab:orange"]
    group_labels = [label_group1, label_group2]

    ax.plot(particles_per_frame["frame"], particles_per_frame["n_particles"],
            color="black", linewidth=1.0, zorder=2)

    # Excluded lead-in: readout before the first pulse, in no cycle. Shaded so
    # it is visibly present but visibly not part of the analysis.
    lead_in = lead_in_span(max_time, group1_size)
    if lead_in is not None:
        ax.axvspan(lead_in[0] - 0.5, lead_in[1] + 0.5, color="gray",
                   alpha=0.18, linewidth=0, label="lead-in (excluded)",
                   zorder=0)

    # Background band per group. Bands span whole frames, hence the ±0.5.
    labeled = set()
    for info in cycle_info:
        group = info["group"]
        label = None if group in labeled else group_labels[(group - 1) % 2]
        labeled.add(group)
        ax.axvspan(info["start_frame"] - 0.5, info["end_frame"] + 0.5,
                   color=group_colors[(group - 1) % 2], alpha=0.12,
                   linewidth=0, label=label, zorder=0)

    # Dotted separator at each cycle boundary (end of the cycle's last group).
    last_cycle = max((i["cycle"] for i in cycle_info), default=0)
    for info in cycle_info:
        is_cycle_end = not any(
            o["cycle"] == info["cycle"] and o["start_frame"] > info["start_frame"]
            for o in cycle_info
        )
        if is_cycle_end and info["cycle"] < last_cycle:
            ax.axvline(x=info["end_frame"] + 0.5, color="gray",
                       linestyle=":", alpha=0.8, linewidth=1.0, zorder=1)

    ax.set_xlim(0.5, max_time + 0.5)


def _cycle_spans(cycle_info):
    """Map each cycle number to its (first frame, last frame) across groups."""
    spans = {}
    for info in cycle_info:
        lo, hi = spans.get(info["cycle"], (info["start_frame"], info["end_frame"]))
        spans[info["cycle"]] = (min(lo, info["start_frame"]),
                                max(hi, info["end_frame"]))
    return spans


def make_combined_graph(particles_per_frame, cycle_info, max_time,
                        group1_size, group2_size, skip_frames,
                        label_group1, label_group2, save_path,
                        normalize=False):
    """Create and save the combined particles-per-frame graph.

    The excluded lead-in block is shaded grey; each cycle's group ranges are
    shaded as background bands (group 1 blue, group 2 orange) and a dotted
    separator marks each cycle boundary. The particle trace covers the whole
    stack, so lead-in data stays visible even though it belongs to no cycle.
    """
    trace = particles_per_frame
    if normalize:
        trace, _ = _normalize_for_plot(particles_per_frame, cycle_info)
        if trace is None:
            print("Flat or empty trace; skipping normalised combined graph.")
            return

    fig, ax = plt.subplots(figsize=(FIG_W_TRACE, FIG_H))
    _draw_trace_and_bands(ax, trace, cycle_info, max_time,
                          group1_size, label_group1, label_group2)

    ax.set_title("Particles per frame")
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel(NORM_YLABEL if normalize else "Number of Particles")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=FONT_ANNOT_SMALL)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved combined graph to {save_path}")


def make_combined_ratio_graph(particles_per_frame, cycle_info, max_time,
                              group1_size, group2_size, skip_frames,
                              label_group1, label_group2, save_path,
                              ratio_frame_fraction=1.0, normalize=False):
    """The combined graph with the per-cycle G2/G1 ratio line drawn over it.

    Same trace and shading as :func:`make_combined_graph`, plus the ratio from
    the ``_total`` chart on a twin right-hand axis. Each ratio is one number
    per cycle, so its marker sits at the centre frame of that cycle — putting
    the summary statistic in the same time axis as the data it summarises.
    """
    summary = compute_per_cycle_summary(
        particles_per_frame, cycle_info, label_group1, label_group2,
        frame_fraction=ratio_frame_fraction,
    )
    if summary.empty:
        print("No complete cycles in range; skipping combined ratio graph.")
        return

    # `summary` above comes from the raw counts, so normalising rescales the
    # plotted trace only and the ratio line is identical either way. It could
    # not be recovered from the normalised trace: subtracting the minimum
    # breaks the proportionality a ratio depends on.
    trace = particles_per_frame
    if normalize:
        trace, _ = _normalize_for_plot(particles_per_frame, cycle_info)
        if trace is None:
            print("Flat or empty trace; "
                  "skipping normalised combined ratio graph.")
            return

    fig, ax = plt.subplots(figsize=(FIG_W_TRACE, FIG_H))
    _draw_trace_and_bands(ax, trace, cycle_info, max_time,
                          group1_size, label_group1, label_group2)

    ax.set_title(
        f"Particles per frame and {label_group2}/{label_group1} ratio"
    )
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel(NORM_YLABEL if normalize else "Number of Particles")
    ax.grid(alpha=0.3)

    # Per-cycle ratio, anchored at each cycle's centre frame.
    spans = _cycle_spans(cycle_info)
    ratios = summary[f"{label_group2}_over_{label_group1}"].tolist()
    centres = [(spans[c][0] + spans[c][1]) / 2 for c in summary["cycle"]]

    ratio_color = "tab:green"
    ax_ratio = ax.twinx()
    ax_ratio.plot(centres, ratios, color=ratio_color, marker="o",
                  linewidth=2, markersize=6, zorder=3,
                  label=f"{label_group2} / {label_group1} ratio")
    for xi, r in zip(centres, ratios):
        if np.isfinite(r):
            # White backing box: the label lands on a shaded band, where
            # green-on-blue/orange is barely legible.
            ax_ratio.annotate(f"{r:.2f}", (float(xi), float(r)),
                              textcoords="offset points",
                              xytext=(0, 8), ha="center",
                              fontsize=FONT_ANNOT_SMALL,
                              color=ratio_color, zorder=4,
                              bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                        ec="none", alpha=0.75))
    ax_ratio.set_ylabel(f"{label_group2} / {label_group1} ratio",
                        color=ratio_color)
    ax_ratio.tick_params(axis="y", colors=ratio_color)
    finite_ratios = [r for r in ratios if np.isfinite(r)]
    if finite_ratios:
        # Headroom above the highest point so its label clears the frame.
        ax_ratio.set_ylim(0, max(finite_ratios) * 1.4)
    ax_ratio.grid(False)

    bands, band_labels = ax.get_legend_handles_labels()
    line, line_label = ax_ratio.get_legend_handles_labels()
    ax.legend(bands + line, band_labels + line_label, loc="upper right",
              fontsize=FONT_ANNOT_SMALL)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved combined ratio graph to {save_path}")


def make_ratio_graph(particles_per_frame, cycle_info, label_group1,
                     label_group2, save_path, ratio_frame_fraction=1.0):
    """Save three per-cycle figures derived from ``save_path``'s stem:

    ``_total``   grouped count bars with the G2/G1 ratio overlaid as a line
                 on a twin right-hand axis
    ``_g1_g2``   G1/G2 ratio bars
    ``_g2_g1``   G2/G1 ratio bars
    """
    summary = compute_per_cycle_summary(
        particles_per_frame, cycle_info, label_group1, label_group2,
        frame_fraction=ratio_frame_fraction,
    )
    if summary.empty:
        print("No complete cycles in range; skipping ratio graphs.")
        return

    cycles = summary["cycle"].tolist()
    g1_counts = summary[f"{label_group1}_count"].tolist()
    g2_counts = summary[f"{label_group2}_count"].tolist()
    ratios_g1_g2 = summary[f"{label_group1}_over_{label_group2}"].tolist()
    ratios_g2_g1 = summary[f"{label_group2}_over_{label_group1}"].tolist()

    x = np.arange(len(cycles))
    bar_width = 0.35

    fig, ax1 = plt.subplots(1, 1, figsize=(FIG_W + FIG_W_PANEL, FIG_H))

    # Grouped bar chart of total counts, with the G2/G1 ratio overlaid as a
    # line on a twin right-hand axis.
    ax1.bar(x - bar_width / 2, g1_counts, bar_width, color="tab:blue",
            label=f"{label_group1} (G1)")
    ax1.bar(x + bar_width / 2, g2_counts, bar_width, color="tab:orange",
            label=f"{label_group2} (G2)")
    ax1.set_ylabel("Total Particle Count")
    ax1.set_xlabel("Cycle")
    ax1.set_title(
        f"Per-cycle counts and {label_group2}/{label_group1} ratio"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in cycles])
    ax1.grid(axis="y", alpha=0.3)

    ratio_color = "tab:green"
    ax_ratio = ax1.twinx()
    ax_ratio.plot(x, ratios_g2_g1, color=ratio_color, marker="o",
                  linewidth=2, markersize=6, zorder=3,
                  label=f"{label_group2} / {label_group1} ratio")
    for xi, r in zip(x, ratios_g2_g1):
        if np.isfinite(r):
            # White backing box: labels can land on top of a bar, where
            # green-on-blue/orange is barely legible.
            ax_ratio.annotate(f"{r:.2f}", (float(xi), float(r)),
                              textcoords="offset points",
                              xytext=(0, 8), ha="center",
                              fontsize=FONT_ANNOT_SMALL,
                              color=ratio_color, zorder=4,
                              bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                        ec="none", alpha=0.75))
    ax_ratio.set_ylabel(f"{label_group2} / {label_group1} ratio",
                        color=ratio_color)
    ax_ratio.tick_params(axis="y", colors=ratio_color)
    finite_ratios = [r for r in ratios_g2_g1 if np.isfinite(r)]
    if finite_ratios:
        # Headroom above the highest point so its label clears the frame.
        ax_ratio.set_ylim(0, max(finite_ratios) * 1.4)
    ax_ratio.grid(False)

    bars, bar_labels = ax1.get_legend_handles_labels()
    line, line_label = ax_ratio.get_legend_handles_labels()
    ax1.legend(bars + line, bar_labels + line_label, loc="upper right",
               fontsize=FONT_ANNOT_SMALL)
    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_total.png", dpi=FIG_DPI,
                facecolor="white")
    plt.close(fig)

    fig, ax2 = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))

    # Standalone G1/G2 ratio bars
    ax2.bar(x, ratios_g1_g2, 0.5, color="tab:green")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel(f"Ratio ({label_group1} / {label_group2})")
    ax2.set_xlabel("Cycle")
    ax2.set_title(f"{label_group1} / {label_group2} ratio per cycle")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cycles])
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_g1_g2.png", dpi=FIG_DPI,
                facecolor="white")
    plt.close(fig)


    fig, ax3 = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    # Standalone G2/G1 ratio bars
    ax3.bar(x, ratios_g2_g1, 0.5, color="tab:red")
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_ylabel(f"Ratio ({label_group2} / {label_group1})")
    ax3.set_xlabel("Cycle")
    ax3.set_title(f"{label_group2} / {label_group1} ratio per cycle")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in cycles])
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_g2_g1.png", dpi=FIG_DPI,
                facecolor="white")
    plt.close(fig)
    print(f"Saved ratio graph to {save_path}")


def main():
    args = parse_args()

    # Derive output paths from input filename
    stem = Path(args.input_tiff).with_suffix("")
    seg_path = f"{stem}_seg.tif"
    blob_csv_path = f"{stem}_blob.csv"
    graph_path = f"{stem}_combine_graph.png"
    graph_norm_path = f"{stem}_combine_graph_norm.png"
    combine_ratio_path = f"{stem}_combine_ratio_graph.png"
    combine_ratio_norm_path = f"{stem}_combine_ratio_graph_norm.png"
    info_path = f"{stem}_info.txt"
    ratio_path = f"{stem}_ratio.png"

    # Load model (GPU when available — mirrors Spotiflow's old auto-device load)
    device = _get_device(use_gpu=True)
    print(f"Loading Spotiflow model '{args.spotiflow_model}' on {device}...")
    model = load_model(args.spotiflow_model, device=device)
    print("Model loaded.")

    # Load image stack
    image = tifffile.imread(args.input_tiff)
    print(f"Loaded {args.input_tiff}  shape={image.shape}  dtype={image.dtype}")

    if image.ndim == 2:
        image = image[np.newaxis, ...]

    # --raw: crop right half, save uncompressed, then memmap the cropped file
    if args.raw:
        mid_x = image.shape[-1] // 2
        crop_path = f"{stem}_crop.tif"
        cropped = image[..., mid_x:]
        tifffile.imwrite(crop_path, cropped)
        del image, cropped
        image = tifffile.memmap(crop_path, mode="r")
        print(f"Saved cropped right half to {crop_path}  shape={image.shape}")

    n_frames = image.shape[0]

    # Allocate output mask stack (memory-mapped for large files)
    use_bigtiff = image.nbytes > 2**31
    masks = tifffile.memmap(seg_path, shape=image.shape, dtype=np.uint8,
                            bigtiff=use_bigtiff)

    # Process each frame
    all_blobs = []
    for i in tqdm(range(n_frames), desc="Segmenting"):
        points, details = detect_spots(
            image[i], model,
            prob_thresh=args.prob_thresh,
            min_distance=args.min_distance,
        )
        mask = create_spot_mask(image[i].shape, points, radius=args.spot_radius)
        masks[i] = mask

        if len(points) > 0:
            # `details` is Spotiflow's predict result; `.intens` exists at
            # runtime but detect_spots types it loosely as ``object``.
            intensities = np.asarray(details.intens).ravel()  # type: ignore[attr-defined]
            frame_df = pd.DataFrame({
                "y": points[:, 0],
                "x": points[:, 1],
                "intensity": intensities,
                "time": i + 1,
            })
            all_blobs.append(frame_df)

    masks.flush()
    del masks
    print(f"Saved segmentation to {seg_path}")

    # Save blob CSV
    if all_blobs:
        df_blobs = pd.concat(all_blobs, ignore_index=True)
        df_blobs.sort_values("time", inplace=True)
        df_blobs.to_csv(blob_csv_path, index=False)
        print(f"Saved blobs to {blob_csv_path}  ({len(df_blobs)} spots)")
    else:
        df_blobs = pd.DataFrame(columns=["y", "x", "intensity", "time"])
        df_blobs.to_csv(blob_csv_path, index=False)
        print("No spots detected in any frame.")

    # Particles per frame — 1-indexed `frame` column, matching the plugin's
    # PAPA-analysis convention (compute_per_cycle_summary expects `frame`).
    particles_per_frame = (
        df_blobs.groupby("time").size().reset_index(name="n_particles")
        .rename(columns={"time": "frame"})
    )

    # Build cycle info and save graphs
    max_time = n_frames
    cycle_info = build_cycle_info(max_time, args.group1_size, args.group2_size,
                                  args.skip_frames)
    make_combined_graph(particles_per_frame, cycle_info, max_time,
                        args.group1_size, args.group2_size, args.skip_frames,
                        args.label_group1, args.label_group2, graph_path)
    make_combined_graph(particles_per_frame, cycle_info, max_time,
                        args.group1_size, args.group2_size, args.skip_frames,
                        args.label_group1, args.label_group2, graph_norm_path,
                        normalize=True)
    make_combined_ratio_graph(particles_per_frame, cycle_info, max_time,
                              args.group1_size, args.group2_size,
                              args.skip_frames, args.label_group1,
                              args.label_group2, combine_ratio_path,
                              ratio_frame_fraction=args.ratio_frame_fraction)
    make_combined_ratio_graph(particles_per_frame, cycle_info, max_time,
                              args.group1_size, args.group2_size,
                              args.skip_frames, args.label_group1,
                              args.label_group2, combine_ratio_norm_path,
                              ratio_frame_fraction=args.ratio_frame_fraction,
                              normalize=True)
    make_ratio_graph(particles_per_frame, cycle_info,
                     args.label_group1, args.label_group2, ratio_path,
                     ratio_frame_fraction=args.ratio_frame_fraction)
    write_info_file(info_path, args, max_time, cycle_info, particles_per_frame)

    # Print summary
    n_cycles = max((c["cycle"] for c in cycle_info), default=0)
    frames_per_cycle = (
        args.group1_size + args.skip_frames + args.group1_size
        + args.skip_frames + args.group2_size
    )
    print(f"\nTotal complete cycles: {n_cycles}")
    print(f"Pattern: lead-in {args.group1_size}f (excluded), then "
          f"{args.label_group1}={args.group1_size}f + skip{args.skip_frames} "
          f"+ {args.label_group2}={args.group2_size}f + skip{args.skip_frames}")
    lead_in = lead_in_span(max_time, args.group1_size)
    if lead_in is not None:
        print(f"  Lead-in (excluded): frames {lead_in[0]}-{lead_in[1]}")
    for info in cycle_info:
        print(f"  Cycle {info['cycle']}, Group {info['group']}: "
              f"frames {info['start_frame']}-{info['end_frame']}")
    if n_cycles == 0:
        print(f"  No complete cycles fit in {max_time} frames "
              f"(need at least {frames_per_cycle}).")


if __name__ == "__main__":
    main()
