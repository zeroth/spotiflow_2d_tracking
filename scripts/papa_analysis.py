"""PAPA microscopy analysis pipeline.

Runs Spotiflow spot detection on every frame of a TIFF stack, saves
segmentation masks, a blob CSV, and combined particles-per-frame / per-cycle
ratio graphs with cycle-group separators.

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
)
from napari_spotiflow_tracking._preprocessing import _get_device
from napari_spotiflow_tracking._segmentation import (
    create_spot_mask,
    detect_spots,
    load_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="PAPA microscopy analysis pipeline.")
    p.add_argument("input_tiff", type=str, help="Path to input TIFF stack.")
    p.add_argument("group1_size", type=int, help="Number of frames in group 1.")
    p.add_argument("group2_size", type=int, help="Number of frames in group 2.")
    p.add_argument("--skip_frames", type=int, default=1,
                   help="Number of skip frames between groups (default: 1).")
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
    return p.parse_args()


def make_combined_graph(particles_per_frame, cycle_info, max_time,
                        group1_size, group2_size, skip_frames,
                        label_group1, label_group2, save_path):
    """Create and save the combined particles-per-frame graph.

    Each group's frame range is shaded as a background band (group 1 blue,
    group 2 orange); a dotted separator marks each cycle boundary.
    """
    group_colors = ["tab:blue", "tab:orange"]
    group_labels = [label_group1, label_group2]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(particles_per_frame["frame"], particles_per_frame["n_particles"],
            color="black", linewidth=1.0, zorder=2)

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

    ax.set_title(
        f"All Cycles Combined — Particles per Frame  "
        f"(G1={group1_size}f + skip{skip_frames} + G2={group2_size}f + skip{skip_frames})"
    )
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Number of Particles")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined graph to {save_path}")


def make_ratio_graph(particles_per_frame, cycle_info, label_group1,
                     label_group2, save_path):
    """Save three per-cycle figures derived from ``save_path``'s stem:

    ``_total``   grouped count bars with the G2/G1 ratio overlaid as a line
                 on a twin right-hand axis
    ``_g1_g2``   G1/G2 ratio bars
    ``_g2_g1``   G2/G1 ratio bars
    """
    summary = compute_per_cycle_summary(
        particles_per_frame, cycle_info, label_group1, label_group2,
    )
    cycles = summary["cycle"].tolist()
    g1_counts = summary[f"{label_group1}_count"].tolist()
    g2_counts = summary[f"{label_group2}_count"].tolist()
    ratios_g1_g2 = summary[f"{label_group1}_over_{label_group2}"].tolist()
    ratios_g2_g1 = summary[f"{label_group2}_over_{label_group1}"].tolist()

    x = np.arange(len(cycles))
    bar_width = 0.35

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

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
                              xytext=(0, 8), ha="center", fontsize=8,
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
               fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_total.png", dpi=150)
    plt.close(fig)

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 4) )

    # Standalone G1/G2 ratio bars
    ax2.bar(x, ratios_g1_g2, 0.5, color="tab:green")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel(f"Ratio ({label_group1} / {label_group2})")
    ax2.set_xlabel("Cycle")
    ax2.set_title(f"Particle Count Ratio ({label_group1} / {label_group2}) per Cycle")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cycles])
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_g1_g2.png", dpi=150)
    plt.close(fig)


    fig, ax3 = plt.subplots(1, 1, figsize=(10, 4))
    # Standalone G2/G1 ratio bars
    ax3.bar(x, ratios_g2_g1, 0.5, color="tab:red")
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_ylabel(f"Ratio ({label_group2} / {label_group1})")
    ax3.set_xlabel("Cycle")
    ax3.set_title(f"Particle Count Ratio ({label_group2} / {label_group1}) per Cycle")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in cycles])
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{Path(save_path).with_suffix("")}_g2_g1.png", dpi=150)
    plt.close(fig)
    print(f"Saved ratio graph to {save_path}")


def main():
    args = parse_args()

    # Derive output paths from input filename
    stem = Path(args.input_tiff).with_suffix("")
    seg_path = f"{stem}_seg.tif"
    blob_csv_path = f"{stem}_blob.csv"
    graph_path = f"{stem}_combine_graph.png"
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
    make_ratio_graph(particles_per_frame, cycle_info,
                     args.label_group1, args.label_group2, ratio_path)

    # Print summary
    n_cycles = max(c["cycle"] for c in cycle_info) if cycle_info else 0
    print(f"\nTotal cycles: {n_cycles}")
    print(f"Pattern: G1={args.group1_size}f + skip{args.skip_frames} "
          f"+ G2={args.group2_size}f + skip{args.skip_frames}")
    for info in cycle_info:
        print(f"  Cycle {info['cycle']}, Group {info['group']}: "
              f"frames {info['start_frame']}-{info['end_frame']}")


if __name__ == "__main__":
    main()
