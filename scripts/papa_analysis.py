"""PAPA microscopy analysis pipeline.

Runs Spotiflow spot detection on every frame of a TIFF stack, saves
segmentation masks, blob CSV, and a combined particles-per-frame graph
with cycle/group separators.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from segment import find_spots_spotiflow, create_spot_mask


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


def build_cycle_info(max_time, group1_size, group2_size, skip_frames):
    """Generate cycle/group boundary information."""
    cycle_info = []
    cycle_num = 0
    current_frame = 1  # 1-indexed

    while current_frame <= max_time:
        cycle_num += 1

        # Group 1
        g1_start = current_frame
        g1_end = min(current_frame + group1_size - 1, max_time)
        cycle_info.append({
            "cycle": cycle_num,
            "group": 1,
            "start_frame": g1_start,
            "end_frame": g1_end,
        })
        current_frame = g1_end + 1 + skip_frames

        # Group 2
        if current_frame <= max_time:
            g2_start = current_frame
            g2_end = min(current_frame + group2_size - 1, max_time)
            cycle_info.append({
                "cycle": cycle_num,
                "group": 2,
                "start_frame": g2_start,
                "end_frame": g2_end,
            })
            current_frame = g2_end + 1 + skip_frames

    return cycle_info


def make_combined_graph(particles_per_frame, cycle_info, max_time,
                        group1_size, group2_size, skip_frames,
                        label_group1, label_group2, save_path):
    """Create and save the combined particles-per-frame graph."""
    sep_colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(particles_per_frame["time"], particles_per_frame["n_particles"],
            color="black", linewidth=1.0)

    sep1_labeled = False
    sep2_labeled = False
    for info in cycle_info:
        if info["end_frame"] < max_time:
            sep_color = sep_colors[(info["group"] - 1) % 2]
            if info["group"] == 1 and not sep1_labeled:
                label = label_group1
                sep1_labeled = True
            elif info["group"] == 2 and not sep2_labeled:
                label = label_group2
                sep2_labeled = True
            else:
                label = None
            ax.axvline(x=info["end_frame"] + 0.5, color=sep_color,
                       linestyle="-", alpha=0.6, linewidth=1.5, label=label)

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
    """Create a 2-panel figure: total particle counts per group and G1/G2 ratio per cycle."""
    cycles = sorted(set(info["cycle"] for info in cycle_info))

    g1_counts = []
    g2_counts = []
    ratios_g1_g2 = []
    ratios_g2_g1 = []

    for cyc in cycles:
        g1_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 1]
        g2_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 2]

        g1_total = 0
        if g1_info:
            g1_frames = particles_per_frame[
                (particles_per_frame["time"] >= g1_info[0]["start_frame"]) &
                (particles_per_frame["time"] <= g1_info[0]["end_frame"])
            ]
            g1_total = int(g1_frames["n_particles"].sum())

        g2_total = 0
        if g2_info:
            g2_frames = particles_per_frame[
                (particles_per_frame["time"] >= g2_info[0]["start_frame"]) &
                (particles_per_frame["time"] <= g2_info[0]["end_frame"])
            ]
            g2_total = int(g2_frames["n_particles"].sum())

        g1_counts.append(g1_total)
        g2_counts.append(g2_total)
        ratios_g1_g2.append(g1_total / g2_total if g2_total > 0 else np.nan)
        ratios_g2_g1.append(g2_total / g1_total if g1_total > 0 else np.nan)

    x = np.arange(len(cycles))
    bar_width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1, 1]})

    # Top panel: grouped bar chart of total counts
    ax1.bar(x - bar_width / 2, g1_counts, bar_width, color="tab:blue",
            label=f"{label_group1} (G1)")
    ax1.bar(x + bar_width / 2, g2_counts, bar_width, color="tab:orange",
            label=f"{label_group2} (G2)")
    ax1.set_ylabel("Total Particle Count")
    ax1.set_title("Total Particle Count per Group per Cycle")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom panel: ratio G1/G2
    ax2.bar(x, ratios_g1_g2, 0.5, color="tab:green")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel(f"Ratio ({label_group1} / {label_group2})")
    ax2.set_xlabel("Cycle")
    ax2.set_title(f"Particle Count Ratio ({label_group1} / {label_group2}) per Cycle")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cycles])
    ax2.grid(axis="y", alpha=0.3)

    # Bottom panel: ratio G2/G1
    ax3.bar(x, ratios_g2_g1, 0.5, color="tab:red")
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_ylabel(f"Ratio ({label_group2} / {label_group1})")
    ax3.set_xlabel("Cycle")
    ax3.set_title(f"Particle Count Ratio ({label_group2} / {label_group1}) per Cycle")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in cycles])
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
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

    # Load model
    from spotiflow.model import Spotiflow
    print(f"Loading Spotiflow model '{args.spotiflow_model}'...")
    model = Spotiflow.from_pretrained(args.spotiflow_model, verbose=True)
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
        points, details = find_spots_spotiflow(
            image[i], model,
            prob_thresh=args.prob_thresh,
            min_distance=args.min_distance,
        )
        mask = create_spot_mask(image[i].shape, points, radius=args.spot_radius)
        masks[i] = mask

        if len(points) > 0:
            frame_df = pd.DataFrame({
                "y": points[:, 0],
                "x": points[:, 1],
                "intensity": np.asarray(details.intens).ravel(),
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

    # Particles per frame
    particles_per_frame = (
        df_blobs.groupby("time").size().reset_index(name="n_particles")
    )

    # Build cycle info and save graph
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
