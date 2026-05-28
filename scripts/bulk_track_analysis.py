"""Bulk track analysis CLI.

Headless batch pipeline: preprocess, detect (Spotiflow), fit, mask, track
(trackpy), and analyze every .tif/.tiff stack in an input directory. One
sub-folder of artifacts per input file.

This script reuses the napari-spotiflow-tracking plugin's processing functions
directly (no napari / Qt / UI). Install the plugin into the same environment
first (e.g. ``pip install -e .``). See
docs/superpowers/specs/2026-05-08-bulk-track-analysis-cli-design.md
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting — no display needed
import matplotlib.pyplot as plt
import numpy as np
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
from napari_spotiflow_tracking._segmentation import detect_spots, load_model
from napari_spotiflow_tracking._track_properties import compute_track_properties
from napari_spotiflow_tracking._tracking import track_blobs

# ── segmentation ────────────────────────────────────────────────────


def _detect_all_frames(
    stack: np.ndarray,
    model,
    prob_thresh: float,
    min_distance: int,
) -> pd.DataFrame:
    """Run the plugin's Spotiflow detection per frame; return time, y, x.

    Thin loop over ``_segmentation.detect_spots`` (which works on a single 2D
    frame). The column name ``time`` matches what ``fit_all_frames_gpu``
    expects.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {stack.ndim}D")

    rows_t, rows_y, rows_x = [], [], []
    for t in tqdm(range(stack.shape[0]), desc="  detect", leave=False, unit="frame"):
        points, _ = detect_spots(
            stack[t], model, prob_thresh=prob_thresh, min_distance=min_distance,
        )
        if len(points) == 0:
            continue
        rows_t.extend([t] * len(points))
        rows_y.extend(points[:, 0].astype(float).tolist())
        rows_x.extend(points[:, 1].astype(float).tolist())

    return pd.DataFrame({"time": rows_t, "y": rows_y, "x": rows_x})


# ── plotting ────────────────────────────────────────────────────────


def _plot_length_hist(track_props: pd.DataFrame, out_path: Path) -> None:
    """Histogram of per-track length."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(track_props):
        ax.hist(track_props["length"], bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("# tracks")
    ax.set_title("Track length distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_speed_hist(track_props: pd.DataFrame, out_path: Path) -> None:
    """Histogram of per-track mean_speed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(track_props):
        ax.hist(track_props["mean_speed"], bins=30, color="seagreen", edgecolor="black")
    ax.set_xlabel("Mean speed (px / frame)")
    ax.set_ylabel("# tracks")
    ax.set_title("Track mean-speed distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _top_n_track_ids(tracked_df: pd.DataFrame, max_tracks: int) -> list[int]:
    """Return up to max_tracks particle IDs with the longest tracks."""
    if "particle" not in tracked_df.columns or len(tracked_df) == 0:
        return []
    counts = tracked_df.groupby("particle").size().sort_values(ascending=False)
    return counts.head(max_tracks).index.tolist()


def _plot_length_per_track(
    tracked_df: pd.DataFrame, out_path: Path, max_tracks: int,
) -> None:
    """One line per track: cumulative displacement vs frame."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ids = _top_n_track_ids(tracked_df, max_tracks)
    for tid in ids:
        grp = tracked_df[tracked_df["particle"] == tid].sort_values("frame")
        if len(grp) < 2:
            continue
        coords = grp[["y", "x"]].to_numpy()
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(steps)])
        ax.plot(grp["frame"].to_numpy(), cum, alpha=0.5, linewidth=0.7)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative displacement (px)")
    ax.set_title(f"Per-track cumulative displacement (top {len(ids)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_speed_per_track(
    tracked_df: pd.DataFrame, out_path: Path, max_tracks: int,
) -> None:
    """One line per track: frame-to-frame speed vs frame.

    Speed at frame f is the distance from the previous frame to frame f
    (so the first frame of every track has no value and is skipped).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ids = _top_n_track_ids(tracked_df, max_tracks)
    for tid in ids:
        grp = tracked_df[tracked_df["particle"] == tid].sort_values("frame")
        if len(grp) < 2:
            continue
        coords = grp[["y", "x"]].to_numpy()
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        ax.plot(grp["frame"].to_numpy()[1:], steps, alpha=0.5, linewidth=0.7)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Instantaneous speed (px / frame)")
    ax.set_title(f"Per-track speed (top {len(ids)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    spots = _detect_all_frames(
        pre, model, prob_thresh=args.prob_thresh, min_distance=args.min_distance,
    )
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
                              "mean_intensity", "max_intensity", "intensity_std"]
                     ).to_csv(file_out / "track_analysis.csv", index=False)
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

    # ── Analyze ──────────────────────────────────────────────────
    track_props = compute_track_properties(tracked, image=pre)
    track_props.to_csv(file_out / "track_analysis.csv", index=False)

    # ── Plots (filter applies to plots only) ─────────────────────
    plot_props = track_props[track_props["length"] >= args.min_track_length]
    plot_tracked = tracked[tracked["particle"].isin(plot_props["track_id"])]
    _plot_length_hist(plot_props, file_out / "length_hist.png")
    _plot_speed_hist(plot_props, file_out / "speed_hist.png")
    _plot_length_per_track(plot_tracked, file_out / "length_per_track.png",
                           args.max_tracks_plot)
    _plot_speed_per_track(plot_tracked, file_out / "speed_per_track.png",
                          args.max_tracks_plot)

    return {"stem": stem, "n_spots": int(len(spots_out)),
            "n_tracks": int(track_props["track_id"].nunique()),
            "status": "OK", "error": ""}


# ── CLI ─────────────────────────────────────────────────────────────


@dataclass
class Args:
    input_dir: Path
    output_dir: Path
    walk_window: int
    no_walk: bool
    gauss_sigma: float
    no_blur: bool
    no_submean: bool
    prob_thresh: float
    min_distance: int
    device: str
    fit_win: int
    max_radius: float
    search_range: float
    memory: int
    min_track_length: int
    max_tracks_plot: int
    skip_existing: bool


def parse_args(argv=None) -> Args:
    p = argparse.ArgumentParser(
        description="Bulk preprocess + segment + track + analyze TIFF stacks.",
    )
    p.add_argument("input_dir", type=Path, help="Directory of .tif/.tiff stacks (T,Y,X).")
    p.add_argument("output_dir", type=Path, help="Output directory (created if missing).")

    p.add_argument("--walk-window", type=int, default=4,
                   help="Walking-average window (odd; coerced if even). Default: 4.")
    p.add_argument("--no-walk", action="store_true", help="Skip walking average.")
    p.add_argument("--gauss-sigma", type=float, default=1.0,
                   help="Gaussian-blur sigma (per frame, Y/X only). Default: 1.0.")
    p.add_argument("--no-blur", action="store_true", help="Skip Gaussian blur.")
    p.add_argument("--no-submean", action="store_true", help="Skip stack-mean subtraction.")

    p.add_argument("--prob-thresh", type=float, default=0.5,
                   help="Spotiflow probability threshold. Default: 0.5.")
    p.add_argument("--min-distance", type=int, default=2,
                   help="Spotiflow min-distance NMS. Default: 2.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="Compute device (cuda falls back to cpu). Default: cuda.")

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

    ns = p.parse_args(argv)
    return Args(
        input_dir=ns.input_dir, output_dir=ns.output_dir,
        walk_window=ns.walk_window, no_walk=ns.no_walk,
        gauss_sigma=ns.gauss_sigma, no_blur=ns.no_blur, no_submean=ns.no_submean,
        prob_thresh=ns.prob_thresh, min_distance=ns.min_distance, device=ns.device,
        fit_win=ns.fit_win, max_radius=ns.max_radius,
        search_range=ns.search_range, memory=ns.memory,
        min_track_length=ns.min_track_length,
        max_tracks_plot=ns.max_tracks_plot,
        skip_existing=ns.skip_existing,
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"ERROR: input_dir is not a directory: {args.input_dir}",
              file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(
        list(args.input_dir.glob("*.tif")) + list(args.input_dir.glob("*.tiff"))
    )
    if not inputs:
        print(f"ERROR: no .tif/.tiff files found in {args.input_dir}",
              file=sys.stderr)
        return 2

    device = _get_device(use_gpu=(args.device == "cuda"))
    if device != args.device:
        print(f"WARN: requested device={args.device}, falling back to {device}",
              file=sys.stderr)
    # Use the resolved device everywhere downstream (model load + GPU fit).
    args.device = device

    print(f"Loading Spotiflow 'general' on {device}...", file=sys.stderr)
    model = load_model("general", device=device)

    log_path = args.output_dir / "run.log"
    n_ok = n_fail = n_skip = 0
    with log_path.open("a", encoding="utf-8") as log_f:
        for tif in tqdm(inputs, desc="files", unit="file"):
            try:
                row = process_file(tif, args.output_dir, model, args)
            except Exception:
                tb = traceback.format_exc()
                row = {"stem": tif.stem, "n_spots": 0, "n_tracks": 0,
                       "status": "FAIL", "error": tb.replace("\n", " | ")}
                print(f"FAIL: {tif.name} — see run.log", file=sys.stderr)

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

    print(
        f"Done: {n_ok} OK, {n_skip} skipped, {n_fail} failed. "
        f"Log: {log_path}",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
