"""Bulk track analysis CLI.

Standalone script: preprocess, segment (Spotiflow), track (trackpy), and
analyze every .tif/.tiff stack in an input directory. One sub-folder of
artifacts per input file. See
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
import torch
import trackpy as tp
from scipy.ndimage import gaussian_filter
from torchimize.functions import lsq_lma_parallel
from tqdm import tqdm

# ── helpers ─────────────────────────────────────────────────────────


def _select_device(prefer: str = "cuda") -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _walking_average(image_stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a walking (rolling) average along the time axis.

    For a T,Y,X stack, each frame is replaced by the mean of the surrounding
    *window* frames (centered). Frames near the edges use a truncated window.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")
    if window % 2 == 0:
        window += 1  # ensure odd

    n_frames = image_stack.shape[0]
    half = window // 2
    result = np.empty_like(image_stack, dtype=np.float64)

    cumsum = np.cumsum(image_stack.astype(np.float64), axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)

    for t in range(n_frames):
        lo = max(t - half, 0)
        hi = min(t + half + 1, n_frames)
        result[t] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result.astype(image_stack.dtype)


# ── segmentation ────────────────────────────────────────────────────


# ── fitting + mask ──────────────────────────────────────────────────


# ── tracking + analysis ─────────────────────────────────────────────


# ── plotting ────────────────────────────────────────────────────────


# ── per-file pipeline ───────────────────────────────────────────────


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

    p.add_argument("--walk-window", type=int, default=3,
                   help="Walking-average window (odd; coerced if even). Default: 3.")
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

    p.add_argument("--min-track-length", type=int, default=2,
                   help="Min track length included in plots (CSVs unfiltered). Default: 2.")
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
    print(f"Input:  {args.input_dir}", file=sys.stderr)
    print(f"Output: {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
