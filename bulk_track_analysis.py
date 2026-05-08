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


def _gaussian_blur_stack(image_stack: np.ndarray, sigma: float) -> np.ndarray:
    """Per-frame Gaussian blur on a (T,Y,X) stack.

    Applies the 2D filter only on Y/X (sigma=0 along T) so frames are
    smoothed independently. Returns the same dtype as the input.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")
    if sigma <= 0:
        return image_stack
    blurred = gaussian_filter(image_stack.astype(np.float64), sigma=(0, sigma, sigma))
    return blurred.astype(image_stack.dtype)


def _subtract_stack_mean(image_stack: np.ndarray) -> np.ndarray:
    """Subtract the mean projection along T from every frame.

    Computes mean in float64. For integer dtypes, clips to the dtype range
    before casting back (so unsigned negatives clamp to 0 instead of
    wrapping). For float dtypes, casts directly.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {image_stack.ndim}D")

    mean_proj = image_stack.mean(axis=0)  # 2D Y,X (float64)
    diff = image_stack.astype(np.float64) - mean_proj  # broadcasts over T

    if np.issubdtype(image_stack.dtype, np.integer):
        info = np.iinfo(image_stack.dtype)
        diff = np.clip(diff, info.min, info.max)
    return diff.astype(image_stack.dtype)


# ── segmentation ────────────────────────────────────────────────────


def _load_spotiflow(model_name: str, device: str):
    """Load a pretrained Spotiflow model."""
    from spotiflow.model import Spotiflow
    return Spotiflow.from_pretrained(model_name, map_location=device)


def _detect_all_frames(
    stack: np.ndarray,
    model,
    prob_thresh: float,
    min_distance: int,
) -> pd.DataFrame:
    """Run Spotiflow per frame; return a DataFrame with columns time, y, x.

    The column name `time` matches what `_fit_all_frames_gpu` expects.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D (T,Y,X) stack, got {stack.ndim}D")

    rows_t, rows_y, rows_x = [], [], []
    for t in tqdm(range(stack.shape[0]), desc="  detect", leave=False, unit="frame"):
        points, _ = model.predict(
            stack[t], prob_thresh=prob_thresh,
            min_distance=min_distance, verbose=False,
        )
        if len(points) == 0:
            continue
        ys = points[:, 0].astype(float)
        xs = points[:, 1].astype(float)
        rows_t.extend([t] * len(points))
        rows_y.extend(ys.tolist())
        rows_x.extend(xs.tolist())

    return pd.DataFrame({"time": rows_t, "y": rows_y, "x": rows_x})


# ── fitting + mask ──────────────────────────────────────────────────


def _gaussian_2d_batched(p, xy):
    amp = p[:, 0:1]; x0 = p[:, 1:2]; y0 = p[:, 2:3]
    sx = p[:, 3:4]; sy = p[:, 4:5]; bg = p[:, 5:6]
    x = xy[0:1, :]; y = xy[1:2, :]
    dx = (x - x0) / sx
    dy = (y - y0) / sy
    out = amp * torch.exp(-0.5 * (dx * dx + dy * dy)) + bg
    return out.unsqueeze(1)


def _fit_residuals(p, data, xy):
    return _gaussian_2d_batched(p, xy) - data


def _extract_crops_vec(stack: np.ndarray, df: pd.DataFrame, win: int):
    T, H, W = stack.shape
    pts_y = np.round(df["y"].to_numpy()).astype(np.int64)
    pts_x = np.round(df["x"].to_numpy()).astype(np.int64)
    pts_t = df["time"].to_numpy().astype(np.int64)
    valid = (
        (pts_y >= win) & (pts_y < H - win)
        & (pts_x >= win) & (pts_x < W - win)
        & (pts_t >= 0) & (pts_t < T)
    )
    ty = pts_y[valid]; tx = pts_x[valid]; tt = pts_t[valid]
    dy, dx = np.meshgrid(np.arange(-win, win + 1),
                         np.arange(-win, win + 1), indexing="ij")
    ys = ty[:, None, None] + dy[None]
    xs = tx[:, None, None] + dx[None]
    ts = tt[:, None, None]
    crops = stack[ts, ys, xs].astype(np.float32)
    return crops, valid


def _fit_all_frames_gpu(
    stack: np.ndarray,
    df: pd.DataFrame,
    win: int = 2,
    device: str = "cuda",
    max_iter: int = 50,
    chunk_size: int = 20000,
) -> pd.DataFrame:
    """Batch 2D Gaussian fit across an entire (T,Y,X) stack via torchimize.

    `df` must have columns time, y, x. Output adds amp, x_fit, y_fit,
    sigma_x, sigma_y, bg.
    """
    size = 2 * win + 1
    M = size * size
    n_total = len(df)

    crops, valid = _extract_crops_vec(stack, df, win)
    N = len(crops)

    amp_out = np.full(n_total, np.nan, dtype=np.float64)
    x_out = np.full(n_total, np.nan, dtype=np.float64)
    y_out = np.full(n_total, np.nan, dtype=np.float64)
    sx_out = np.full(n_total, np.nan, dtype=np.float64)
    sy_out = np.full(n_total, np.nan, dtype=np.float64)
    bg_out = np.full(n_total, np.nan, dtype=np.float64)

    if N == 0:
        return df.assign(
            amp=amp_out, x_fit=x_out, y_fit=y_out,
            sigma_x=sx_out, sigma_y=sy_out, bg=bg_out,
        )

    yy, xx = torch.meshgrid(
        torch.arange(-win, win + 1, dtype=torch.float32),
        torch.arange(-win, win + 1, dtype=torch.float32),
        indexing="ij",
    )
    xy = torch.stack([xx.flatten(), yy.flatten()]).to(device)
    wvec = torch.ones(1, device=device)

    final_all = np.empty((N, 6), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = crops[start:end]
        B = len(chunk)

        data = torch.from_numpy(chunk.reshape(B, 1, M)).to(device)
        d2 = data[:, 0, :]
        amp0 = d2.max(1).values - d2.min(1).values
        bg0 = d2.min(1).values
        init = torch.stack([
            amp0,
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.full((B,), 1.5, device=device),
            torch.full((B,), 1.5, device=device),
            bg0,
        ], dim=1)

        coeffs_list = lsq_lma_parallel(
            p=init, function=_fit_residuals, args=(data, xy), wvec=wvec,
            ftol=1e-6, ptol=1e-6, gtol=1e-6, max_iter=max_iter,
        )
        final_all[start:end] = coeffs_list[-1].cpu().numpy()

    pts_y = np.round(df["y"].to_numpy()).astype(np.int64)
    pts_x = np.round(df["x"].to_numpy()).astype(np.int64)

    final = final_all.astype(np.float64)
    amp_out[valid] = final[:, 0]
    x_out[valid] = pts_x[valid] + final[:, 1]
    y_out[valid] = pts_y[valid] + final[:, 2]
    sx_out[valid] = np.abs(final[:, 3])
    sy_out[valid] = np.abs(final[:, 4])
    bg_out[valid] = final[:, 5]

    return df.assign(
        amp=amp_out, x_fit=x_out, y_fit=y_out,
        sigma_x=sx_out, sigma_y=sy_out, bg=bg_out,
    )


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
