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


def _make_gpu_spot_mask(
    stack_shape,
    df: pd.DataFrame,
    intensity_stack: np.ndarray | None = None,
    dtype=np.int32,
    min_radius: float = 1.0,
    max_radius: float | None = None,
    label_col: str | None = None,
):
    """Build a (T,Y,X) stack of per-frame 2D labels from fitted sigmas.

    When `intensity_stack` is given, also computes per-spot area,
    mean_intensity, max_intensity, spot_radius.
    """
    T, H, W = stack_shape
    n_total = len(df)

    valid = df[["x", "y", "sigma_x", "sigma_y", "time"]].notna().all(axis=1)
    sub = df.loc[valid]

    area_out = np.full(n_total, np.nan, dtype=np.float64)
    mean_out = np.full(n_total, np.nan, dtype=np.float64)
    max_out = np.full(n_total, np.nan, dtype=np.float64)
    radii_out = np.full(n_total, np.nan, dtype=np.float64)

    if len(sub) == 0:
        return np.zeros(stack_shape, dtype=dtype), df.assign(
            area=area_out, mean_intensity=mean_out,
            max_intensity=max_out, spot_radius=radii_out,
        )

    cx = sub["x"].to_numpy()
    cy = sub["y"].to_numpy()
    ct = sub["time"].to_numpy().astype(np.int64)
    sx = sub["sigma_x"].to_numpy()
    sy = sub["sigma_y"].to_numpy()
    valid_idx = np.where(valid)[0]

    if label_col is not None:
        labels = sub[label_col].to_numpy().astype(dtype)
    else:
        labels = np.arange(1, len(sub) + 1, dtype=dtype)

    radii = np.sqrt(2.0) * 0.5 * (sx + sy)
    radii = np.clip(radii, min_radius, max_radius if max_radius else np.inf)
    radii_out[valid_idx] = radii

    mask = np.zeros((T, H, W), dtype=dtype)
    r_int = np.ceil(radii).astype(np.int32)

    for r in np.unique(r_int):
        sel = np.where(r_int == r)[0]
        size = 2 * r + 1
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]

        for i in sel:
            t = ct[i]
            xi = int(round(cx[i])) - r
            yi = int(round(cy[i])) - r
            dx = cx[i] - round(cx[i])
            dy = cy[i] - round(cy[i])
            disk = (xx - dx) ** 2 + (yy - dy) ** 2 <= radii[i] ** 2

            x_start = max(0, xi); x_end = min(W, xi + size)
            y_start = max(0, yi); y_end = min(H, yi + size)
            dx_start = x_start - xi; dx_end = dx_start + (x_end - x_start)
            dy_start = y_start - yi; dy_end = dy_start + (y_end - y_start)
            if x_end <= x_start or y_end <= y_start:
                continue

            local_disk = disk[dy_start:dy_end, dx_start:dx_end]
            target = mask[t, y_start:y_end, x_start:x_end]
            np.copyto(target, labels[i], where=local_disk & (target == 0))

            i_global = valid_idx[i]
            area_out[i_global] = int(local_disk.sum())
            if intensity_stack is not None and local_disk.any():
                disk_pixels = intensity_stack[t, y_start:y_end, x_start:x_end][local_disk]
                mean_out[i_global] = disk_pixels.mean()
                max_out[i_global] = disk_pixels.max()

    df_out = df.assign(
        area=area_out, mean_intensity=mean_out,
        max_intensity=max_out, spot_radius=radii_out,
    )
    return mask, df_out


# ── tracking + analysis ─────────────────────────────────────────────


def _track_blobs(
    blobs_df: pd.DataFrame,
    search_range: float,
    memory: int,
) -> pd.DataFrame:
    """Link detected spots across frames via trackpy. Adds 'particle' column."""
    tp.quiet()
    return tp.link_df(blobs_df, search_range=search_range, memory=memory)


def _compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-track geometric (and optional intensity) properties.

    Input must have columns: frame (int), y, x, particle (int).
    """
    cols = ["track_id", "length", "total_displacement",
            "net_displacement", "mean_speed"]
    if image is not None:
        cols += ["mean_intensity", "max_intensity", "intensity_std"]

    if len(tracked_df) == 0:
        return pd.DataFrame(columns=cols)

    if image is not None:
        if image.ndim == 3:
            t_max = int(tracked_df["frame"].max())
            if image.shape[0] <= t_max:
                raise ValueError("Image T-axis does not span tracked frames")
        elif image.ndim != 2:
            raise ValueError("Image must be 2D (Y, X) or 3D (T, Y, X)")

    rows = []
    for tid, grp in tracked_df.groupby("particle"):
        grp = grp.sort_values("frame")
        coords = grp[["y", "x"]].to_numpy()
        length = len(grp)
        if length > 1:
            diffs = np.diff(coords, axis=0)
            total = float(np.linalg.norm(diffs, axis=1).sum())
            net = float(np.linalg.norm(coords[-1] - coords[0]))
        else:
            total = 0.0
            net = 0.0
        mean_speed = total / max(length - 1, 1)
        row = {
            "track_id": int(tid),
            "length": length,
            "total_displacement": total,
            "net_displacement": net,
            "mean_speed": mean_speed,
        }
        if image is not None:
            frames = grp["frame"].astype(int).to_numpy()
            ys = np.clip(np.rint(coords[:, 0]).astype(int), 0, image.shape[-2] - 1)
            xs = np.clip(np.rint(coords[:, 1]).astype(int), 0, image.shape[-1] - 1)
            vals = image[frames, ys, xs] if image.ndim == 3 else image[ys, xs]
            row["mean_intensity"] = float(vals.mean())
            row["max_intensity"] = float(vals.max())
            row["intensity_std"] = float(vals.std())
        rows.append(row)

    return pd.DataFrame(rows, columns=cols).sort_values("track_id").reset_index(drop=True)


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
