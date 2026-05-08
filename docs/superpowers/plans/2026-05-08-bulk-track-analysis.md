# Bulk track analysis CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `bulk_track_analysis.py` stub with a standalone CLI that runs preprocessing → segmentation → tracking → track-analysis over every TIFF in an input directory and writes per-file artifacts.

**Architecture:** Single self-contained Python script in the project root. Functions are copied (not imported) from the `napari_spotiflow_tracking` package so the script can ship without depending on the plugin. One `process_file()` per input TIFF, called from a `main()` loop that manages the Spotiflow model, `tqdm` progress, and `run.log`.

**Tech Stack:** numpy, scipy.ndimage, pandas, tifffile, trackpy, spotiflow, torch, torchimize, matplotlib, tqdm.

**Spec:** [docs/superpowers/specs/2026-05-08-bulk-track-analysis-cli-design.md](../specs/2026-05-08-bulk-track-analysis-cli-design.md).

**Source files (copy-from references):**
- `src/napari_spotiflow_tracking/_preprocessing.py` — `walking_average` (lines 306–339), `apply_calculation` clipping pattern (lines 405–440).
- `src/napari_spotiflow_tracking/_segmentation.py` — `load_model` (lines 13–19), `detect_spots` (lines 22–38).
- `src/napari_spotiflow_tracking/_fitting.py` — `_gaussian_2d_batched`, `_residuals`, `_extract_crops_vec`, `fit_all_frames_gpu`, `make_gpu_spot_mask` (lines 389–619).
- `src/napari_spotiflow_tracking/_tracking.py` — `track_blobs` (lines 6–22).
- `src/napari_spotiflow_tracking/_track_properties.py` — `compute_track_properties` (lines 12–72).

**Tests:** Per spec, no tests are written for this script. Verification is via a smoke run on real data after the final task.

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `bulk_track_analysis.py` | **Replace** existing 14-line stub | Single-file CLI: imports, helpers, segmentation, fitting, tracking, analysis, plotting, `process_file`, `main`. ~400 lines. |

No new tests, no new modules.

---

## Task 1: Replace stub with skeleton + imports

**Files:**
- Modify: `bulk_track_analysis.py` (overwrite)

- [ ] **Step 1: Replace the file with the skeleton below.**

```python
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


# ── segmentation ────────────────────────────────────────────────────


# ── fitting + mask ──────────────────────────────────────────────────


# ── tracking + analysis ─────────────────────────────────────────────


# ── plotting ────────────────────────────────────────────────────────


# ── per-file pipeline ───────────────────────────────────────────────


# ── CLI ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it imports.**

Run:
```bash
.venv\Scripts\python -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'bulk_track_analysis.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)"
```
Expected: No output, exit 0. (`main` is referenced under `__main__` but not defined yet — the import will succeed because the `__main__` block isn't run on import.)

If you get `NameError: name 'main' is not defined`, you accidentally indented the `if __name__` differently or are running the file directly — re-check.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): replace stub with module skeleton"
```

---

## Task 2: CLI argument parser

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── CLI ──`)

- [ ] **Step 1: Add `parse_args` and the `Args` dataclass under the `# ── CLI ──` section, and a placeholder `main`.**

```python
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
```

- [ ] **Step 2: Verify `--help` works.**

Run: `.venv\Scripts\python bulk_track_analysis.py --help`
Expected: Help text listing all the flags above; exit 0.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add CLI argument parser"
```

---

## Task 3: Device selection helper

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── helpers ──`)

- [ ] **Step 1: Add `_select_device` under `# ── helpers ──`.**

```python
def _select_device(prefer: str = "cuda") -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 2: Verify.**

Run:
```bash
.venv\Scripts\python -c "import bulk_track_analysis as m; print(m._select_device('cuda')); print(m._select_device('cpu'))"
```
Expected: First line is `cuda` if you have a GPU else `cpu`; second line is always `cpu`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add _select_device helper"
```

---

## Task 4: Walking average

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── helpers ──`)

- [ ] **Step 1: Append `_walking_average` under `# ── helpers ──`.**

This is `walking_average` from `src/napari_spotiflow_tracking/_preprocessing.py:306-339`, renamed with a leading underscore. Paste verbatim:

```python
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
```

- [ ] **Step 2: Verify.**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, bulk_track_analysis as m; s = np.arange(60).reshape(5,3,4).astype(np.uint16); out = m._walking_average(s, 3); print(out.shape, out.dtype)"
```
Expected: `(5, 3, 4) uint16`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add walking-average helper"
```

---

## Task 5: Gaussian blur per frame

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── helpers ──`)

- [ ] **Step 1: Append `_gaussian_blur_stack` under `# ── helpers ──`.**

```python
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
```

- [ ] **Step 2: Verify.**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, bulk_track_analysis as m; s = np.zeros((4,16,16), dtype=np.uint16); s[:,8,8] = 1000; out = m._gaussian_blur_stack(s, 1.5); print(out.shape, out.dtype, out[0,8,8] < 1000)"
```
Expected: `(4, 16, 16) uint16 True` (the central pixel is spread by the blur).

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add per-frame Gaussian blur helper"
```

---

## Task 6: Stack-mean subtraction with dtype-safe cast

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── helpers ──`)

- [ ] **Step 1: Append `_subtract_stack_mean` under `# ── helpers ──`.**

```python
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
```

- [ ] **Step 2: Verify (uint16 negatives clamp to 0; values above the mean stay positive).**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, bulk_track_analysis as m; s = np.array([[[10,200]],[[100,50]]], dtype=np.uint16); out = m._subtract_stack_mean(s); print(out.tolist(), out.dtype)"
```
Expected: `[[[0, 75]], [[45, 0]]] uint16` — frame 0 col 0 (10−55) clamps to 0; frame 0 col 1 (200−125) = 75; frame 1 col 0 (100−55) = 45; frame 1 col 1 (50−125) clamps to 0.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add stack-mean subtraction with dtype clipping"
```

---

## Task 7: Spotiflow loader + per-frame detection

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── segmentation ──`)

- [ ] **Step 1: Append `_load_spotiflow` and `_detect_all_frames` under `# ── segmentation ──`.**

```python
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
```

- [ ] **Step 2: Verify the loader (skip if no GPU/network — will download on first run).**

Run:
```bash
.venv\Scripts\python -c "import bulk_track_analysis as m; mdl = m._load_spotiflow('general', 'cpu'); print(type(mdl).__name__)"
```
Expected: `Spotiflow` printed (or class name from spotiflow). May take several seconds to download on first run.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add Spotiflow loader and per-frame detection"
```

---

## Task 8: GPU fit — `_fit_all_frames_gpu` + helpers

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── fitting + mask ──`)

- [ ] **Step 1: Append the GPU fit helpers and entry point under `# ── fitting + mask ──`.**

These are copied from `src/napari_spotiflow_tracking/_fitting.py:402-519`. The functions reference `torch` and `lsq_lma_parallel` directly (already imported at top of `bulk_track_analysis.py`). Module-level `_torch` / `_lsq_lma_parallel` aliases are not needed — use the top-level imports.

```python
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
```

- [ ] **Step 2: Verify with a tiny synthetic stack (CPU is fine if no GPU).**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, pandas as pd, bulk_track_analysis as m; s = np.zeros((2,32,32),dtype=np.float32); s[:,16,16]=1.0; df = pd.DataFrame({'time':[0,1],'y':[16.0,16.0],'x':[16.0,16.0]}); dev = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'; out = m._fit_all_frames_gpu(s, df, win=2, device=dev); print(list(out.columns))"
```
Expected: `['time', 'y', 'x', 'amp', 'x_fit', 'y_fit', 'sigma_x', 'sigma_y', 'bg']`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add GPU-batched 2D Gaussian fit"
```

---

## Task 9: Mask generation — `_make_gpu_spot_mask`

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── fitting + mask ──`)

- [ ] **Step 1: Append `_make_gpu_spot_mask` under `# ── fitting + mask ──`.**

This is `make_gpu_spot_mask` from `src/napari_spotiflow_tracking/_fitting.py:522-619`, copied verbatim with a leading underscore:

```python
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
```

- [ ] **Step 2: Verify mask shape and props columns.**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, pandas as pd, bulk_track_analysis as m; s = np.zeros((2,32,32),dtype=np.float32); df = pd.DataFrame({'time':[0],'y':[16.0],'x':[16.0],'sigma_x':[1.5],'sigma_y':[1.5]}); mask, props = m._make_gpu_spot_mask(s.shape, df, intensity_stack=s, max_radius=2.0); print(mask.shape, mask.dtype, list(props.columns))"
```
Expected: `(2, 32, 32) int32 ['time', 'y', 'x', 'sigma_x', 'sigma_y', 'area', 'mean_intensity', 'max_intensity', 'spot_radius']`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add make_gpu_spot_mask"
```

---

## Task 10: Tracking + per-track properties

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── tracking + analysis ──`)

- [ ] **Step 1: Append `_track_blobs` and `_compute_track_properties` under `# ── tracking + analysis ──`.**

`_track_blobs` is from `src/napari_spotiflow_tracking/_tracking.py:6-22`. `_compute_track_properties` is from `src/napari_spotiflow_tracking/_track_properties.py:12-72`. Both renamed with a leading underscore. The tracker expects a `frame` column (not `time`), so the caller must rename before calling — see Task 13.

```python
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
```

- [ ] **Step 2: Verify with a synthetic two-frame tracking case.**

Run:
```bash
.venv\Scripts\python -c "import numpy as np, pandas as pd, bulk_track_analysis as m; df = pd.DataFrame({'frame':[0,1,0,1],'y':[10.0,11.0,50.0,52.0],'x':[10.0,10.0,50.0,50.0]}); tr = m._track_blobs(df, 5.0, 1); props = m._compute_track_properties(tr); print(len(props), 'tracks; lengths', props['length'].tolist())"
```
Expected: `2 tracks; lengths [2, 2]`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add tracking + per-track properties"
```

---

## Task 11: Histogram plots

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── plotting ──`)

- [ ] **Step 1: Append `_plot_length_hist` and `_plot_speed_hist` under `# ── plotting ──`.**

```python
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
```

- [ ] **Step 2: Verify by writing a histogram to a temp file.**

Run:
```bash
.venv\Scripts\python -c "import pandas as pd, tempfile, os, bulk_track_analysis as m; from pathlib import Path; df = pd.DataFrame({'track_id':[1,2,3],'length':[5,8,12],'mean_speed':[0.5,1.2,0.9]}); tmp = Path(tempfile.mkdtemp()) / 'h.png'; m._plot_length_hist(df, tmp); print(tmp.exists(), tmp.stat().st_size > 1000)"
```
Expected: `True True` (PNG was written, non-trivial size).

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add length and speed histogram plots"
```

---

## Task 12: Per-track plots (cumulative displacement, instantaneous speed)

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── plotting ──`)

- [ ] **Step 1: Append `_plot_length_per_track` and `_plot_speed_per_track` under `# ── plotting ──`.**

```python
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
```

- [ ] **Step 2: Verify per-track plot writes a PNG.**

Run:
```bash
.venv\Scripts\python -c "import pandas as pd, tempfile, bulk_track_analysis as m; from pathlib import Path; df = pd.DataFrame({'particle':[1,1,1,2,2],'frame':[0,1,2,0,1],'y':[10.0,11.0,13.0,50.0,52.0],'x':[10.0,11.0,12.0,50.0,50.0]}); tmp = Path(tempfile.mkdtemp()) / 'p.png'; m._plot_length_per_track(df, tmp, 200); print(tmp.exists())"
```
Expected: `True`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add per-track displacement and speed plots"
```

---

## Task 13: `process_file` — per-file pipeline

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── per-file pipeline ──`)

- [ ] **Step 1: Append `process_file` under `# ── per-file pipeline ──`.**

```python
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
        pre = _walking_average(pre, window=args.walk_window)
    if not args.no_blur:
        pre = _gaussian_blur_stack(pre, sigma=args.gauss_sigma)
    if not args.no_submean:
        pre = _subtract_stack_mean(pre)

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
    fitted = _fit_all_frames_gpu(pre, spots, win=args.fit_win, device=args.device)
    mask, props = _make_gpu_spot_mask(
        pre.shape, fitted, intensity_stack=pre, max_radius=args.max_radius,
    )

    # spots.csv columns: rename `time` → `frame`
    spots_out = props.rename(columns={"time": "frame"})
    spots_out.to_csv(file_out / "spots.csv", index=False)
    tifffile.imwrite(str(file_out / "mask.tif"), mask, bigtiff=True)

    # ── Track ────────────────────────────────────────────────────
    tracker_in = spots_out[["frame", "y", "x"]].copy()
    tracked = _track_blobs(tracker_in, args.search_range, args.memory)
    # Re-attach the fit/intensity columns for downstream consumers.
    tracked = tracked.merge(
        spots_out.drop(columns=["frame", "y", "x"], errors="ignore"),
        left_index=True, right_index=True, how="left",
    )
    tracked.to_csv(file_out / "tracks.csv", index=False)

    # ── Analyze ──────────────────────────────────────────────────
    track_props = _compute_track_properties(tracked, image=pre)
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
```

- [ ] **Step 2: Verify the function imports without error (no smoke run yet — that comes in Task 15).**

Run:
```bash
.venv\Scripts\python -c "import bulk_track_analysis as m; print(callable(m.process_file))"
```
Expected: `True`.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add per-file pipeline (process_file)"
```

---

## Task 14: `main()` — directory loop, model load, run.log

**Files:**
- Modify: `bulk_track_analysis.py` (under `# ── CLI ──`, replace the placeholder `main`)

- [ ] **Step 1: Replace the placeholder `main` with the full version below.**

```python
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

    device = _select_device(args.device)
    if device != args.device:
        print(f"WARN: requested device={args.device}, falling back to {device}",
              file=sys.stderr)

    print(f"Loading Spotiflow 'general' on {device}...", file=sys.stderr)
    model = _load_spotiflow("general", device=device)

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
```

- [ ] **Step 2: Verify `--help` still works and a missing input directory exits non-zero.**

Run: `.venv\Scripts\python bulk_track_analysis.py --help`
Expected: Help text, exit 0.

Run: `.venv\Scripts\python bulk_track_analysis.py nonexistent_dir out_dir`
Expected: `ERROR: input_dir is not a directory: nonexistent_dir`, exit 2.

- [ ] **Step 3: Commit.**

```bash
git add bulk_track_analysis.py
git commit -m "feat(bulk_track_analysis): add main() with directory loop and run.log"
```

---

## Task 15: Smoke run on real data

**Files:**
- No code changes. Verification only.

- [ ] **Step 1: Pick a small directory of TIFF stacks the user already has (or create a one-file dir).**

If the project root has a sample TIFF (e.g. `walkAv_SA_n2v_3frames.zip` extracted, or any T,Y,X `.tif` the user supplies), copy a single file into a temporary `smoke_in/` directory:

```bash
mkdir smoke_in smoke_out
# manually copy ONE sample .tif into smoke_in/
```

If no sample is available, ask the user to point you at one before continuing.

- [ ] **Step 2: Run the pipeline.**

Run:
```bash
.venv\Scripts\python bulk_track_analysis.py smoke_in smoke_out
```

Expected:
- Stderr: `Loading Spotiflow 'general' on cuda...` (or `cpu`).
- A `tqdm` progress bar `files: 100%|...|`.
- Per-file inner bar `detect: 100%|...|` while inference runs.
- Final line `Done: 1 OK, 0 skipped, 0 failed.`

- [ ] **Step 3: Inspect the output folder.**

Run: `dir smoke_out\<stem>`
Expected files: `mask.tif`, `spots.csv`, `tracks.csv`, `track_analysis.csv`, `length_hist.png`, `speed_hist.png`, `length_per_track.png`, `speed_per_track.png`.

Run: `.venv\Scripts\python -c "import pandas as pd; print(pd.read_csv(r'smoke_out/<stem>/track_analysis.csv').head())"`
Expected: A DataFrame head with `track_id, length, total_displacement, net_displacement, mean_speed, mean_intensity, max_intensity, intensity_std` columns.

Open one of the PNGs in your image viewer to sanity-check it's not blank.

- [ ] **Step 4: Test `--skip-existing` re-running the same command.**

Run again: `.venv\Scripts\python bulk_track_analysis.py smoke_in smoke_out --skip-existing`
Expected: Final line `Done: 0 OK, 1 skipped, 0 failed.`

- [ ] **Step 5: If everything looks good, clean up the smoke artifacts (don't commit them) and commit the plan execution as done.**

```bash
# Remove the smoke dirs from the working tree (they should not be tracked).
rmdir /s /q smoke_in smoke_out
# No new commit needed — the script is feature-complete from Task 14's commit.
```

---

## Self-Review

**Spec coverage check (against `docs/superpowers/specs/2026-05-08-bulk-track-analysis-cli-design.md`):**

| Spec requirement | Task |
| --- | --- |
| CLI surface (input_dir, output_dir, options) | Task 2 |
| `--walk-window`, `--no-walk` | Task 2 + Task 13 |
| `--gauss-sigma`, `--no-blur` | Task 2 + Task 13 |
| `--no-submean` | Task 2 + Task 13 |
| `--prob-thresh`, `--min-distance`, `--device` | Task 2 + Task 7 + Task 14 |
| `--fit-win`, `--max-radius` | Task 2 + Tasks 8, 9, 13 |
| `--search-range`, `--memory` | Task 2 + Task 13 |
| `--min-track-length` | Task 2 + Task 13 |
| `--max-tracks-plot` | Task 2 + Task 12 + Task 13 |
| `--skip-existing` | Task 2 + Task 13 + Task 14 |
| Walking average | Task 4 |
| Per-frame Gaussian blur | Task 5 |
| Stack-mean subtraction with dtype clipping | Task 6 |
| Spotiflow general model loaded once | Task 7 + Task 14 |
| Per-frame Spotiflow detection with tqdm | Task 7 |
| `fit_all_frames_gpu` | Task 8 |
| `make_gpu_spot_mask` with intensity_stack | Task 9 |
| trackpy `track_blobs` | Task 10 |
| `compute_track_properties` | Task 10 |
| `mask.tif` (T,Y,X int32, BigTIFF) | Task 13 |
| `spots.csv` (with `time`→`frame` rename) | Task 13 |
| `tracks.csv` | Task 13 |
| `track_analysis.csv` (unfiltered) | Task 13 |
| `length_hist.png`, `speed_hist.png` | Task 11 + Task 13 |
| `length_per_track.png`, `speed_per_track.png` | Task 12 + Task 13 |
| `run.log` per-file lines | Task 14 |
| Per-file try/except → continue loop | Task 14 |
| Outer + inner tqdm bars | Task 7 + Task 14 |
| 2D inputs skipped with warning | Task 13 (ndim check) |
| No new tests | Plan-wide (smoke run in Task 15 only) |

No gaps.

**Type/name consistency check:**
- `Args` dataclass fields ↔ `parse_args` return ↔ `process_file(args)` accesses: cross-checked in Task 2 / 13 / 14.
- DataFrame column names: detection emits `time, y, x` → fit adds `amp, x_fit, y_fit, sigma_x, sigma_y, bg` → mask adds `area, mean_intensity, max_intensity, spot_radius` → renamed `time→frame` for CSV → tracker adds `particle`. Consistent through Tasks 7, 8, 9, 10, 13.
- `_compute_track_properties` reads `frame, y, x, particle` — Task 13 uses the renamed-frame DataFrame, ✓.

**Placeholder scan:** No "TBD", "fill in", or hand-wavy steps. Every code-changing step shows the actual code.
