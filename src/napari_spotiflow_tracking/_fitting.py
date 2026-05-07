from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

FWHM_CONSTANT = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.355


# ── Detect available fitting backends ────────────────────────────────

def _available_backends() -> list[str]:
    """Return list of available fitting backends, best first."""
    backends = ["scipy"]
    try:
        import torchimize  # noqa: F401
        backends.append("torchimize")
    except ImportError:
        pass
    return backends


def get_best_backend() -> str:
    """Return the best available fitting backend name."""
    return _available_backends()[0]


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class SpotFit2D:
    """Result of fitting a 2D Gaussian to a single spot."""

    y0: float
    x0: float
    sigma_y: float
    sigma_x: float
    amplitude: float
    background: float
    success: bool

    @property
    def fwhm_y(self) -> float:
        return FWHM_CONSTANT * self.sigma_y

    @property
    def fwhm_x(self) -> float:
        return FWHM_CONSTANT * self.sigma_x

    def paint_mask(
        self, mask: np.ndarray, center_yx: tuple[int, int], label: int
    ) -> None:
        """Paint an elliptical region into *mask* at the given center."""
        if not self.success:
            return

        cy, cx = center_yx
        ry = self.fwhm_y / 2.0
        rx = self.fwhm_x / 2.0

        y_lo = max(int(np.floor(cy - ry)), 0)
        y_hi = min(int(np.ceil(cy + ry)) + 1, mask.shape[0])
        x_lo = max(int(np.floor(cx - rx)), 0)
        x_hi = min(int(np.ceil(cx + rx)) + 1, mask.shape[1])

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dist_sq = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        inside = dist_sq <= 1.0
        mask[y_lo:y_hi, x_lo:x_hi][inside] = label


@dataclass
class FitAndMaskResult:
    """Combined result of fitting + mask painting for all spots in a frame."""

    fits: list[SpotFit2D]
    mask: np.ndarray  # uint16 labeled mask


# ── 2D Gaussian model ───────────────────────────────────────────────


def _gaussian_2d(yx, y0, x0, sigma_y, sigma_x, A, B):
    """2D Gaussian model function for curve_fit."""
    y, x = yx
    return A * np.exp(
        -((y - y0) ** 2 / (2 * sigma_y ** 2) + (x - x0) ** 2 / (2 * sigma_x ** 2))
    ) + B


# ── Patch extraction (shared by all backends) ───────────────────────


def _extract_patch(image, center_yx, patch_radius):
    """Extract and normalize a patch around a spot center.

    Returns (patch_norm, local_y, local_x, p_min, p_max) or None if flat.
    """
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))
    r = patch_radius
    h, w = image.shape

    y_coords = np.arange(max(cy - r, 0), min(cy + r + 1, h))
    x_coords = np.arange(max(cx - r, 0), min(cx + r + 1, w))
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    patch = map_coordinates(image, [yy.ravel(), xx.ravel()], order=3, mode="reflect")
    patch = patch.reshape(yy.shape).astype(np.float64)

    p_min, p_max = patch.min(), patch.max()
    if p_max - p_min < 1e-10:
        return None

    patch_norm = (patch - p_min) / (p_max - p_min)
    local_y = yy - cy
    local_x = xx - cx
    return patch_norm, local_y, local_x, p_min, p_max


# ── Scipy backend (default) ─────────────────────────────────────────


def _fit_single_spot_scipy(image, center_yx, patch_radius):
    """Fit a single spot using scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit

    extracted = _extract_patch(image, center_yx, patch_radius)
    if extracted is None:
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)

    patch_norm, local_y, local_x, p_min, p_max = extracted

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            (local_y.ravel(), local_x.ravel()),
            patch_norm.ravel(),
            p0=(0, 0, 1.5, 1.5, 1.0, 0.0),
            bounds=(
                [-1, -1, 0.3, 0.3, 0, -0.5],
                [1, 1, patch_radius, patch_radius, 2.0, 1.0],
            ),
            maxfev=1000,
        )
        y0, x0, sigma_y, sigma_x, A_norm, B_norm = popt
        intensity_range = p_max - p_min
        return SpotFit2D(
            y0, x0, sigma_y, sigma_x,
            A_norm * intensity_range,
            B_norm * intensity_range + p_min,
            success=True,
        )
    except (RuntimeError, ValueError):
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)


def _fit_single_spot_scipy_args(args):
    """Picklable wrapper for ProcessPoolExecutor."""
    return _fit_single_spot_scipy(*args)


# ── Torchimize backend (GPU batched LM) ─────────────────────────────


def _extract_patch_args(args):
    """Picklable wrapper for parallel patch extraction."""
    return _extract_patch(*args)


def _fit_spots_torchimize(image, points, patch_radius, max_workers=None):
    """Fit all spots using parallel extraction + batched GPU LM fitting.

    1. Extract patches in parallel across CPU cores
    2. Build batch tensors (numpy vectorized → single GPU transfer)
    3. Fit all spots in one batched lsq_lma_parallel call on GPU
    4. Edge spots (near image border) fall back to scipy

    Returns a list of SpotFit2D results.
    """
    import torch
    from torchimize.functions import lsq_lma_parallel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_spots = len(points)
    patch_size = 2 * patch_radius + 1
    n_pixels = patch_size * patch_size

    # ── Step 1: Parallel patch extraction ────────────────────────────
    args_list = [(image, (py, px), patch_radius) for py, px in points]
    if n_spots > 20 and max_workers != 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            extractions = list(executor.map(_extract_patch_args, args_list))
    else:
        extractions = [_extract_patch(image, (py, px), patch_radius) for py, px in points]

    # ── Step 2: Build batch arrays (numpy, one GPU transfer) ─────────
    valid_full = []  # (spot_index, extraction) for full-size patches
    valid_edge = []  # spot indices for edge patches (scipy fallback)

    for i, ext in enumerate(extractions):
        if ext is None:
            continue
        patch_norm = ext[0]
        if patch_norm.shape == (patch_size, patch_size):
            valid_full.append((i, ext))
        else:
            valid_edge.append(i)

    fits = [SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)] * n_spots

    if not valid_full:
        # All edge cases — scipy fallback
        for i in valid_edge:
            fits[i] = _fit_single_spot_scipy(image, points[i], patch_radius)
        return fits

    n_batch = len(valid_full)

    # Pre-allocate numpy arrays, fill vectorized, single transfer to GPU
    data_np = np.empty((n_batch, n_pixels), dtype=np.float32)
    y_np = np.empty((n_batch, n_pixels), dtype=np.float32)
    x_np = np.empty((n_batch, n_pixels), dtype=np.float32)
    p_mins = np.empty(n_batch, dtype=np.float64)
    p_maxs = np.empty(n_batch, dtype=np.float64)
    spot_indices = np.empty(n_batch, dtype=np.intp)

    for k, (si, (patch_norm, local_y, local_x, p_min, p_max)) in enumerate(valid_full):
        data_np[k] = patch_norm.ravel()
        y_np[k] = local_y.ravel()
        x_np[k] = local_x.ravel()
        p_mins[k] = p_min
        p_maxs[k] = p_max
        spot_indices[k] = si

    # Single bulk transfer to GPU
    batch_data = torch.from_numpy(data_np).to(device)
    batch_y = torch.from_numpy(y_np).to(device)
    batch_x = torch.from_numpy(x_np).to(device)

    # ── Step 3: Batched GPU fitting ──────────────────────────────────
    p_init = torch.zeros(n_batch, 6, dtype=torch.float32, device=device)
    p_init[:, 2] = 1.5   # sigma_y
    p_init[:, 3] = 1.5   # sigma_x
    p_init[:, 4] = 1.0   # amplitude

    def residuals(p):
        y0 = p[:, 0:1]
        x0 = p[:, 1:2]
        sy = p[:, 2:3].clamp(min=0.3)
        sx = p[:, 3:4].clamp(min=0.3)
        A = p[:, 4:5].clamp(min=0)
        B = p[:, 5:6]

        dy = batch_y - y0
        dx = batch_x - x0
        model = A * torch.exp(
            -(dy ** 2 / (2 * sy ** 2) + dx ** 2 / (2 * sx ** 2))
        ) + B
        return (model - batch_data).unsqueeze(-1)

    try:
        result_list = lsq_lma_parallel(p_init, residuals, max_iter=50)
        final_params = result_list[-1].cpu().numpy()  # [n_batch, 6]
    except Exception:
        # Total fallback to scipy
        for k, (si, _) in enumerate(valid_full):
            fits[si] = _fit_single_spot_scipy(image, points[si], patch_radius)
        for si in valid_edge:
            fits[si] = _fit_single_spot_scipy(image, points[si], patch_radius)
        return fits

    # ── Step 4: Build SpotFit2D results ──────────────────────────────
    intensity_ranges = p_maxs - p_mins

    for k in range(n_batch):
        si = spot_indices[k]
        y0, x0, sigma_y, sigma_x, A_norm, B_norm = final_params[k]
        fits[si] = SpotFit2D(
            float(y0), float(x0),
            max(abs(float(sigma_y)), 0.3),
            max(abs(float(sigma_x)), 0.3),
            float(A_norm * intensity_ranges[k]),
            float(B_norm * intensity_ranges[k] + p_mins[k]),
            success=True,
        )

    # Edge spots — scipy fallback (parallel)
    if valid_edge:
        edge_args = [(image, points[si], patch_radius) for si in valid_edge]
        if len(valid_edge) > 5 and max_workers != 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for si, fit in zip(valid_edge, executor.map(_fit_single_spot_scipy_args, edge_args)):
                    fits[si] = fit
        else:
            for si in valid_edge:
                fits[si] = _fit_single_spot_scipy(image, points[si], patch_radius)

    return fits


# ── Mask painting ────────────────────────────────────────────────────


def _paint_spot(mask, fit, center_yx, label, fallback_radius):
    """Paint a single spot into the shared mask."""
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))
    if fit.success:
        fit.paint_mask(mask, center_yx=(cy, cx), label=label)
    else:
        fallback_sigma = 2.0 * fallback_radius / FWHM_CONSTANT
        fallback = SpotFit2D(
            0, 0,
            sigma_y=fallback_sigma,
            sigma_x=fallback_sigma,
            amplitude=0, background=0, success=True,
        )
        fallback.paint_mask(mask, center_yx=(cy, cx), label=label)


# ── Main API ─────────────────────────────────────────────────────────


def fit_and_mask_2d(
    image: np.ndarray,
    points: np.ndarray,
    patch_radius: int = 4,
    fallback_radius: float = 2.0,
    progress_callback=None,
    max_workers: int | None = None,
    backend: str | None = None,
) -> FitAndMaskResult:
    """Fit 2D Gaussians to detected spots and paint labeled masks.

    Args:
        image: 2D image (Y, X).
        points: (N, 2) array of [y, x] spot coordinates.
        patch_radius: half-size of patch extracted around each spot for fitting.
        fallback_radius: radius for circular mask when fitting fails.
        progress_callback: optional callable(current, total) for progress updates.
        max_workers: number of parallel workers for CPU fitting. None = cpu_count.
        backend: fitting backend — 'torchimize' or 'scipy'.
                 None = auto-detect best available.

    Returns:
        FitAndMaskResult with per-spot fits and uint16 labeled mask.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint16)

    if len(points) == 0:
        return FitAndMaskResult(fits=[], mask=mask)

    if backend is None:
        backend = get_best_backend()

    n_spots = len(points)

    # Torchimize — parallel extraction + batched GPU fitting
    if backend == "torchimize":
        fits = _fit_spots_torchimize(image, points, patch_radius, max_workers=max_workers)
        if progress_callback is not None:
            progress_callback(n_spots, n_spots)

    # Scipy — parallel via ProcessPoolExecutor
    elif n_spots > 10 and max_workers != 1:
        args_list = [(image, (py, px), patch_radius) for py, px in points]
        fits = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, fit in enumerate(executor.map(_fit_single_spot_scipy_args, args_list)):
                fits.append(fit)
                if progress_callback is not None:
                    progress_callback(i + 1, n_spots)
    else:
        fits = []
        for i, (py, px) in enumerate(points):
            fits.append(_fit_single_spot_scipy(image, (py, px), patch_radius))
            if progress_callback is not None:
                progress_callback(i + 1, n_spots)

    # Sequential mask painting (shared array)
    for i, fit in enumerate(fits):
        _paint_spot(mask, fit, points[i], i + 1, fallback_radius)

    return FitAndMaskResult(fits=fits, mask=mask)


# ── GPU-batched 2D Gaussian fitting + mask generation ──────────────
# Adapted from point_fitting.py at the project root. Copied (not imported)
# so the package is self-contained. The mask is a (T, Y, X) stack of
# independent 2D labelings per frame — labels do NOT connect across time.

try:
    import torch as _torch
    from torchimize.functions import lsq_lma_parallel as _lsq_lma_parallel
    HAS_GPU_FIT = _torch.cuda.is_available()
except ImportError:
    HAS_GPU_FIT = False


def _gaussian_2d_batched(p, xy):
    amp = p[:, 0:1]; x0 = p[:, 1:2]; y0 = p[:, 2:3]
    sx = p[:, 3:4]; sy = p[:, 4:5]; bg = p[:, 5:6]
    x = xy[0:1, :]; y = xy[1:2, :]
    dx = (x - x0) / sx
    dy = (y - y0) / sy
    out = amp * _torch.exp(-0.5 * (dx * dx + dy * dy)) + bg
    return out.unsqueeze(1)


def _residuals(p, data, xy):
    return _gaussian_2d_batched(p, xy) - data


def _extract_crops_vec(stack, df, win):
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


def fit_all_frames_gpu(
    stack: np.ndarray,
    df,
    win: int = 2,
    device: str = "cuda",
    max_iter: int = 50,
    chunk_size: int = 20000,
):
    """Batch 2D Gaussian fit across an entire (T, Y, X) stack via torchimize.

    ``df`` must have columns ``time, y, x``. Output adds
    ``amp, x_fit, y_fit, sigma_x, sigma_y, bg``.
    """
    import pandas as pd

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

    yy, xx = _torch.meshgrid(
        _torch.arange(-win, win + 1, dtype=_torch.float32),
        _torch.arange(-win, win + 1, dtype=_torch.float32),
        indexing="ij",
    )
    xy = _torch.stack([xx.flatten(), yy.flatten()]).to(device)
    wvec = _torch.ones(1, device=device)

    final_all = np.empty((N, 6), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = crops[start:end]
        B = len(chunk)

        data = _torch.from_numpy(chunk.reshape(B, 1, M)).to(device)
        d2 = data[:, 0, :]
        amp0 = d2.max(1).values - d2.min(1).values
        bg0 = d2.min(1).values
        init = _torch.stack([
            amp0,
            _torch.zeros(B, device=device),
            _torch.zeros(B, device=device),
            _torch.full((B,), 1.5, device=device),
            _torch.full((B,), 1.5, device=device),
            bg0,
        ], dim=1)

        coeffs_list = _lsq_lma_parallel(
            p=init, function=_residuals, args=(data, xy), wvec=wvec,
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


def make_gpu_spot_mask(
    stack_shape,
    df,
    intensity_stack=None,
    dtype=np.int32,
    min_radius: float = 1.0,
    max_radius: float | None = None,
    label_col: str | None = None,
):
    """Build a ``(T, Y, X)`` stack of per-frame 2D labels from fitted sigmas.

    Each frame in the output is an independent 2D labels image (labels are
    not connected across time). When ``intensity_stack`` is given, also
    computes per-spot ``area, mean_intensity, max_intensity, spot_radius``.

    The output column names match our regionprops conventions
    (``area``, ``mean_intensity``, ``max_intensity``) so the FilterPanel
    whitelist picks them up directly. ``spot_radius`` is kept under that
    name (no regionprops equivalent).
    """
    import pandas as pd

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
            area=area_out,
            mean_intensity=mean_out,
            max_intensity=max_out,
            spot_radius=radii_out,
        )

    cx = sub["x"].to_numpy()
    cy = sub["y"].to_numpy()
    ct = sub["time"].to_numpy().astype(np.int64) - 1
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
        area=area_out,
        mean_intensity=mean_out,
        max_intensity=max_out,
        spot_radius=radii_out,
    )
    return mask, df_out
