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
        backends.insert(0, "torchimize")
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


def _fit_spots_torchimize(image, points, patch_radius):
    """Fit all spots in one batched GPU call using torchimize.

    Uses lsq_lma_parallel to fit all spots simultaneously on GPU/CPU.
    Returns a list of SpotFit2D results.
    """
    import torch
    from torchimize.functions import lsq_lma_parallel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract all patches
    extractions = []
    for py, px in points:
        extractions.append(_extract_patch(image, (py, px), patch_radius))

    # Separate valid from invalid
    valid_indices = [i for i, e in enumerate(extractions) if e is not None]
    if not valid_indices:
        return [SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)] * len(points)

    # Build uniform-sized coordinate grids and data arrays
    patch_size = 2 * patch_radius + 1
    n_valid = len(valid_indices)
    n_pixels = patch_size * patch_size

    # Pre-allocate tensors
    data_tensor = torch.zeros(n_valid, n_pixels, dtype=torch.float32, device=device)
    y_tensor = torch.zeros(n_valid, n_pixels, dtype=torch.float32, device=device)
    x_tensor = torch.zeros(n_valid, n_pixels, dtype=torch.float32, device=device)
    p_mins = np.zeros(n_valid)
    p_maxs = np.zeros(n_valid)

    # Track which valid spots have full-size patches
    batch_indices = []  # indices into valid_indices that go into GPU batch

    for j, vi in enumerate(valid_indices):
        patch_norm, local_y, local_x, p_min, p_max = extractions[vi]
        if patch_norm.shape == (patch_size, patch_size):
            data_tensor[j] = torch.from_numpy(patch_norm.ravel().astype(np.float32))
            y_tensor[j] = torch.from_numpy(local_y.ravel().astype(np.float32))
            x_tensor[j] = torch.from_numpy(local_x.ravel().astype(np.float32))
            p_mins[j] = p_min
            p_maxs[j] = p_max
            batch_indices.append(j)

    if not batch_indices:
        # All valid spots are edge cases — fall back to scipy
        fits = [SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)] * len(points)
        for vi in valid_indices:
            fits[vi] = _fit_single_spot_scipy(image, points[vi], patch_radius)
        return fits

    n_batch = len(batch_indices)
    batch_data = data_tensor[batch_indices]
    batch_y = y_tensor[batch_indices]
    batch_x = x_tensor[batch_indices]

    # Initial parameters: [y0, x0, sigma_y, sigma_x, A, B]
    p_init = torch.tensor(
        [[0, 0, 1.5, 1.5, 1.0, 0.0]] * n_batch,
        dtype=torch.float32, device=device,
    )

    # Residual function for batched LM
    def residuals(p):
        # p: [batch, 6]
        y0 = p[:, 0:1]       # [batch, 1]
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
        res = (model - batch_data).unsqueeze(-1)  # [batch, pixels, 1]
        return res

    try:
        result_list = lsq_lma_parallel(p_init, residuals, max_iter=100)
        final_params = result_list[-1].cpu().numpy()  # [n_batch, 6]
    except Exception:
        # Fall back to scipy for all
        fits = [SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)] * len(points)
        for vi in valid_indices:
            fits[vi] = _fit_single_spot_scipy(image, points[vi], patch_radius)
        return fits

    # Build results
    fits = [SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)] * len(points)

    for k, bi in enumerate(batch_indices):
        vi = valid_indices[bi]
        y0, x0, sigma_y, sigma_x, A_norm, B_norm = final_params[k]
        sigma_y = max(abs(sigma_y), 0.3)
        sigma_x = max(abs(sigma_x), 0.3)
        intensity_range = p_maxs[bi] - p_mins[bi]

        fits[vi] = SpotFit2D(
            float(y0), float(x0),
            float(sigma_y), float(sigma_x),
            float(A_norm * intensity_range),
            float(B_norm * intensity_range + p_mins[bi]),
            success=True,
        )

    # Handle edge spots (non-full-size patches) with scipy fallback
    handled = set(valid_indices[bi] for bi in batch_indices)
    for vi in valid_indices:
        if vi not in handled:
            fits[vi] = _fit_single_spot_scipy(image, points[vi], patch_radius)

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

    # Torchimize — batched GPU fitting
    if backend == "torchimize":
        fits = _fit_spots_torchimize(image, points, patch_radius)
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
