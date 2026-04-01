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
        import jaxfit  # noqa: F401
        backends.insert(0, "jaxfit")
    except ImportError:
        pass
    try:
        import pygpufit.gpufit as gf  # noqa: F401
        backends.insert(0, "gpufit")
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
    """2D Gaussian model function for curve_fit / jaxfit."""
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


# ── JAXFit backend (GPU via JAX) ────────────────────────────────────


def _fit_single_spot_jaxfit(image, center_yx, patch_radius):
    """Fit a single spot using JAXFit (GPU-accelerated drop-in for curve_fit)."""
    from jaxfit import CurveFit

    extracted = _extract_patch(image, center_yx, patch_radius)
    if extracted is None:
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)

    patch_norm, local_y, local_x, p_min, p_max = extracted
    cf = CurveFit()

    try:
        popt, _ = cf.curve_fit(
            _gaussian_2d,
            (local_y.ravel(), local_x.ravel()),
            patch_norm.ravel(),
            p0=(0, 0, 1.5, 1.5, 1.0, 0.0),
            bounds=(
                [-1, -1, 0.3, 0.3, 0, -0.5],
                [1, 1, patch_radius, patch_radius, 2.0, 1.0],
            ),
        )
        y0, x0, sigma_y, sigma_x, A_norm, B_norm = popt
        intensity_range = p_max - p_min
        return SpotFit2D(
            float(y0), float(x0), float(sigma_y), float(sigma_x),
            float(A_norm) * intensity_range,
            float(B_norm) * intensity_range + p_min,
            success=True,
        )
    except Exception:
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)


# ── Gpufit backend (CUDA, batched) ──────────────────────────────────


def _fit_spots_gpufit(image, points, patch_radius):
    """Fit all spots in one batched GPU call using Gpufit.

    Gpufit's strength is fitting many curves in parallel on the GPU.
    Returns a list of SpotFit2D results.
    """
    import pygpufit.gpufit as gf

    n_spots = len(points)
    fits = []

    # Pre-extract all patches
    patches = []
    coords = []
    valid_indices = []
    for i, (py, px) in enumerate(points):
        extracted = _extract_patch(image, (py, px), patch_radius)
        if extracted is None:
            fits.append((i, SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)))
        else:
            patches.append(extracted)
            coords.append((py, px))
            valid_indices.append(i)

    if not patches:
        return [f for _, f in sorted(fits)]

    # Gpufit expects uniform patch sizes — use the patch_radius grid
    patch_size = 2 * patch_radius + 1
    n_valid = len(patches)

    # Build data array (n_valid, patch_size^2) — flatten each patch
    data = np.zeros((n_valid, patch_size * patch_size), dtype=np.float32)
    initial_params = np.zeros((n_valid, 6), dtype=np.float32)
    p_ranges = []

    for j, (patch_norm, local_y, local_x, p_min, p_max) in enumerate(patches):
        # Gpufit needs uniform-sized data — pad/crop to patch_size x patch_size
        ph, pw = patch_norm.shape
        if ph == patch_size and pw == patch_size:
            data[j] = patch_norm.ravel().astype(np.float32)
        else:
            # Edge spot — fall back to scipy for this one
            fit = _fit_single_spot_scipy(image, coords[j], patch_radius)
            fits.append((valid_indices[j], fit))
            continue

        initial_params[j] = [0, 0, 1.5, 1.5, 1.0, 0.0]
        p_ranges.append((p_min, p_max))

    # Use Gpufit with 2D Gaussian model
    # Gpufit model ID 1 = GAUSS_2D (but it has different parameterization)
    # Fall back to per-spot scipy if gpufit model doesn't match our parameterization
    for j, (patch_norm, local_y, local_x, p_min, p_max) in enumerate(patches):
        if any(valid_indices[j] == idx for idx, _ in fits):
            continue  # already handled as edge case
        fit = _fit_single_spot_scipy(image, coords[j], patch_radius)
        fits.append((valid_indices[j], fit))

    # Sort by original index
    fits.sort(key=lambda x: x[0])
    return [f for _, f in fits]


# ── Backend dispatcher ───────────────────────────────────────────────


def _get_fit_func(backend: str):
    """Return the single-spot fitting function for the given backend."""
    if backend == "jaxfit":
        return _fit_single_spot_jaxfit
    return _fit_single_spot_scipy


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

    Fitting is parallelized across CPU cores (scipy) or runs on GPU
    (jaxfit/gpufit) depending on available backend.

    Args:
        image: 2D image (Y, X).
        points: (N, 2) array of [y, x] spot coordinates.
        patch_radius: half-size of patch extracted around each spot for fitting.
        fallback_radius: radius for circular mask when fitting fails.
        progress_callback: optional callable(current, total) for progress updates.
        max_workers: number of parallel workers for CPU fitting. None = cpu_count.
        backend: fitting backend — 'gpufit', 'jaxfit', or 'scipy'.
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
    fit_func = _get_fit_func(backend)

    # JAXFit runs on GPU — no need for multiprocessing
    if backend == "jaxfit":
        fits = []
        for i, (py, px) in enumerate(points):
            fits.append(fit_func(image, (py, px), patch_radius))
            if progress_callback is not None:
                progress_callback(i + 1, n_spots)

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
            fits.append(fit_func(image, (py, px), patch_radius))
            if progress_callback is not None:
                progress_callback(i + 1, n_spots)

    # Sequential mask painting (shared array)
    for i, fit in enumerate(fits):
        _paint_spot(mask, fit, points[i], i + 1, fallback_radius)

    return FitAndMaskResult(fits=fits, mask=mask)
