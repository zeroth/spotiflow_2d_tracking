from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FWHM_CONSTANT = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.355


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

        # Integer bounding box clipped to image
        y_lo = max(int(np.floor(cy - ry)), 0)
        y_hi = min(int(np.ceil(cy + ry)) + 1, mask.shape[0])
        x_lo = max(int(np.floor(cx - rx)), 0)
        x_hi = min(int(np.ceil(cx + rx)) + 1, mask.shape[1])

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dist_sq = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        inside = dist_sq <= 1.0
        mask[y_lo:y_hi, x_lo:x_hi][inside] = label


from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit


@dataclass
class FitAndMaskResult:
    """Combined result of fitting + mask painting for all spots in a frame."""

    fits: list[SpotFit2D]
    mask: np.ndarray  # uint16 labeled mask


def _gaussian_2d(yx, y0, x0, sigma_y, sigma_x, A, B):
    """2D Gaussian model function for curve_fit."""
    y, x = yx
    return A * np.exp(
        -((y - y0) ** 2 / (2 * sigma_y ** 2) + (x - x0) ** 2 / (2 * sigma_x ** 2))
    ) + B


def _fit_single_spot(image: np.ndarray, center_yx: tuple[int, int],
                      patch_radius: int) -> SpotFit2D:
    """Extract a patch and fit a 2D Gaussian to a single spot."""
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))
    r = patch_radius
    h, w = image.shape

    # Build coordinate grid for extraction (clipped to image)
    y_coords = np.arange(max(cy - r, 0), min(cy + r + 1, h))
    x_coords = np.arange(max(cx - r, 0), min(cx + r + 1, w))
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    patch = map_coordinates(image, [yy.ravel(), xx.ravel()], order=3, mode="reflect")
    patch = patch.reshape(yy.shape).astype(np.float64)

    # Normalize to [0, 1]
    p_min, p_max = patch.min(), patch.max()
    if p_max - p_min < 1e-10:
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)
    patch_norm = (patch - p_min) / (p_max - p_min)

    # Coordinate grid relative to patch center
    local_y = yy - cy
    local_x = xx - cx

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
        A = A_norm * intensity_range
        B = B_norm * intensity_range + p_min

        return SpotFit2D(y0, x0, sigma_y, sigma_x, A, B, success=True)
    except (RuntimeError, ValueError):
        return SpotFit2D(0, 0, 1.0, 1.0, 0, 0, success=False)


def fit_and_mask_2d(
    image: np.ndarray,
    points: np.ndarray,
    patch_radius: int = 4,
    fallback_radius: float = 2.0,
    progress_callback=None,
) -> FitAndMaskResult:
    """Fit 2D Gaussians to detected spots and paint labeled masks.

    Single-pass: each spot is fitted and immediately painted into the shared mask.

    Args:
        image: 2D image (Y, X).
        points: (N, 2) array of [y, x] spot coordinates.
        patch_radius: half-size of patch extracted around each spot for fitting.
        fallback_radius: radius for circular mask when fitting fails.
        progress_callback: optional callable(current, total) for progress updates.

    Returns:
        FitAndMaskResult with per-spot fits and uint16 labeled mask.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint16)
    fits: list[SpotFit2D] = []

    if len(points) == 0:
        return FitAndMaskResult(fits=fits, mask=mask)

    for i, (py, px) in enumerate(points):
        fit = _fit_single_spot(image, (py, px), patch_radius)
        fits.append(fit)

        label = i + 1
        cy, cx = int(round(py)), int(round(px))

        if fit.success:
            fit.paint_mask(mask, center_yx=(cy, cx), label=label)
        else:
            # Fallback: paint circular mask with fixed radius
            # Convert radius to sigma: radius = FWHM/2 = FWHM_CONSTANT * sigma / 2
            # => sigma = 2 * radius / FWHM_CONSTANT
            fallback_sigma = 2.0 * fallback_radius / FWHM_CONSTANT
            fallback = SpotFit2D(
                0, 0,
                sigma_y=fallback_sigma,
                sigma_x=fallback_sigma,
                amplitude=0, background=0, success=True,
            )
            fallback.paint_mask(mask, center_yx=(cy, cx), label=label)

        if progress_callback is not None:
            progress_callback(i + 1, len(points))

    return FitAndMaskResult(fits=fits, mask=mask)
