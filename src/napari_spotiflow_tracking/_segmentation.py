from __future__ import annotations

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log

_PRETRAINED_MODELS = {"general", "synth_complex", "hybiss", "fluo_live"}


# ── Spotiflow ────────────────────────────────────────────────────────


def load_model(model_name: str, device: str = "cpu"):
    """Load a Spotiflow model (pretrained or from a custom folder path)."""
    from spotiflow.model import Spotiflow

    if model_name in _PRETRAINED_MODELS:
        return Spotiflow.from_pretrained(model_name, map_location=device)
    return Spotiflow.from_folder(model_name, map_location=device)


def detect_spots(
    image_2d: np.ndarray,
    model,
    prob_thresh: float | None = 0.5,
    min_distance: int = 2,
) -> tuple[np.ndarray, object]:
    """Detect spots in a single 2D image using a loaded Spotiflow model.

    Returns (points, details) where points is (N, 2) array of [y, x].
    """
    points, details = model.predict(
        image_2d,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        verbose=False,
    )
    return points, details


# ── LoG blob detection ───────────────────────────────────────────────


def detect_spots_log(
    image_2d: np.ndarray,
    min_sigma: float = 2,
    max_sigma: float = 10,
    num_sigma: int = 10,
    threshold: float = 0.09,
) -> np.ndarray:
    """Detect spots using Laplacian of Gaussian (blob_log).

    Returns (N, 2) array of [y, x] coordinates.
    """
    im_norm = rescale_intensity(image_2d.astype(np.float64), out_range=(0, 1))
    blobs = blob_log(
        im_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=0.0,
        exclude_border=True,
    )
    if len(blobs) == 0:
        return np.empty((0, 2))
    # blob_log returns [y, x, sigma] — return only [y, x]
    return blobs[:, :2]


def sample_point_intensities(image_2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample image intensity at each point's nearest integer coords."""
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    ys = np.clip(np.rint(points[:, 0]).astype(int), 0, image_2d.shape[0] - 1)
    xs = np.clip(np.rint(points[:, 1]).astype(int), 0, image_2d.shape[1] - 1)
    return image_2d[ys, xs].astype(float)


# ── Fixed-radius point mask ──────────────────────────────────────────


def create_spot_mask(image_shape, points: np.ndarray, radius: int = 2) -> np.ndarray:
    """Paint a fixed-radius filled disk at each detected point.

    A simple, fit-free mask: every spot becomes a disk of the same ``radius``.
    For fitted, per-spot sized masks see ``_fitting.fit_and_mask_2d`` /
    ``_fitting.make_gpu_spot_mask``.

    Args:
        image_shape: shape ``(H, W)`` of the target 2D image.
        points: ``(N, 2+)`` array; the first two columns are taken as ``(y, x)``.
        radius: disk radius in pixels.

    Returns:
        A ``uint8`` mask of ``image_shape`` with 255 inside each disk and 0
        elsewhere.
    """
    from skimage.draw import disk

    mask = np.zeros(image_shape, dtype=np.uint8)
    for blob in points:
        y, x = blob[:2]
        rr, cc = disk((int(y), int(x)), radius, shape=image_shape)
        mask[rr, cc] = 255
    return mask
