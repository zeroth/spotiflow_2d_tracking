from __future__ import annotations

import numpy as np
from skimage import morphology
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


# ── Background removal ───────────────────────────────────────────────


def remove_background(image: np.ndarray, disk_size: int = 10) -> np.ndarray:
    """Remove background using morphological reconstruction.

    Erodes the image with a disk structuring element, reconstructs by
    dilation, and subtracts the result from the original.
    """
    seed = morphology.erosion(image, morphology.disk(disk_size))
    background = morphology.reconstruction(seed, image, method="dilation")
    return image - background


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
