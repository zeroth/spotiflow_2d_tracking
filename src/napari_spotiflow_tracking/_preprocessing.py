from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def remove_background(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Remove background by subtracting a Gaussian low-pass filtered image.

    Same approach as spotiflow's BackgroundRemover: subtracts a blurred
    version of the image to remove low-frequency background while preserving
    spot-scale features.

    Args:
        image: 2D image (Y, X).
        sigma: standard deviation of the Gaussian filter. Larger values
               remove broader background structures.

    Returns:
        Background-subtracted image.
    """
    background = gaussian_filter(image.astype(np.float64), sigma=sigma)
    return image - background


def walking_average(image_stack: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a walking (rolling) average along the time axis.

    For a T,Y,X stack, each frame is replaced by the mean of the surrounding
    *window* frames (centered). Frames near the edges use a truncated window.

    Args:
        image_stack: 3D array (T, Y, X).
        window: number of frames in the averaging window (must be odd).

    Returns:
        Smoothed stack with same shape and dtype as input.
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

    # Cumulative sum for efficient rolling mean
    cumsum = np.cumsum(image_stack.astype(np.float64), axis=0)
    # Prepend zeros for the cumsum trick
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)

    for t in range(n_frames):
        lo = max(t - half, 0)
        hi = min(t + half + 1, n_frames)
        result[t] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result.astype(image_stack.dtype)
