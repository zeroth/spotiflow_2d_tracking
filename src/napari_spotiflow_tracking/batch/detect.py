"""Detection, optional 2D Gaussian fit + masks, and trackpy linking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._segmentation import detect_spots, detect_spots_log
from .._tracking import track_blobs
from .config import DetectionSettings, FitSettings, TrackingSettings

SPOT_COLUMNS = ["frame", "y", "x"]


def torchimize_available() -> bool:
    try:
        import torchimize  # noqa: F401
    except ImportError:
        return False
    return True


def load_detector(detection: DetectionSettings, device: str):
    """Spotiflow model for ``method: spotiflow``; ``None`` for LoG."""
    if detection.method != "spotiflow":
        return None
    from .._segmentation import load_model

    return load_model(detection.spotiflow.model, device=device)


def detect_stack(stack: np.ndarray, detection: DetectionSettings, model) -> pd.DataFrame:
    frames, ys, xs = [], [], []
    for t in range(stack.shape[0]):
        if detection.method == "log":
            log = detection.log
            points = detect_spots_log(stack[t], min_sigma=log.min_sigma,
                                      max_sigma=log.max_sigma, num_sigma=log.num_sigma,
                                      threshold=log.threshold)
        else:
            sf = detection.spotiflow
            points, _ = detect_spots(stack[t], model, prob_thresh=sf.prob_thresh,
                                     min_distance=sf.min_distance)
        points = np.asarray(points)
        if len(points) == 0:
            continue
        frames.extend([t] * len(points))
        ys.extend(points[:, 0].astype(float))
        xs.extend(points[:, 1].astype(float))
    return pd.DataFrame({
        "frame": np.asarray(frames, dtype=np.int64),
        "y": np.asarray(ys, dtype=float),
        "x": np.asarray(xs, dtype=float),
    })


def fit_spots(stack: np.ndarray, spots: pd.DataFrame, fit: FitSettings, device: str):
    """Batched Gaussian fit + labelled mask, as scripts/bulk_track_analysis.py does.

    Returns consistent columns even for empty input: frame, y, x, amp, x_fit, y_fit,
    sigma_x, sigma_y, bg, area, mean_intensity, max_intensity, spot_radius (from
    fit_all_frames_gpu and make_gpu_spot_mask).
    """
    from .._fitting import fit_all_frames_gpu, make_gpu_spot_mask

    fitted = fit_all_frames_gpu(stack, spots.rename(columns={"frame": "time"}),
                                win=fit.window, device=device)
    mask, props = make_gpu_spot_mask(stack.shape, fitted, intensity_stack=stack,
                                     max_radius=fit.max_radius)
    return props.rename(columns={"time": "frame"}).reset_index(drop=True), mask


def track_spots(spots: pd.DataFrame, tracking: TrackingSettings) -> pd.DataFrame:
    if len(spots) == 0:
        return spots.assign(particle=pd.Series(dtype=np.int64))
    tracked = track_blobs(spots[SPOT_COLUMNS].copy(), search_range=tracking.search_range,
                          memory=tracking.memory)
    extra = spots.drop(columns=SPOT_COLUMNS)
    if len(extra.columns):
        tracked = tracked.join(extra)  # trackpy keeps the input index
    lengths = tracked.groupby("particle")["frame"].transform("size")
    tracked = tracked[lengths >= tracking.min_track_length]
    return tracked.sort_values(["particle", "frame"]).reset_index(drop=True)
