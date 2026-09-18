"""Detection, 2D Gaussian fit + masks, and trackpy linking."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._segmentation import detect_spots, detect_spots_log
from .._tracking import track_blobs
from .config import DetectionSettings, FitSettings, TrackingSettings

log = logging.getLogger("napari_spotiflow_tracking.batch")

SPOT_COLUMNS = ["frame", "y", "x"]


def torchimize_available() -> bool:
    try:
        import torchimize  # noqa: F401
    except ImportError:
        return False
    return True


def fit_backend() -> str:
    """The fitting backend in use: batched torchimize when importable, else scipy."""
    return "torchimize" if torchimize_available() else "scipy"


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


def _fit_frames_scipy(stack: np.ndarray, df: pd.DataFrame, fit: FitSettings) -> pd.DataFrame:
    """Per-frame scipy fit producing the columns ``fit_all_frames_gpu`` produces.

    Slower than the batched torchimize path, but it needs no torch — so fitting
    can be on by default on any install. Column names and order match exactly,
    because ``make_gpu_spot_mask`` consumes them and spots.csv must have one
    schema regardless of backend. A failed fit gives NaN in the fit columns,
    same as the GPU path.
    """
    from .._fitting import _fit_single_spot_scipy

    patch_radius = fit.window * 2
    n = len(df)
    amp = np.full(n, np.nan, dtype=np.float64)
    x_fit = np.full(n, np.nan, dtype=np.float64)
    y_fit = np.full(n, np.nan, dtype=np.float64)
    sigma_x = np.full(n, np.nan, dtype=np.float64)
    sigma_y = np.full(n, np.nan, dtype=np.float64)
    bg = np.full(n, np.nan, dtype=np.float64)

    ys = df["y"].to_numpy()
    xs = df["x"].to_numpy()
    ts = df["time"].to_numpy().astype(np.int64)
    for i in range(n):
        result = _fit_single_spot_scipy(stack[ts[i]], (ys[i], xs[i]), patch_radius)
        if not result.success:
            continue
        # Same coordinate convention as fit_all_frames_gpu: the patch is centred
        # on the *rounded* spot position, and the fit's local offset (y0, x0) is
        # added back to it to get an absolute coordinate.
        amp[i] = result.amplitude
        x_fit[i] = round(xs[i]) + result.x0
        y_fit[i] = round(ys[i]) + result.y0
        sigma_x[i] = abs(result.sigma_x)
        sigma_y[i] = abs(result.sigma_y)
        bg[i] = result.background

    return df.assign(amp=amp, x_fit=x_fit, y_fit=y_fit,
                     sigma_x=sigma_x, sigma_y=sigma_y, bg=bg)


def fit_spots(stack: np.ndarray, spots: pd.DataFrame, fit: FitSettings, device: str):
    """Gaussian fit + labelled mask, as scripts/bulk_track_analysis.py does.

    Uses the batched torchimize backend when importable, else a per-frame scipy
    fallback (``_fit_frames_scipy``) — see ``fit_backend()``. Both feed the same
    ``make_gpu_spot_mask``, so spots.csv has one schema regardless of backend.
    If the batched CPU solver itself blows up on a pathological batch (seen on
    some CPU LAPACK builds as a ``RuntimeError`` from ``torch.linalg.lstsq``),
    this movie's fit degrades to the scipy backend rather than failing the run.

    Returns consistent columns even for empty input: frame, y, x, amp, x_fit, y_fit,
    sigma_x, sigma_y, bg, area, mean_intensity, max_intensity, spot_radius.
    """
    from .._fitting import fit_all_frames_gpu, make_gpu_spot_mask

    backend = fit_backend()
    log.info("fitting %d spot(s) with the %s backend", len(spots), backend)
    df = spots.rename(columns={"frame": "time"})
    if backend == "torchimize":
        try:
            fitted = fit_all_frames_gpu(stack, df, win=fit.window, device=device)
        except RuntimeError as exc:
            log.warning("torchimize fit failed (%s); falling back to the scipy "
                       "backend for this movie", exc)
            fitted = _fit_frames_scipy(stack, df, fit)
    else:
        fitted = _fit_frames_scipy(stack, df, fit)
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
