from __future__ import annotations

import numpy as np

_PRETRAINED_MODELS = {"general", "synth_complex", "hybiss", "fluo_live"}


def load_model(model_name: str, device: str = "cpu"):
    """Load a Spotiflow model (pretrained or from a custom folder path)."""
    from spotiflow.model import Spotiflow

    if model_name in _PRETRAINED_MODELS:
        return Spotiflow.from_pretrained(model_name, map_location=device)
    return Spotiflow.from_folder(model_name, map_location=device)


def detect_spots(
    image_2d: np.ndarray,
    model: Spotiflow,
    prob_thresh: float = 0.5,
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
