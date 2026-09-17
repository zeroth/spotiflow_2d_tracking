"""Detection (LoG, fake Spotiflow), optional fit, and tracking stages."""
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch import detect
from napari_spotiflow_tracking.batch.config import (
    DetectionSettings,
    FitSettings,
    TrackingSettings,
)


def _blob_stack(n_frames=4, size=48, centres=((12, 12), (34, 30))):
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    for t in range(n_frames):
        for cy, cx in centres:
            stack[t] += 1000 * np.exp(-((yy - cy) ** 2 + (xx - (cx + t)) ** 2) / (2 * 2.0 ** 2))
    return stack.astype(np.uint16)


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        t = len(self.calls) - 1
        return np.array([[5.0, 6.0 + t], [20.0, 21.0]]), None


def test_log_detection_finds_blobs():
    settings = DetectionSettings(method="log")
    spots = detect.detect_stack(_blob_stack(), settings, model=None)
    assert list(spots.columns) == ["frame", "y", "x"]
    assert spots["frame"].dtype.kind == "i"
    first = spots[spots.frame == 0][["y", "x"]].to_numpy()
    for centre in ((12, 12), (34, 30)):
        assert np.min(np.hypot(*(first - centre).T)) < 2.0


def test_spotiflow_detection_uses_model_and_settings():
    model = FakeModel()
    settings = DetectionSettings()
    settings.spotiflow.prob_thresh = 0.4
    settings.spotiflow.min_distance = 3
    spots = detect.detect_stack(np.zeros((3, 32, 32), np.uint16), settings, model)
    assert len(spots) == 6
    assert spots["frame"].tolist() == [0, 0, 1, 1, 2, 2]
    assert model.calls[0] == {"prob_thresh": 0.4, "min_distance": 3, "verbose": False}


def test_empty_detection_has_columns():
    class Empty:
        def predict(self, image, **kwargs):
            return np.empty((0, 2)), None

    spots = detect.detect_stack(np.zeros((2, 8, 8)), DetectionSettings(), Empty())
    assert list(spots.columns) == ["frame", "y", "x"]
    assert len(spots) == 0


def test_load_detector_returns_none_for_log():
    assert detect.load_detector(DetectionSettings(method="log"), "cpu") is None


def test_tracking_links_filters_and_keeps_columns():
    rows = []
    for t in range(5):
        rows.append({"frame": t, "y": 10.0, "x": 10.0 + 0.5 * t, "amp": 1.0 + t})
        rows.append({"frame": t, "y": 40.0, "x": 40.0, "amp": 9.0})
    rows.append({"frame": 0, "y": 80.0, "x": 80.0, "amp": 0.0})  # one-frame spot
    spots = pd.DataFrame(rows)
    tracks = detect.track_spots(spots, TrackingSettings(search_range=3, memory=0,
                                                       min_track_length=3))
    assert tracks["particle"].nunique() == 2
    assert len(tracks) == 10
    assert "amp" in tracks.columns
    first = tracks[tracks.particle == tracks.particle.iloc[0]]
    assert first["frame"].tolist() == [0, 1, 2, 3, 4]


def test_tracking_empty_spots():
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    tracks = detect.track_spots(empty, TrackingSettings())
    assert "particle" in tracks.columns and len(tracks) == 0


def test_fit_empty_spots_returns_empty_mask():
    stack = np.zeros((2, 8, 8), np.uint16)
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    spots, mask = detect.fit_spots(stack, empty, FitSettings(enabled=True), "cpu")
    assert len(spots) == 0 and mask.shape == stack.shape and not mask.any()


def test_fit_spots_adds_fit_columns():
    pytest.importorskip("torchimize")
    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fitted, mask = detect.fit_spots(stack, spots, FitSettings(enabled=True), "cpu")
    assert {"frame", "y", "x", "sigma_x", "sigma_y", "spot_radius"} <= set(fitted.columns)
    assert mask.shape == stack.shape and mask.any()
