"""Detection (LoG, fake Spotiflow), optional fit, and tracking stages."""
import logging

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


def test_tracking_links_and_keeps_columns():
    rows = []
    for t in range(5):
        rows.append({"frame": t, "y": 10.0, "x": 10.0 + 0.5 * t, "amp": 1.0 + t})
        rows.append({"frame": t, "y": 40.0, "x": 40.0, "amp": 9.0})
    spots = pd.DataFrame(rows)
    tracks = detect.track_spots(spots, TrackingSettings(search_range=3, memory=0))
    assert tracks["particle"].nunique() == 2
    assert len(tracks) == 10
    assert "amp" in tracks.columns
    first = tracks[tracks.particle == tracks.particle.iloc[0]]
    assert first["frame"].tolist() == [0, 1, 2, 3, 4]


def test_tracking_keeps_short_tracks():
    """tracks.csv is unfiltered: a one-frame spot survives as its own track."""
    rows = []
    for t in range(5):
        rows.append({"frame": t, "y": 10.0, "x": 10.0 + 0.5 * t, "amp": 1.0})
    rows.append({"frame": 0, "y": 80.0, "x": 80.0, "amp": 0.0})   # one-frame spot
    tracks = detect.track_spots(pd.DataFrame(rows),
                                TrackingSettings(search_range=3, memory=0))
    assert tracks["particle"].nunique() == 2
    assert len(tracks) == 6
    lengths = sorted(tracks.groupby("particle").size())
    assert lengths == [1, 5]


def test_tracking_empty_spots():
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    tracks = detect.track_spots(empty, TrackingSettings())
    assert "particle" in tracks.columns and len(tracks) == 0


def test_fit_empty_spots_returns_empty_mask():
    pytest.importorskip("torchimize")
    stack = np.zeros((2, 8, 8), np.uint16)
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    empty_spots, empty_mask = detect.fit_spots(stack, empty, FitSettings(enabled=True), "cpu")

    # Verify empty result has same columns as non-empty result
    blob_stack = _blob_stack(n_frames=2, size=8)
    blob_spots = detect.detect_stack(blob_stack, DetectionSettings(method="log"), model=None)
    fitted_spots, _ = detect.fit_spots(blob_stack, blob_spots, FitSettings(enabled=True), "cpu")

    assert len(empty_spots) == 0 and empty_mask.shape == stack.shape and not empty_mask.any()
    assert list(empty_spots.columns) == list(fitted_spots.columns)


def test_fit_spots_adds_fit_columns():
    pytest.importorskip("torchimize")
    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fitted, mask = detect.fit_spots(stack, spots, FitSettings(enabled=True), "cpu")
    assert {"frame", "y", "x", "sigma_x", "sigma_y", "spot_radius"} <= set(fitted.columns)
    assert mask.shape == stack.shape and mask.any()


def test_fit_backend_prefers_torchimize_when_available(monkeypatch):
    monkeypatch.setattr(detect, "torchimize_available", lambda: True)
    assert detect.fit_backend("cuda") == "torchimize"
    monkeypatch.setattr(detect, "torchimize_available", lambda: False)
    assert detect.fit_backend("cuda") == "scipy"
    # cpu always uses the scipy fallback: the batched solver's CPU LM path has
    # a known numerical-stability issue (see fit_spots), so it is never tried.
    monkeypatch.setattr(detect, "torchimize_available", lambda: True)
    assert detect.fit_backend("cpu") == "scipy"


def test_scipy_fit_produces_the_same_columns_as_torchimize(monkeypatch):
    """The fallback must not change spots.csv's schema.

    fit_backend forces scipy on device="cpu" regardless of torchimize's
    availability, so the two backends are selected directly here rather than
    through the device argument.
    """
    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fit = FitSettings(enabled=True)
    monkeypatch.setattr(detect, "fit_backend", lambda device: "torchimize")
    gpu_spots, gpu_mask = detect.fit_spots(stack, spots, fit, "cpu")
    monkeypatch.setattr(detect, "fit_backend", lambda device: "scipy")
    cpu_spots, cpu_mask = detect.fit_spots(stack, spots, fit, "cpu")
    assert list(cpu_spots.columns) == list(gpu_spots.columns)
    assert len(cpu_spots) == len(gpu_spots)
    assert cpu_mask.shape == gpu_mask.shape and cpu_mask.dtype == gpu_mask.dtype
    assert cpu_mask.any()


def test_scipy_fit_recovers_blob_positions(monkeypatch):
    monkeypatch.setattr(detect, "torchimize_available", lambda: False)
    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fitted, _ = detect.fit_spots(stack, spots, FitSettings(enabled=True), "cpu")
    ok = fitted[["x_fit", "y_fit"]].notna().all(axis=1)
    assert ok.sum() >= len(fitted) - 1
    assert (fitted.loc[ok, "sigma_x"] > 0).all()


def test_scipy_and_torchimize_fits_agree_closely():
    """Same support and border rule (patch_radius = fit.window on both), so the
    backends must agree on which spots are NaN, and fit values must be close.
    Not bit-identical: the scipy fit bounds its search, the batched LM does not.

    One of the three synthetic spots sits at row 1, inside the ``window``-pixel
    border margin (window=2, so rows/cols 0-1 and 46-47 are excluded — see
    ``_extract_crops_vec``). It has a real, strong Gaussian peak there (not a
    flat patch), so a scipy backend that forgot the border rule would still
    fit it successfully instead of leaving it NaN like the batched path always
    does — making the NaN-parity check below fail, rather than pass vacuously
    the way it would if every spot were far from an edge.
    """
    pytest.importorskip("torchimize")
    from napari_spotiflow_tracking._fitting import fit_all_frames_gpu

    win = 2
    stack = _blob_stack(n_frames=1, size=48, centres=((12, 12), (34, 30), (1, 24)))
    df = pd.DataFrame({"time": [0, 0, 0], "y": [12.0, 34.0, 1.0], "x": [12.0, 30.0, 24.0]})
    fit = FitSettings(enabled=True, window=win)

    gpu = fit_all_frames_gpu(stack, df, win=win, device="cpu")
    cpu = detect._fit_frames_scipy(stack, df, fit)

    gpu_nan = gpu[["x_fit", "y_fit"]].isna().any(axis=1)
    cpu_nan = cpu[["x_fit", "y_fit"]].isna().any(axis=1)
    assert (gpu_nan == cpu_nan).all()
    # The border rule must actually have been exercised: some spots NaN, some not.
    assert cpu_nan.any() and not cpu_nan.all()
    assert cpu_nan.iloc[2]  # the row-1 spot specifically, not some other one

    ok = ~gpu_nan
    assert ok.any()
    assert np.allclose(gpu.loc[ok, "x_fit"], cpu.loc[ok, "x_fit"], atol=0.25)
    assert np.allclose(gpu.loc[ok, "y_fit"], cpu.loc[ok, "y_fit"], atol=0.25)
    assert np.allclose(gpu.loc[ok, "sigma_x"], cpu.loc[ok, "sigma_x"], rtol=0.2)
    assert np.allclose(gpu.loc[ok, "sigma_y"], cpu.loc[ok, "sigma_y"], rtol=0.2)


def test_scipy_fit_requests_patch_radius_equal_to_window(monkeypatch):
    """Pins the exact line the support finding is about: the scipy backend
    must call _fit_single_spot_scipy with patch_radius == fit.window (matching
    the batched path's win), not fit.window * 2. A tolerance-based comparison
    on well-separated, low-noise synthetic blobs can pass either way (5x5 and
    9x9 patches fit nearly identical Gaussians there), so this asserts the
    argument directly instead of inferring it from fitted values.
    """
    import napari_spotiflow_tracking._fitting as fitting_module

    real_fit_single_spot_scipy = fitting_module._fit_single_spot_scipy
    captured_radii = []

    def spy(image, center_yx, patch_radius):
        captured_radii.append(patch_radius)
        return real_fit_single_spot_scipy(image, center_yx, patch_radius)

    monkeypatch.setattr(fitting_module, "_fit_single_spot_scipy", spy)

    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fit = FitSettings(enabled=True, window=2)
    df = spots.rename(columns={"frame": "time"})
    detect._fit_frames_scipy(stack, df, fit)

    assert captured_radii, "expected at least one scipy fit call"
    assert all(radius == fit.window for radius in captured_radii)


def test_fit_empty_spots_scipy_path_returns_empty_mask():
    """Zero spots on the scipy backend (forced via device='cpu') is a distinct
    code path from the torchimize one and needs its own empty-input check."""
    assert detect.fit_backend("cpu") == "scipy"
    stack = np.zeros((2, 8, 8), np.uint16)
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    empty_spots, empty_mask = detect.fit_spots(stack, empty, FitSettings(enabled=True), "cpu")
    assert len(empty_spots) == 0 and empty_mask.shape == stack.shape and not empty_mask.any()

    blob_stack = _blob_stack(n_frames=2, size=8)
    blob_spots = detect.detect_stack(blob_stack, DetectionSettings(method="log"), model=None)
    fitted_spots, _ = detect.fit_spots(blob_stack, blob_spots, FitSettings(enabled=True), "cpu")
    assert list(empty_spots.columns) == list(fitted_spots.columns)


def test_fit_spots_falls_back_to_scipy_on_a_generic_runtime_error(monkeypatch, caplog):
    """A torchimize failure that isn't GPU-memory-exhaustion degrades this
    movie's fit to scipy instead of failing the run."""
    import napari_spotiflow_tracking._fitting as fitting_module

    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    monkeypatch.setattr(detect, "torchimize_available", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitting_module, "fit_all_frames_gpu", broken)

    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        fitted, mask = detect.fit_spots(stack, spots, FitSettings(enabled=True), "cuda")

    assert {"frame", "y", "x", "amp", "x_fit", "y_fit", "sigma_x", "sigma_y", "bg",
            "area", "mean_intensity", "max_intensity", "spot_radius"} <= set(fitted.columns)
    assert mask.any()
    assert any("falling back to the scipy" in r.getMessage() for r in caplog.records)


def test_fit_spots_reraises_gpu_out_of_memory(monkeypatch):
    """GPU memory exhaustion must fail loudly, not silently become the much
    slower scipy path."""
    import napari_spotiflow_tracking._fitting as fitting_module

    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    monkeypatch.setattr(detect, "torchimize_available", lambda: True)

    class OutOfMemoryError(RuntimeError):
        pass

    def broken(*args, **kwargs):
        raise OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(fitting_module, "fit_all_frames_gpu", broken)

    with pytest.raises(OutOfMemoryError):
        detect.fit_spots(stack, spots, FitSettings(enabled=True), "cuda")
