"""MSD computation: fits, ensemble, class counts, directed metrics, units."""
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch.analyses.msd import compute
from napari_spotiflow_tracking.batch.analyses.msd.units import Units
from napari_spotiflow_tracking.batch.config import Calibration


def brownian_tracks(n_tracks=6, n_frames=60, step_px=1.0, seed=0):
    """Tracks whose per-frame steps are N(0, step_px) in each axis → alpha ≈ 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        pos = np.cumsum(rng.normal(0, step_px, size=(n_frames, 2)), axis=0)
        for frame, (y, x) in enumerate(pos):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


def directed_tracks(n_tracks=4, n_frames=60, speed=2.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        drift = np.arange(n_frames)[:, None] * np.array([speed, speed])
        jitter = rng.normal(0, 0.05, size=(n_frames, 2))
        for frame, (y, x) in enumerate(drift + jitter):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


PX = Units(Calibration(), sigma_nm=35.0)
UM = Units(Calibration(pixel_size_nm=100, frame_interval_s=0.5), sigma_nm=35.0)


def test_units_pixel_mode():
    assert not PX.is_physical
    assert (PX.length_scale, PX.delta, PX.sigma) == (1.0, 1.0, 0.0)
    assert PX.length_unit == "px" and PX.time_unit == "frames"
    assert PX.msd_unit == "px²" and PX.d_unit == "px²/frame^α"
    assert "px" in PX.describe()


def test_units_physical_mode():
    assert UM.is_physical
    assert UM.length_scale == pytest.approx(0.1)      # 100 nm → µm
    assert UM.delta == pytest.approx(0.5)
    assert UM.sigma == pytest.approx(0.035)
    assert UM.length_unit == "µm" and UM.time_unit == "s"
    assert UM.msd_unit == "µm²" and UM.d_unit == "µm²/s^α"


def test_curves_and_fits_shape_and_columns():
    curves, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                                min_length=5)
    assert list(curves.columns) == compute.MSD_CURVE_COLS
    assert list(fits.columns) == compute.MSD_FIT_COLS
    assert set(fits["track_id"]) == set(range(6))
    assert curves["lag"].max() <= 10
    assert (curves["n_pairs"] > 0).all()


def test_brownian_tracks_are_classified_diffusive():
    _, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=15,
                                           min_length=5)
    assert fits["alpha"].median() == pytest.approx(1.0, abs=0.25)
    assert (fits["motility_class"] == "diffusive").mean() >= 0.8


def test_directed_tracks_are_classified_directed():
    _, fits = compute.compute_msd_analysis(directed_tracks(), PX, max_lag=15,
                                           min_length=5)
    assert fits["alpha"].median() > 1.5
    assert (fits["motility_class"] == "directed").all()


def test_short_tracks_are_dropped():
    tracks = brownian_tracks(n_tracks=2, n_frames=60)
    short = pd.DataFrame({"frame": [0, 1, 2], "y": [0.0, 1.0, 2.0],
                          "x": [0.0, 0.0, 1.0], "particle": [99, 99, 99]})
    _, fits = compute.compute_msd_analysis(pd.concat([tracks, short]), PX,
                                           max_lag=10, min_length=5)
    assert 99 not in set(fits["track_id"])


def test_empty_input_gives_empty_frames_with_columns():
    empty = pd.DataFrame({"frame": [], "y": [], "x": [], "particle": []})
    curves, fits = compute.compute_msd_analysis(empty, PX, max_lag=10, min_length=5)
    assert len(curves) == 0 and list(curves.columns) == compute.MSD_CURVE_COLS
    assert len(fits) == 0 and list(fits.columns) == compute.MSD_FIT_COLS


def test_physical_units_scale_d_by_pixel_and_frame():
    tracks = brownian_tracks(seed=3)
    _, px_fits = compute.compute_msd_analysis(tracks, PX, max_lag=10, min_length=5)
    _, um_fits = compute.compute_msd_analysis(tracks, UM, max_lag=10, min_length=5)
    # D scales as length²/time: (0.1 µm/px)² / (0.5 s/frame) = 0.02
    ratio = (um_fits["D"] / px_fits["D"]).median()
    assert ratio == pytest.approx(0.02, rel=0.25)
    assert um_fits["alpha"].median() == pytest.approx(px_fits["alpha"].median(), abs=0.1)


def test_ensemble_msd_averages_curves():
    curves, _ = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                             min_length=5)
    ens = compute.ensemble_msd(curves)
    assert {"lag", "msd", "sem", "n_tracks"} <= set(ens.columns)
    assert len(ens) == curves["lag"].nunique()
    assert ens["msd"].is_monotonic_increasing


def test_class_counts_covers_every_class():
    _, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                           min_length=5)
    counts = compute.class_counts(fits)
    # "unclassified" appears only when some track has one (source behaviour).
    assert {"constrained", "diffusive", "directed"} <= set(counts)
    assert set(counts) <= {"constrained", "diffusive", "directed", "unclassified"}
    assert sum(counts.values()) == len(fits)
    assert compute.class_counts(fits.iloc[:0]) == {"constrained": 0, "diffusive": 0,
                                                   "directed": 0}


def test_directed_metrics_only_directed_tracks():
    tracked = pd.concat([brownian_tracks(), directed_tracks()])
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=15, min_length=5)
    props = compute.track_table(tracked, None, fits, PX)
    directed = compute.directed_metrics(props, PX)
    assert len(directed) == (props["motility_class"] == "directed").sum()
    assert {"net_displacement", "mean_speed"} <= set(directed.columns)


def test_track_table_merges_fits_and_marks_unclassified():
    tracked = brownian_tracks(n_tracks=3, n_frames=60)
    short = pd.DataFrame({"frame": [0, 1], "y": [0.0, 1.0], "x": [0.0, 1.0],
                          "particle": [77, 77]})
    tracked = pd.concat([tracked, short])
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=10, min_length=5)
    table = compute.track_table(tracked, None, fits, PX)
    assert {"track_id", "length", "mean_speed", "D", "alpha", "motility_class"} <= set(table.columns)
    assert table.loc[table.track_id == 77, "motility_class"].item() == "unclassified"


def test_track_table_adds_physical_columns_only_when_calibrated():
    tracked = brownian_tracks(n_tracks=2, n_frames=40)
    _, fits = compute.compute_msd_analysis(tracked, UM, max_lag=10, min_length=5)
    table = compute.track_table(tracked, None, fits, UM)
    assert {"duration_s", "mean_speed_um_s"} <= set(table.columns)
    assert "length" in table.columns          # pixel columns stay beside them
    px_table = compute.track_table(tracked, None, fits, PX)
    assert "duration_s" not in px_table.columns


def test_track_table_samples_intensity_when_image_given():
    tracked = brownian_tracks(n_tracks=2, n_frames=20, step_px=0.2)
    tracked[["y", "x"]] = tracked[["y", "x"]].abs() + 5
    image = np.tile(np.arange(32, dtype=np.uint16), (20, 32, 1))
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=5, min_length=5)
    table = compute.track_table(tracked, image, fits, PX)
    assert "mean_intensity" in table.columns and table["mean_intensity"].notna().all()
