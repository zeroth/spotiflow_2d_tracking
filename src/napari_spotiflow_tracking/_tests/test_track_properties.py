"""Tests for compute_track_properties pure-Python helper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._track_properties import compute_track_properties


def _track_df(rows):
    return pd.DataFrame(rows, columns=["frame", "y", "x", "particle"])


class TestGeometricOnly:
    def test_three_known_tracks(self):
        # Track 0: 5 frames straight line, 1 px right per frame
        # Track 1: 3 frames diagonal, sqrt(2) px per frame
        # Track 2: 1 frame (length-1)
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 1.0, 0],
            [2, 0.0, 2.0, 0],
            [3, 0.0, 3.0, 0],
            [4, 0.0, 4.0, 0],
            [0, 5.0, 5.0, 1],
            [1, 6.0, 6.0, 1],
            [2, 7.0, 7.0, 1],
            [10, 10.0, 10.0, 2],
        ])
        out = compute_track_properties(df)
        # Sorted by track_id
        assert out["track_id"].tolist() == [0, 1, 2]
        # Track 0: length=5, total=4, net=4, mean_speed=1
        assert out.loc[out["track_id"] == 0, "length"].iloc[0] == 5
        assert out.loc[out["track_id"] == 0, "total_displacement"].iloc[0] == pytest.approx(4.0)
        assert out.loc[out["track_id"] == 0, "net_displacement"].iloc[0] == pytest.approx(4.0)
        assert out.loc[out["track_id"] == 0, "mean_speed"].iloc[0] == pytest.approx(1.0)
        # Track 1: 3 frames, sqrt(2) per step, total = 2*sqrt(2)
        assert out.loc[out["track_id"] == 1, "length"].iloc[0] == 3
        assert out.loc[out["track_id"] == 1, "total_displacement"].iloc[0] == pytest.approx(2 * np.sqrt(2))
        assert out.loc[out["track_id"] == 1, "net_displacement"].iloc[0] == pytest.approx(2 * np.sqrt(2))
        assert out.loc[out["track_id"] == 1, "mean_speed"].iloc[0] == pytest.approx(np.sqrt(2))

    def test_single_frame_track_zero_motion(self):
        df = _track_df([[0, 5.0, 5.0, 0]])
        out = compute_track_properties(df)
        assert out["length"].iloc[0] == 1
        assert out["total_displacement"].iloc[0] == 0.0
        assert out["net_displacement"].iloc[0] == 0.0
        assert out["mean_speed"].iloc[0] == 0.0

    def test_columns_no_image(self):
        df = _track_df([[0, 0.0, 0.0, 0]])
        out = compute_track_properties(df)
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }

    def test_returns_sorted_by_track_id(self):
        df = _track_df([
            [0, 0.0, 0.0, 5],
            [0, 0.0, 0.0, 1],
            [0, 0.0, 0.0, 3],
        ])
        out = compute_track_properties(df)
        assert out["track_id"].tolist() == [1, 3, 5]

    def test_empty_input(self):
        df = _track_df([])
        out = compute_track_properties(df)
        assert len(out) == 0
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }


class TestWithImage:
    def test_image_intensity_3d(self):
        # 3D image: image[t, y, x] = t * 100 + y * 10 + x
        image = np.zeros((3, 10, 10), dtype=float)
        for t in range(3):
            for y in range(10):
                for x in range(10):
                    image[t, y, x] = t * 100 + y * 10 + x
        # Track 0: stays at (0, 0) for 3 frames -> values 0, 100, 200
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 0.0, 0],
            [2, 0.0, 0.0, 0],
        ])
        out = compute_track_properties(df, image=image)
        assert out["mean_intensity"].iloc[0] == pytest.approx(100.0)
        assert out["max_intensity"].iloc[0] == pytest.approx(200.0)
        assert out["intensity_std"].iloc[0] == pytest.approx(np.std([0, 100, 200]))

    def test_image_2d_fallback(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        # Track at (2, 3) and (4, 5)
        df = _track_df([
            [0, 2.0, 3.0, 0],
            [1, 4.0, 5.0, 0],
        ])
        out = compute_track_properties(df, image=image)
        # Sampling on 2D image: image[2, 3] = 23, image[4, 5] = 45
        assert out["mean_intensity"].iloc[0] == pytest.approx((23 + 45) / 2)

    def test_image_dim_mismatch_raises(self):
        image = np.zeros((2, 10, 10))  # only 2 frames
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 0.0, 0],
            [5, 0.0, 0.0, 0],  # frame 5 out of range
        ])
        with pytest.raises(ValueError, match="T-axis"):
            compute_track_properties(df, image=image)

    def test_oob_coords_clipped(self):
        # Track has coords way outside image bounds
        image = np.ones((1, 5, 5), dtype=float)
        df = _track_df([[0, -10.0, 100.0, 0]])
        out = compute_track_properties(df, image=image)
        # Clipped coord still samples valid pixel
        assert out["mean_intensity"].iloc[0] == 1.0

    def test_columns_with_image(self):
        image = np.ones((1, 5, 5))
        df = _track_df([[0, 0.0, 0.0, 0]])
        out = compute_track_properties(df, image=image)
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
            "mean_intensity", "max_intensity", "intensity_std",
        }

    def test_invalid_image_ndim_raises(self):
        image = np.zeros((3, 3, 3, 3))  # 4D
        df = _track_df([[0, 0.0, 0.0, 0]])
        with pytest.raises(ValueError, match="2D.*3D"):
            compute_track_properties(df, image=image)


class TestMsd:
    def test_msd_basic_two_lag_times(self):
        from napari_spotiflow_tracking._track_properties import msd
        # Track moving 1 px in x per frame, no y motion → at lag t, MSD = t²
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        out = msd(pos, limit=25)
        # lag 1 → mean of (1, 1, 1) squared in x and 0² in y → 1
        # lag 2 → mean of (2, 2)²,0² → 4
        # lag 3 → 3² = 9
        assert out.iloc[0] == pytest.approx(1.0)
        assert out.iloc[1] == pytest.approx(4.0)
        assert out.iloc[2] == pytest.approx(9.0)

    def test_msd_limit_clipped_to_n_minus_1(self):
        from napari_spotiflow_tracking._track_properties import msd
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = msd(pos, limit=25)
        assert len(out) == 1


class TestMsdBasicFit:
    def test_fit_recovers_alpha_two_for_ballistic_motion(self):
        from napari_spotiflow_tracking._track_properties import (
            msd as _msd,
            msd_basic_fit,
        )
        # 1 px/frame ballistic motion → MSD = t² → α=2 in 4·D·t^α.
        pos = np.array([[float(i), 0.0] for i in range(10)])
        series = _msd(pos, limit=9)
        fit = msd_basic_fit(series, delta=1.0)
        assert fit["alpha"] == pytest.approx(2.0, abs=0.05)
        assert fit["diffusion_coefficient"] > 0
        assert np.allclose(fit["y_fit"], series.to_numpy(), rtol=1e-2)

    def test_fit_returns_required_keys(self):
        from napari_spotiflow_tracking._track_properties import (
            msd as _msd,
            msd_basic_fit,
        )
        pos = np.array([[float(i), float(i)] for i in range(8)])
        series = _msd(pos, limit=7)
        fit = msd_basic_fit(series, delta=1.0)
        assert set(fit.keys()) == {
            "x", "y_fit", "diffusion_coefficient", "alpha",
        }
        assert len(fit["x"]) == len(series)
        assert len(fit["y_fit"]) == len(series)


class TestFindSteps:
    def test_find_steps_no_steps_constant_data(self):
        from napari_spotiflow_tracking._track_properties import find_steps
        data = np.full(50, 100.0)
        table, fitx, _ = find_steps(data, window=5, threshold=0.5)
        # No steps detected on flat data
        assert len(table) == 0
        # Fitx falls back to the mean
        assert np.allclose(fitx, 100.0)

    def test_find_steps_detects_obvious_step(self):
        from napari_spotiflow_tracking._track_properties import find_steps
        # Step at frame 25 from level 10 to level 100
        data = np.concatenate([np.full(25, 10.0), np.full(25, 100.0)])
        table, fitx, _ = find_steps(data, window=5, threshold=0.3)
        assert len(table) >= 1
        # fitx should be piecewise-constant with two levels near 10 and 100.
        # Allow tolerance because the detected step index may be 1-2 frames
        # off from the true transition, causing level_before/after means to
        # include a small number of samples from the wrong side.
        assert fitx[5] == pytest.approx(10.0, abs=1.0)
        assert fitx[48] == pytest.approx(100.0, abs=5.0)
