"""Tests for _papa_analysis pure-Python helpers."""

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._papa_analysis import (
    build_cycle_info,
    compute_particles_per_frame,
)


class TestComputeParticlesPerFrame:
    def test_basic_counts(self):
        # Frame 0 has 1 spot, frame 1 has 0 spots, frame 2 has 4 spots.
        # 0-indexed input -> 1-indexed output.
        points = np.array([
            [0, 10.0, 10.0],
            [2, 11.0, 11.0],
            [2, 12.0, 12.0],
            [2, 13.0, 13.0],
            [2, 14.0, 14.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert list(per_frame.columns) == ["frame", "n_particles"]
        assert per_frame["frame"].tolist() == [1, 2, 3]
        assert per_frame["n_particles"].tolist() == [1, 0, 4]

    def test_fills_missing_frames(self):
        # Only frames 0, 2, 5 have detections -> output covers 1..6 with zeros.
        points = np.array([
            [0, 10.0, 10.0],
            [2, 10.0, 10.0],
            [5, 10.0, 10.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert per_frame["frame"].tolist() == [1, 2, 3, 4, 5, 6]
        assert per_frame["n_particles"].tolist() == [1, 0, 1, 0, 0, 1]

    def test_non_integer_frames_rounded(self):
        points = np.array([
            [0.4, 10.0, 10.0],
            [1.6, 10.0, 10.0],  # rounds to 2 -> 1-indexed 3
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert per_frame["frame"].tolist() == [1, 2, 3]
        assert per_frame["n_particles"].tolist() == [1, 0, 1]

    def test_negative_frames_clipped(self):
        points = np.array([
            [-1, 10.0, 10.0],
            [-2, 10.0, 10.0],
            [0, 10.0, 10.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 2
        # Negatives clipped to 0 -> 1-indexed frame 1, joined with the original
        # frame-0 spot -> 3 spots on frame 1.
        assert per_frame["frame"].tolist() == [1]
        assert per_frame["n_particles"].tolist() == [3]

    def test_returns_int_dtypes(self):
        points = np.array([[0, 1.0, 2.0], [1, 3.0, 4.0]])
        per_frame, _ = compute_particles_per_frame(points)
        assert pd.api.types.is_integer_dtype(per_frame["frame"])
        assert pd.api.types.is_integer_dtype(per_frame["n_particles"])


class TestBuildCycleInfo:
    def test_exact_fit_one_cycle(self):
        # max_time=12, g1=5, g2=5, skip=1: G1 frames 1-5, skip frame 6,
        # G2 frames 7-11, skip frame 12. One complete cycle.
        info = build_cycle_info(max_time=12, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
        ]

    def test_partial_g2(self):
        # max_time=8, g1=5, g2=5, skip=1: G1 1-5, skip 6, G2 7-8 (truncated).
        info = build_cycle_info(max_time=8, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 8},
        ]

    def test_no_room_for_g2(self):
        # max_time=5: only G1.
        info = build_cycle_info(max_time=5, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
        ]

    def test_two_full_cycles(self):
        # max_time=24, g1=5, g2=5, skip=1: cycle 1 G1 1-5, G2 7-11; cycle 2
        # G1 13-17, G2 19-23.
        info = build_cycle_info(max_time=24, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
            {"cycle": 2, "group": 1, "start_frame": 13, "end_frame": 17},
            {"cycle": 2, "group": 2, "start_frame": 19, "end_frame": 23},
        ]

    def test_zero_max_time(self):
        info = build_cycle_info(max_time=0, group1_size=5, group2_size=5, skip_frames=1)
        assert info == []

    def test_skip_zero(self):
        # No skip: G1 1-3 then G2 4-6.
        info = build_cycle_info(max_time=6, group1_size=3, group2_size=3, skip_frames=0)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 3},
            {"cycle": 1, "group": 2, "start_frame": 4, "end_frame": 6},
        ]
