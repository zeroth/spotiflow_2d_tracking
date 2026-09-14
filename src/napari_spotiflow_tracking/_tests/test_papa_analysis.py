"""Tests for _papa_analysis pure-Python helpers."""

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
    compute_per_cycle_summary,
    lead_in_span,
    normalize_min_max,
)


class TestNormalizeMinMax:
    """Counts rescaled to 0-1 against the whole trace's min and max."""

    @staticmethod
    def _frames(counts):
        return pd.DataFrame({
            "frame": list(range(1, len(counts) + 1)),
            "n_particles": counts,
        })

    def test_rescales_to_zero_one(self):
        per_frame = self._frames([2, 4, 6, 10])
        out, (lo, hi) = normalize_min_max(per_frame)
        assert (lo, hi) == (2.0, 10.0)
        assert out["n_particles"].tolist() == pytest.approx(
            [0.0, 0.25, 0.5, 1.0]
        )

    def test_min_maps_to_zero_and_max_to_one(self):
        per_frame = self._frames([7, 3, 91, 40])
        out, _ = normalize_min_max(per_frame)
        assert out["n_particles"].min() == pytest.approx(0.0)
        assert out["n_particles"].max() == pytest.approx(1.0)

    def test_keeps_frame_column(self):
        per_frame = self._frames([1, 5, 9])
        out, _ = normalize_min_max(per_frame)
        assert out["frame"].tolist() == [1, 2, 3]

    def test_does_not_mutate_input(self):
        per_frame = self._frames([2, 8, 4])
        _ = normalize_min_max(per_frame)
        assert per_frame["n_particles"].tolist() == [2, 8, 4]

    def test_range_frames_sets_the_scale(self):
        # The lead-in is drawn but excluded from analysis, so the scale comes
        # from the cycle frames; lead-in points ride along on that scale.
        per_frame = self._frames([100, 2, 4, 6, 10])
        out, (lo, hi) = normalize_min_max(per_frame,
                                          range_frames=per_frame.iloc[1:])
        assert (lo, hi) == (2.0, 10.0)
        # Frame 1 sits outside the scaled range, and is not clipped to it.
        assert out["n_particles"].tolist() == pytest.approx(
            [12.25, 0.0, 0.25, 0.5, 1.0]
        )

    def test_flat_trace_reports_no_range(self):
        # max == min would divide by zero; the caller must be able to skip.
        per_frame = self._frames([5, 5, 5])
        out, (lo, hi) = normalize_min_max(per_frame)
        assert out is None
        assert (lo, hi) == (5.0, 5.0)

    def test_empty_stack_reports_no_range(self):
        per_frame = pd.DataFrame({"frame": [], "n_particles": []})
        out, (lo, hi) = normalize_min_max(per_frame)
        assert out is None
        assert (lo, hi) == (0.0, 0.0)


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
    # Block walk for g1=5, g2=5, skip=1:
    #   b0 1-5 (lead-in)   b1 7-11 (G1)   b2 13-17 (G2)
    #   b3 19-23 (G1)      b4 25-29 (G2)
    # The alternation is anchored at b1, not at frame 1, so a cycle is
    # (G1 block, G2 block) and cycle 1 = b1 + b2.

    def test_first_analysed_block_is_group_1(self):
        # The block immediately after the lead-in must be group 1 — the lead-in
        # sits outside the pattern rather than counting as the group-1 block.
        info = build_cycle_info(max_time=29, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info[0]["group"] == 1
        assert info[0]["start_frame"] == 7

    def test_exact_fit_one_cycle(self):
        # 17 frames is the minimum: lead-in 5 + skip 1 + G1 5 + skip 1 + G2 5.
        info = build_cycle_info(max_time=17, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 2, "start_frame": 13, "end_frame": 17},
        ]

    def test_truncated_final_block_dropped(self):
        # One frame short: b2 would end at 17 but max_time is 16, so cycle 1 is
        # incomplete and nothing is reported. No group is ever clipped.
        info = build_cycle_info(max_time=16, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    def test_trailing_odd_block_dropped(self):
        # 23 frames fits b0..b3. After the lead-in three blocks remain, so b3
        # (a G1 block) has no G2 partner and is dropped.
        info = build_cycle_info(max_time=23, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 2, "start_frame": 13, "end_frame": 17},
        ]

    def test_two_full_cycles(self):
        # 29 frames fits b0..b4: lead-in plus two complete (G1, G2) pairs.
        info = build_cycle_info(max_time=29, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 2, "start_frame": 13, "end_frame": 17},
            {"cycle": 2, "group": 1, "start_frame": 19, "end_frame": 23},
            {"cycle": 2, "group": 2, "start_frame": 25, "end_frame": 29},
        ]

    def test_unequal_group_sizes(self):
        # g1=200, g2=500, skip=1: lead-in 1-200 (200f, like the G1 block that
        # follows it), G1 202-401 (200f), G2 403-902 (500f), then G1 904-1103
        # with no G2 partner. Sizes are never re-sliced.
        info = build_cycle_info(max_time=1403, group1_size=200,
                                group2_size=500, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 202, "end_frame": 401},
            {"cycle": 1, "group": 2, "start_frame": 403, "end_frame": 902},
        ]

    def test_skip_zero(self):
        # g1=3, g2=3, skip=0: lead-in 1-3, G1 4-6, G2 7-9.
        info = build_cycle_info(max_time=9, group1_size=3, group2_size=3,
                                skip_frames=0)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 4, "end_frame": 6},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 9},
        ]

    def test_stack_only_covers_lead_in(self):
        info = build_cycle_info(max_time=5, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    def test_zero_max_time(self):
        info = build_cycle_info(max_time=0, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    @pytest.mark.parametrize("g1,g2", [(0, 5), (5, 0)])
    def test_zero_group_size_raises(self, g1, g2):
        # A zero-size block never advances the cursor when skip_frames is 0,
        # which used to hang forever. Fail loudly instead.
        with pytest.raises(ValueError, match="must be >= 1"):
            build_cycle_info(max_time=100, group1_size=g1, group2_size=g2,
                             skip_frames=1)

    def test_negative_skip_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            build_cycle_info(max_time=100, group1_size=5, group2_size=5,
                             skip_frames=-1)


class TestLeadInSpan:
    def test_normal(self):
        assert lead_in_span(max_time=100, group1_size=20) == (1, 20)

    def test_clamped_to_short_stack(self):
        # Shading must not run off the end of a stack shorter than the lead-in.
        assert lead_in_span(max_time=8, group1_size=20) == (1, 8)

    def test_no_frames(self):
        assert lead_in_span(max_time=0, group1_size=20) is None

    def test_zero_group1_size_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            lead_in_span(max_time=100, group1_size=0)


class TestAnnotateWithCycleGroup:
    def test_assigns_cycle_and_group(self):
        per_frame = pd.DataFrame({
            "frame": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "n_particles": [0] * 12,
        })
        cycle_info = [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
        ]
        out = annotate_with_cycle_group(per_frame, cycle_info)

        # Frames 1-5 -> cycle 1 group 1; frame 6 -> NaN/NaN (skip gap);
        # frames 7-11 -> cycle 1 group 2; frame 12 -> NaN/NaN.
        expected_cycle = [1, 1, 1, 1, 1, pd.NA, 1, 1, 1, 1, 1, pd.NA]
        expected_group = [1, 1, 1, 1, 1, pd.NA, 2, 2, 2, 2, 2, pd.NA]
        assert out["cycle"].tolist() == expected_cycle
        assert out["group"].tolist() == expected_group
        # Original columns preserved.
        assert out["frame"].tolist() == per_frame["frame"].tolist()
        assert out["n_particles"].tolist() == per_frame["n_particles"].tolist()

    def test_does_not_mutate_input(self):
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [0, 0]})
        cycle_info = [{"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 2}]
        _ = annotate_with_cycle_group(per_frame, cycle_info)
        assert "cycle" not in per_frame.columns
        assert "group" not in per_frame.columns

    def test_empty_cycle_info(self):
        per_frame = pd.DataFrame({"frame": [1, 2, 3], "n_particles": [0, 0, 0]})
        out = annotate_with_cycle_group(per_frame, [])
        assert out["cycle"].isna().all()
        assert out["group"].isna().all()

    def test_uses_nullable_int64_dtype(self):
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [0, 0]})
        cycle_info = [{"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 2}]
        out = annotate_with_cycle_group(per_frame, cycle_info)
        assert str(out["cycle"].dtype) == "Int64"
        assert str(out["group"].dtype) == "Int64"


class TestComputePerCycleSummary:
    def test_counts_and_ratios(self):
        # 17 frames, g1=5/g2=5/skip=1 -> one cycle: G1 7-11, G2 13-17.
        # Giving the two blocks different densities pins the pairing direction:
        # if G1 were paired with the lead-in instead, DR_count would be 0.
        per_frame = pd.DataFrame({
            "frame": list(range(1, 18)),
            "n_particles": [0] * 17,
        })
        per_frame.loc[per_frame["frame"].between(7, 11), "n_particles"] = 1
        per_frame.loc[per_frame["frame"].between(13, 17), "n_particles"] = 2
        cycle_info = build_cycle_info(max_time=17, group1_size=5,
                                      group2_size=5, skip_frames=1)

        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA")

        assert out["cycle"].tolist() == [1]
        assert out["DR_count"].tolist() == [5]      # 5 frames x 1 spot
        assert out["PAPA_count"].tolist() == [10]   # 5 frames x 2 spots
        assert out["PAPA_over_DR"].tolist() == [2.0]
        assert out["DR_over_PAPA"].tolist() == [0.5]

    def test_empty_cycle_info_keeps_columns(self):
        # A stack too short for one complete cycle must still yield a frame the
        # plotting code can index into.
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [1, 1]})
        out = compute_per_cycle_summary(per_frame, [], "DR", "PAPA")
        assert len(out) == 0
        assert list(out.columns) == [
            "cycle", "DR_count", "PAPA_count",
            "DR_ratio_count", "PAPA_ratio_count",
            "DR_over_PAPA", "PAPA_over_DR",
        ]


class TestComputePerCycleSummaryFrameFraction:
    """``frame_fraction`` narrows the ratio window to the first N% of a block."""

    @staticmethod
    def _ramped_cycle():
        """One cycle, G1 7-11 and G2 13-17, with counts rising across G1.

        G1 holds 1,2,3,4,5 (total 15) and G2 a flat 2 per frame (total 10), so
        a leading sub-window of G1 is a different fraction of its total than a
        trailing one — which is what pins the window to the *start* of a block.
        """
        per_frame = pd.DataFrame({
            "frame": list(range(1, 18)),
            "n_particles": [0] * 17,
        })
        per_frame.loc[per_frame["frame"].between(7, 11), "n_particles"] = [
            1, 2, 3, 4, 5,
        ]
        per_frame.loc[per_frame["frame"].between(13, 17), "n_particles"] = 2
        cycle_info = build_cycle_info(max_time=17, group1_size=5,
                                      group2_size=5, skip_frames=1)
        return per_frame, cycle_info

    def test_default_is_whole_block(self):
        per_frame, cycle_info = self._ramped_cycle()
        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA")
        assert out["DR_ratio_count"].tolist() == [15]
        assert out["PAPA_ratio_count"].tolist() == [10]
        assert out["PAPA_over_DR"].tolist() == [10 / 15]

    def test_fraction_counts_leading_frames_only(self):
        # 0.4 of a 5-frame block -> the first 2 frames.
        # DR 7-8 = 1+2 = 3; PAPA 13-14 = 2+2 = 4.
        per_frame, cycle_info = self._ramped_cycle()
        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA",
                                        frame_fraction=0.4)
        assert out["DR_ratio_count"].tolist() == [3]
        assert out["PAPA_ratio_count"].tolist() == [4]
        assert out["PAPA_over_DR"].tolist() == [4 / 3]
        assert out["DR_over_PAPA"].tolist() == [3 / 4]

    def test_full_block_counts_are_unchanged_by_fraction(self):
        # The displayed totals must not move when only the ratio window shrinks.
        per_frame, cycle_info = self._ramped_cycle()
        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA",
                                        frame_fraction=0.4)
        assert out["DR_count"].tolist() == [15]
        assert out["PAPA_count"].tolist() == [10]

    def test_tiny_fraction_still_counts_one_frame(self):
        # Rounding to zero frames would make every ratio NaN; floor at 1.
        per_frame, cycle_info = self._ramped_cycle()
        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA",
                                        frame_fraction=0.01)
        assert out["DR_ratio_count"].tolist() == [1]    # frame 7 only
        assert out["PAPA_ratio_count"].tolist() == [2]  # frame 13 only

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_fraction(self, bad):
        per_frame, cycle_info = self._ramped_cycle()
        with pytest.raises(ValueError, match="frame_fraction"):
            compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA",
                                      frame_fraction=bad)
