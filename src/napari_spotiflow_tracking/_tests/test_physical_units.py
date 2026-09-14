"""Tests for physical-unit conversion of per-track properties.

``compute_track_properties`` works in the units napari uses -- pixels and
frames. ``add_physical_columns`` appends the calibrated equivalents alongside
them, so a publication table carries µm and seconds without the pixel columns
(which the bulk script's ``--summary-only`` and ``--compare`` paths read back
and rescale themselves) changing meaning.
"""
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._track_properties import (
    add_physical_columns,
    compute_track_properties,
)

PX_NM = 92.0          # effective pixel size of the TIRF rig
DT_S = 0.005004       # frame interval


def _props():
    """Two tracks: one straight 3-frame track, one single-frame track."""
    tracked = pd.DataFrame({
        "frame": [0, 1, 2, 0],
        "y": [0.0, 0.0, 0.0, 5.0],
        "x": [0.0, 3.0, 7.0, 5.0],
        "particle": [1, 1, 1, 2],
    })
    return compute_track_properties(tracked)


def test_adds_micrometre_and_second_columns_when_calibrated():
    out = add_physical_columns(_props(), PX_NM, DT_S)
    assert {"duration_s", "total_displacement_um", "net_displacement_um",
            "mean_speed_um_s"} <= set(out.columns)


def test_lengths_convert_at_the_pixel_size():
    out = add_physical_columns(_props(), PX_NM, DT_S)
    row = out[out["track_id"] == 1].iloc[0]
    # 3 px + 4 px of path, 7 px net, at 0.092 µm/px
    assert row["total_displacement_um"] == pytest.approx(7 * 0.092)
    assert row["net_displacement_um"] == pytest.approx(7 * 0.092)


def test_duration_spans_the_steps_not_the_localisations():
    """A 3-frame track lasts 2 frame intervals, not 3."""
    out = add_physical_columns(_props(), PX_NM, DT_S)
    row = out[out["track_id"] == 1].iloc[0]
    assert row["duration_s"] == pytest.approx(2 * DT_S)


def test_speed_is_the_path_length_over_the_duration():
    out = add_physical_columns(_props(), PX_NM, DT_S)
    row = out[out["track_id"] == 1].iloc[0]
    assert row["mean_speed_um_s"] == pytest.approx(
        row["total_displacement_um"] / row["duration_s"])


def test_single_frame_track_has_zero_duration_and_no_speed():
    """A one-localisation track has no elapsed time, so speed is undefined
    rather than an infinity that would poison the axis limits downstream."""
    out = add_physical_columns(_props(), PX_NM, DT_S)
    row = out[out["track_id"] == 2].iloc[0]
    assert row["duration_s"] == 0.0
    assert np.isnan(row["mean_speed_um_s"])


def test_pixel_columns_are_left_untouched():
    props = _props()
    out = add_physical_columns(props, PX_NM, DT_S)
    for col in ("length", "total_displacement", "net_displacement", "mean_speed"):
        pd.testing.assert_series_equal(out[col], props[col])


def test_input_frame_is_not_mutated():
    props = _props()
    before = list(props.columns)
    add_physical_columns(props, PX_NM, DT_S)
    assert list(props.columns) == before


@pytest.mark.parametrize("px_nm,dt_s", [
    (None, None), (PX_NM, None), (None, DT_S),
])
def test_no_columns_added_without_a_full_calibration(px_nm, dt_s):
    """Half a calibration is no calibration: a µm column computed from a
    missing pixel size would be pixels wearing the wrong label."""
    props = _props()
    out = add_physical_columns(props, px_nm, dt_s)
    assert list(out.columns) == list(props.columns)


def test_empty_input_still_gains_the_columns():
    """Schema stability: a stack with no tracks must write the same header as
    one with tracks, or reading a run back becomes conditional."""
    empty = compute_track_properties(
        pd.DataFrame({"frame": [], "y": [], "x": [], "particle": []}))
    out = add_physical_columns(empty, PX_NM, DT_S)
    assert "mean_speed_um_s" in out.columns
    assert len(out) == 0
