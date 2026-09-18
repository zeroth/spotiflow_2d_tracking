"""The ported msd analysis must reproduce the old script's numbers.

Opt-in: `pytest -m slow`. It runs the old script's analysis path and the new
analysis over the same tracks and compares the fit tables.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bulk_track_analysis.py"

pytestmark = pytest.mark.slow


def _load_script():
    if "bulk_track_analysis" in sys.modules:
        return sys.modules["bulk_track_analysis"]
    spec = importlib.util.spec_from_file_location("bulk_track_analysis", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bulk_track_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tracks(n_tracks=12, n_frames=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        drift = 0.8 if tid % 3 == 0 else 0.0
        pos = np.cumsum(rng.normal(drift, 1.0, size=(n_frames, 2)), axis=0)
        for frame, (y, x) in enumerate(pos):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("pixel_size_nm, frame_interval_s", [(None, None), (100, 0.5)])
def test_fits_match_the_old_script(pixel_size_nm, frame_interval_s):
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    bulk = _load_script()
    tracked = _tracks()
    old_cal = bulk.Calibration(pixel_size_nm=pixel_size_nm,
                               frame_interval_s=frame_interval_s, sigma_nm=35.0)
    old_curves, old_fits = bulk.compute_msd_analysis(tracked, old_cal, max_lag=20,
                                                     min_length=5)
    units = Units(Calibration(pixel_size_nm=pixel_size_nm,
                              frame_interval_s=frame_interval_s), sigma_nm=35.0)
    new_curves, new_fits = compute.compute_msd_analysis(tracked, units, max_lag=20,
                                                        min_length=5)

    pd.testing.assert_frame_equal(old_curves.reset_index(drop=True),
                                  new_curves.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(old_fits.reset_index(drop=True),
                                  new_fits.reset_index(drop=True),
                                  check_exact=False, rtol=1e-6)
    assert (old_fits["motility_class"].values == new_fits["motility_class"].values).all()


def test_class_counts_match_the_old_script():
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    bulk = _load_script()
    tracked = _tracks(seed=3)
    _, old_fits = bulk.compute_msd_analysis(
        tracked, bulk.Calibration(sigma_nm=35.0), max_lag=20, min_length=5)
    _, new_fits = compute.compute_msd_analysis(
        tracked, Units(Calibration(), sigma_nm=35.0), max_lag=20, min_length=5)
    assert bulk.class_counts(old_fits) == compute.class_counts(new_fits)


def test_track_table_matches_the_old_columns():
    """Every column analyze_folder writes to track_analysis.csv must exist."""
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    tracked = _tracks(seed=4)
    units = Units(Calibration(pixel_size_nm=100, frame_interval_s=0.5), sigma_nm=35.0)
    _, fits = compute.compute_msd_analysis(tracked, units, max_lag=20, min_length=5)
    table = compute.track_table(tracked, None, fits, units)
    expected = {"track_id", "length", "total_displacement", "net_displacement",
                "mean_speed", "D", "alpha", "motility_class", "duration_s",
                "total_displacement_um", "net_displacement_um", "mean_speed_um_s"}
    assert expected <= set(table.columns)
