"""Group summaries and cross-group comparison for the msd analysis."""
import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.config import load_config


def drifting_movie(path, n_frames=45, size=96, n_spots=6, drift=0.8, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    starts = rng.uniform(10, size / 2, size=(n_spots, 2))
    stack = np.full((n_frames, size, size), 100.0)
    for t in range(n_frames):
        for cy, cx in starts + t * drift + rng.normal(0, 0.3, size=starts.shape):
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


def two_group_config(tmp_path, groups):
    return load_config(data={
        "inputs": [{"path": str(p), "label": label} for label, p in groups],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "calibration": {"pixel_size_nm": 100, "frame_interval_s": 0.5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"msd": {"max_lag": 10, "min_length": 5}},
    })


GROUP_FIGURES = ["motility_summary", "d_vs_alpha_summary", "length_summary",
                 "speed_summary", "length_vs_speed_summary",
                 "directed_displacement_summary", "directed_velocity_summary",
                 "directed_displacement_vs_velocity_summary"]


def test_group_summary_written_for_single_group(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=1)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=2)
    result = runner.run(two_group_config(tmp_path, [("Control", tmp_path / "ctrl")]))
    assert result.exit_code == 0
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    assert (summary / "motility_summary.csv").exists()
    counts = pd.read_csv(summary / "motility_summary.csv")
    assert len(counts) == 2                       # one row per movie
    assert {"movie", "constrained", "diffusive", "directed"} <= set(counts.columns)
    fits = pd.read_csv(summary / "group_fits.csv")
    assert set(fits["movie"]) == {"m1", "m2"}
    written = {p.stem for p in summary.glob("*.png")}
    assert {"motility_summary", "d_vs_alpha_summary", "length_summary",
            "speed_summary"} <= written


def test_group_summary_figure_selection(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=3)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg.analyses["msd"]["figures"] = ["motility_summary"]
    runner.run(cfg)
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    assert (summary / "motility_summary.png").exists()
    assert not (summary / "d_vs_alpha_summary.png").exists()


def test_group_summary_skips_movies_without_msd_outputs(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=4)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=5)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    runner.run(cfg)
    # Remove one movie's tables, then re-summarise via a second run with skip_existing.
    (tmp_path / "out" / "Control" / "m2" / "msd" / "msd_fits.csv").unlink()
    cfg2 = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg2.skip_existing = True
    assert runner.run(cfg2).exit_code == 0
    counts = pd.read_csv(tmp_path / "out" / "Control" / "_summary" / "msd" /
                         "motility_summary.csv")
    assert list(counts["movie"]) == ["m1"]
