"""Group summaries and cross-group comparison for the msd analysis."""
import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.analyses.msd import figures_group
from napari_spotiflow_tracking.batch.analyses.msd.units import Units
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


def two_group_config(tmp_path, groups, calibration="default"):
    """``calibration=None`` omits the section entirely, for an uncalibrated run."""
    data = {
        "inputs": [{"path": str(p), "label": label} for label, p in groups],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"msd": {"max_lag": 10, "min_length": 5}},
    }
    if calibration == "default":
        data["calibration"] = {"pixel_size_nm": 100, "frame_interval_s": 0.5}
    elif calibration is not None:
        data["calibration"] = calibration
    return load_config(data=data)


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

    # The two tables must agree on how many tracks each movie contributed.
    n_tracks_by_movie = {k: int(v) for k, v in
                         counts.set_index("movie")["n_tracks"].items()}
    fits_rows_by_movie = {k: int(v) for k, v in fits.groupby("movie").size().items()}
    assert n_tracks_by_movie == fits_rows_by_movie

    written = {p.stem for p in summary.glob("*.png")}
    assert set(GROUP_FIGURES) <= written


def test_group_summary_uncalibrated(tmp_path):
    """The uncalibrated (px/frames) branches of read_movie_frames/summarize_group."""
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=15)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=16)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")], calibration=None)
    assert runner.run(cfg).exit_code == 0
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    written = {p.stem for p in summary.glob("*.png")}
    assert set(GROUP_FIGURES) <= written
    assert (summary / "motility_summary.csv").exists()
    assert (summary / "group_fits.csv").exists()


def test_group_summary_figure_selection(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=3)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg.analyses["msd"]["figures"] = ["motility_summary"]
    runner.run(cfg)
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    assert (summary / "motility_summary.png").exists()
    assert not (summary / "d_vs_alpha_summary.png").exists()


def test_group_summary_out_dir_not_created_when_nothing_selected(tmp_path):
    """create out_dir only when it writes something -- nothing selected, nothing appears."""
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=17)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg.analyses["msd"]["figures"] = "none"
    cfg.analyses["msd"]["tables"] = ["msd_fits"]
    assert runner.run(cfg).exit_code == 0
    assert not (tmp_path / "out" / "Control" / "_summary").exists()


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


def test_group_summary_keeps_fits_only_movies(tmp_path):
    """A movie with msd_fits.csv but no track_analysis.csv still feeds the
    fits-only summaries (motility_summary, d_vs_alpha_summary), matching
    bulk:_read_group_frames, which reads fits and props independently."""
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=18)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=19)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg.analyses["msd"]["tables"] = ["msd_fits"]  # track_analysis.csv never written
    assert runner.run(cfg).exit_code == 0
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"

    assert (summary / "motility_summary.png").exists()
    assert (summary / "d_vs_alpha_summary.png").exists()
    # Every props-dependent figure has nothing to read and must skip.
    assert not (summary / "length_summary.png").exists()
    assert not (summary / "speed_summary.png").exists()
    assert not (summary / "length_vs_speed_summary.png").exists()
    assert not (summary / "directed_displacement_summary.png").exists()
    assert not (summary / "directed_velocity_summary.png").exists()
    assert not (summary / "directed_displacement_vs_velocity_summary.png").exists()


def test_group_summary_recomputes_duration_when_duration_s_missing(tmp_path):
    """A calibrated summary run over a movie whose track_analysis.csv predates
    (or was written without) duration_s must not silently plot frame counts
    under a seconds axis."""
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=20)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=21)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    assert runner.run(cfg).exit_code == 0

    m2_props_path = tmp_path / "out" / "Control" / "m2" / "msd" / "track_analysis.csv"
    props = pd.read_csv(m2_props_path)
    assert "duration_s" in props.columns
    max_length = int(props["length"].max())
    props.drop(columns=["duration_s"]).to_csv(m2_props_path, index=False)

    cfg2 = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg2.skip_existing = True
    assert runner.run(cfg2).exit_code == 0  # the group hook re-reads m2 without duration_s

    units = Units(cfg2.calibration, 35.0)
    per_movie = figures_group.read_movie_frames(
        [tmp_path / "out" / "Control" / "m2"], cfg2.analyses["msd"]["min_length"], units)
    _, _, props_out, _ = per_movie[0]
    assert props_out is not None
    expected_max = (max_length - 1) * cfg2.calibration.frame_interval_s
    assert props_out["_dur"].max() == pytest.approx(expected_max)
    # Not the raw frame count -- that would be the (wrong) pre-fix value.
    assert props_out["_dur"].max() != pytest.approx(max_length)


def test_comparison_written_for_two_groups(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", drift=0.2, seed=6)
    drifting_movie(tmp_path / "drug" / "m1.tif", drift=1.5, seed=7)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    assert runner.run(cfg).exit_code == 0
    comparison = tmp_path / "out" / "_comparison" / "msd"
    table = pd.read_csv(comparison / "comparison_summary.csv")
    assert set(table["group"]) == {"Control", "Drug"}
    written = {p.stem for p in comparison.glob("*.png")}
    assert {"motility_comparison", "length_comparison", "speed_comparison"} <= written
    assert {"length_comparison_density", "speed_comparison_density"} <= written
    assert (comparison / "harmonised").is_dir()
    harmonised = {p.stem for p in (comparison / "harmonised").glob("*.png")}
    assert any(name.startswith("Control_") for name in harmonised)
    assert any(name.startswith("Drug_") for name in harmonised)


def test_no_comparison_for_a_single_group(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=8)
    runner.run(two_group_config(tmp_path, [("Control", tmp_path / "ctrl")]))
    assert not (tmp_path / "out" / "_comparison").exists()


def test_group_colours_follow_input_order(tmp_path):
    """A group whose movies all failed must not shift the others' colours."""
    from napari_spotiflow_tracking.batch import theme
    from napari_spotiflow_tracking.batch.inputs import color_pins, input_label
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=9)
    drifting_movie(tmp_path / "drug" / "m1.tif", seed=10)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    colors = theme.group_colors([input_label(s) for s in cfg.inputs], color_pins(cfg))
    assert colors["Control"] == theme.GROUP_COLORS[0]
    assert colors["Drug"] == theme.GROUP_COLORS[1]


def test_comparison_figure_selection(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=11)
    drifting_movie(tmp_path / "drug" / "m1.tif", seed=12)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    cfg.analyses["msd"]["figures"] = ["motility_comparison"]
    runner.run(cfg)
    comparison = tmp_path / "out" / "_comparison" / "msd"
    assert (comparison / "motility_comparison.png").exists()
    assert not (comparison / "length_comparison.png").exists()
