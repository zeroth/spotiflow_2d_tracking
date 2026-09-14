"""Tests for physical units on the bulk script's per-file figures.

The four per-track figures predate the ``--pixel-size-nm`` /
``--frame-interval-s`` flags and used to draw pixels and frames whatever the
calibration said. These check that a calibrated run puts µm and seconds on
both the axis labels and the data, and that an uncalibrated run is unchanged.

Figures are inspected rather than merely written: ``_save`` is patched to hand
the figure back, so the assertions are on what was actually drawn.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bulk_track_analysis.py"


def _load_script():
    if "bulk_track_analysis" in sys.modules:
        return sys.modules["bulk_track_analysis"]
    spec = importlib.util.spec_from_file_location("bulk_track_analysis", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bulk_track_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


bulk = _load_script()

PX_NM = 92.0
DT_S = 0.005004
UM = PX_NM / 1000.0


@pytest.fixture
def cal_phys():
    return bulk.Calibration(pixel_size_nm=PX_NM, frame_interval_s=DT_S, sigma_nm=0.0)


@pytest.fixture
def cal_px():
    return bulk.Calibration()


@pytest.fixture
def capture(monkeypatch):
    """Patch _save so a test can look at the figure that was drawn."""
    seen = {}

    def fake_save(fig, out_path):
        seen["fig"] = fig
        seen["ax"] = fig.axes[0]
        seen["path"] = out_path
        out_path.write_bytes(b"")   # the caller still expects a file

    monkeypatch.setattr(bulk, "_save", fake_save)
    return seen


@pytest.fixture
def hist_values(monkeypatch):
    """Record the array each Axes.hist call is given, then plot as normal.

    Bin centres cannot stand in for the data: matplotlib pads the range of a
    degenerate (single-value) histogram, so the centre of the occupied bin is
    not the value. This captures what the production code actually handed to
    hist, which is the thing under test.
    """
    import matplotlib.axes
    original = matplotlib.axes.Axes.hist
    seen = []

    def spy(self, x, *args, **kwargs):
        seen.append(np.asarray(x, dtype=float))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", spy)
    return seen


@pytest.fixture
def tracked():
    """One 3-frame track moving 3 px then 4 px along x."""
    return pd.DataFrame({
        "frame": [0, 1, 2],
        "y": [0.0, 0.0, 0.0],
        "x": [0.0, 3.0, 7.0],
        "particle": [1, 1, 1],
    })


@pytest.fixture
def props(tracked):
    from napari_spotiflow_tracking._track_properties import compute_track_properties
    return compute_track_properties(tracked)


# ── axis labels ─────────────────────────────────────────────────────


@pytest.mark.parametrize("kind,expected", [
    ("duration", "Track duration (s)"),
    ("path_length", "Path length (µm)"),
    ("speed", "Mean speed (µm/s)"),
    ("inst_speed", "Instantaneous speed (µm/s)"),
    ("cum_disp", "Cumulative displacement (µm)"),
    ("time", "Time (s)"),
])
def test_labels_are_physical_when_calibrated(kind, expected, cal_phys):
    assert bulk._axis_label(kind, cal_phys) == expected


@pytest.mark.parametrize("kind,expected", [
    ("duration", "Track length (frames)"),
    ("path_length", "Path length (px)"),
    ("speed", "Mean speed (px / frame)"),
    ("inst_speed", "Instantaneous speed (px / frame)"),
    ("cum_disp", "Cumulative displacement (px)"),
    ("time", "Frame"),
])
def test_labels_keep_pixel_units_when_uncalibrated(kind, expected, cal_px):
    """An uncalibrated run must look exactly as it did before this change."""
    assert bulk._axis_label(kind, cal_px) == expected


# ── the data that gets drawn ────────────────────────────────────────


def test_speed_per_track_draws_micrometres_per_second(
    tracked, cal_phys, capture, tmp_path):
    bulk._plot_speed_per_track(tracked, cal_phys, tmp_path / "s.png", 10)
    line = capture["ax"].lines[0]
    np.testing.assert_allclose(line.get_ydata(),
                               np.array([3.0, 4.0]) * UM / DT_S)
    np.testing.assert_allclose(line.get_xdata(), np.array([1, 2]) * DT_S)
    assert capture["ax"].get_ylabel() == "Instantaneous speed (µm/s)"


def test_length_per_track_draws_micrometres_against_seconds(
    tracked, cal_phys, capture, tmp_path):
    bulk._plot_length_per_track(tracked, cal_phys, tmp_path / "l.png", 10)
    line = capture["ax"].lines[0]
    np.testing.assert_allclose(line.get_ydata(),
                               np.array([0.0, 3.0, 7.0]) * UM)
    np.testing.assert_allclose(line.get_xdata(), np.array([0, 1, 2]) * DT_S)


def test_speed_per_track_is_unchanged_when_uncalibrated(
    tracked, cal_px, capture, tmp_path):
    bulk._plot_speed_per_track(tracked, cal_px, tmp_path / "s.png", 10)
    line = capture["ax"].lines[0]
    np.testing.assert_allclose(line.get_ydata(), [3.0, 4.0])
    np.testing.assert_allclose(line.get_xdata(), [1, 2])


def test_length_hist_draws_seconds(props, cal_phys, capture, hist_values,
                                   tmp_path):
    bulk._plot_length_hist(props, cal_phys, tmp_path / "h.png", 2)
    assert capture["ax"].get_xlabel() == "Track duration (s)"
    # a 3-frame track spans 2 intervals, not 3
    np.testing.assert_allclose(hist_values[0], [2 * DT_S])


def test_length_hist_states_the_cut_in_frames(props, cal_phys, capture, tmp_path):
    """--min-track-length is given in frames, so the annotation must say frames
    even on a calibrated figure -- with the seconds equivalent alongside."""
    bulk._plot_length_hist(props, cal_phys, tmp_path / "h.png", 20)
    texts = [t.get_text() for t in capture["ax"].texts]
    assert any("20 frames" in t for t in texts)
    assert any("0.095 s" in t for t in texts)  # 19 intervals * 0.005004 s


def test_speed_hist_draws_micrometres_per_second(
    props, cal_phys, capture, hist_values, tmp_path):
    bulk._plot_speed_hist(props, cal_phys, tmp_path / "v.png")
    assert capture["ax"].get_xlabel() == "Mean speed (µm/s)"
    # 7 px of path over 2 intervals
    np.testing.assert_allclose(hist_values[0], [7 * UM / (2 * DT_S)])


def test_path_length_hist_draws_micrometres(props, cal_phys, capture,
                                           hist_values, tmp_path):
    bulk._plot_path_length_hist(props, cal_phys, tmp_path / "p.png")
    assert capture["ax"].get_xlabel() == "Path length (µm)"
    np.testing.assert_allclose(hist_values[0], [7 * UM])


# ── the figure set ──────────────────────────────────────────────────


@pytest.mark.parametrize("calibrated", [True, False])
def test_every_per_track_figure_is_written(tracked, props, tmp_path, calibrated):
    cal = (bulk.Calibration(pixel_size_nm=PX_NM, frame_interval_s=DT_S)
           if calibrated else bulk.Calibration())
    bulk._plot_length_hist(props, cal, tmp_path / "length_hist.png", 2)
    bulk._plot_path_length_hist(props, cal, tmp_path / "path_length_hist.png")
    bulk._plot_speed_hist(props, cal, tmp_path / "speed_hist.png")
    bulk._plot_length_per_track(tracked, cal, tmp_path / "length_per_track.png", 10)
    bulk._plot_speed_per_track(tracked, cal, tmp_path / "speed_per_track.png", 10)
    for name in ("length_hist", "path_length_hist", "speed_hist",
                 "length_per_track", "speed_per_track"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_empty_props_still_writes_a_figure(cal_phys, tmp_path):
    empty = pd.DataFrame(columns=["track_id", "length", "total_displacement",
                                  "net_displacement", "mean_speed"])
    bulk._plot_path_length_hist(empty, cal_phys, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 0

# ── uncalibrated figures keep their old data ────────────────────────


def test_length_hist_draws_raw_frames_when_uncalibrated(
    props, cal_px, capture, hist_values, tmp_path):
    bulk._plot_length_hist(props, cal_px, tmp_path / "h.png", 2)
    np.testing.assert_allclose(hist_values[0], [3.0])   # frame count, not 2


def test_path_length_hist_draws_pixels_when_uncalibrated(
    props, cal_px, capture, hist_values, tmp_path):
    bulk._plot_path_length_hist(props, cal_px, tmp_path / "p.png")
    np.testing.assert_allclose(hist_values[0], [7.0])


# ── analyze_folder: the whole per-file output set ────────────────────


def _args(tmp_path, calibrated):
    """A parsed Args for analyze_folder, with or without a calibration."""
    argv = ["--in", "x.tif", str(tmp_path), "--min-track-length", "3",
            "--msd-max-lag", "3"]
    if calibrated:
        argv += ["--pixel-size-nm", str(PX_NM),
                 "--frame-interval-s", str(DT_S), "--sigma-nm", "0"]
    return bulk.parse_args(argv)


@pytest.fixture
def walk():
    """Twelve tracks of 8 frames each, random-walking, so the MSD pass fits."""
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(12):
        pos = np.zeros(2)
        for f in range(8):
            rows.append({"frame": f, "y": pos[0], "x": pos[1], "particle": tid})
            pos = pos + rng.normal(0, 2.0, 2)
    return pd.DataFrame(rows)


def test_analyze_folder_writes_the_path_length_figure(walk, tmp_path):
    bulk.analyze_folder(tmp_path, walk, None, _args(tmp_path, True))
    assert (tmp_path / "path_length_hist.png").stat().st_size > 0


def test_analyze_folder_adds_physical_columns_to_the_csv(walk, tmp_path):
    bulk.analyze_folder(tmp_path, walk, None, _args(tmp_path, True))
    df = pd.read_csv(tmp_path / "track_analysis.csv")
    assert {"duration_s", "total_displacement_um", "net_displacement_um",
            "mean_speed_um_s"} <= set(df.columns)
    # the µm column is the px column times the pixel size, not a recomputation
    np.testing.assert_allclose(df["total_displacement_um"],
                               df["total_displacement"] * UM)


def test_analyze_folder_keeps_the_csv_in_pixels_when_uncalibrated(walk, tmp_path):
    """--summary-only and --compare read this table back and scale it
    themselves, so an uncalibrated run must not gain unit-suffixed columns."""
    bulk.analyze_folder(tmp_path, walk, None, _args(tmp_path, False))
    df = pd.read_csv(tmp_path / "track_analysis.csv")
    assert not [c for c in df.columns if c.endswith(("_um", "_um_s", "_s"))]


def test_analyze_folder_still_cuts_on_frames(walk, tmp_path):
    """length stays a frame count, because --min-track-length cuts on it."""
    bulk.analyze_folder(tmp_path, walk, None, _args(tmp_path, True))
    df = pd.read_csv(tmp_path / "track_analysis.csv")
    assert (df["length"] == 8).all()
    np.testing.assert_allclose(df["duration_s"], 7 * DT_S)

# ── run-level summaries use the same duration convention ────────────


def _write_run(root, stems, n_frames=8, n_tracks=6):
    """Two fake processed folders, each with the CSVs write_run_summary reads."""
    rows = []
    for stem in stems:
        folder = root / stem
        folder.mkdir(parents=True, exist_ok=True)
        props = pd.DataFrame({
            "track_id": range(n_tracks),
            "length": [n_frames] * n_tracks,
            "total_displacement": np.linspace(10.0, 20.0, n_tracks),
            "net_displacement": np.linspace(5.0, 10.0, n_tracks),
            "mean_speed": np.linspace(1.0, 2.0, n_tracks),
            "D": np.linspace(0.1, 0.5, n_tracks),
            "alpha": np.linspace(0.5, 1.5, n_tracks),
            "motility_class": ["diffusive"] * n_tracks,
        })
        props.to_csv(folder / "track_analysis.csv", index=False)
        props[["track_id", "length", "D", "alpha", "motility_class"]].assign(
            n_lags=3).to_csv(folder / "msd_fits.csv", index=False)
        rows.append({"stem": stem,
                     "counts": {"diffusive": n_tracks,
                                "constrained": 0, "directed": 0}})
    return rows


@pytest.fixture
def metric_spy(monkeypatch):
    """Record what write_run_summary asks _plot_metric_summary to plot."""
    original = bulk._plot_metric_summary
    calls = {}

    def spy(per_file, column, xlabel, scale, out_path, **kwargs):
        calls[Path(out_path).name] = (per_file, column, xlabel, scale)
        return original(per_file, column, xlabel, scale, out_path, **kwargs)

    monkeypatch.setattr(bulk, "_plot_metric_summary", spy)
    return calls


def _metric(calls, suffix):
    """One recorded _plot_metric_summary call, found by filename suffix.

    _summary_path prefixes every root-level figure with the run folder's name,
    so the exact filename depends on tmp_path.
    """
    hits = [v for k, v in calls.items() if k.endswith(suffix)]
    assert len(hits) == 1, f"{suffix} not among {sorted(calls)}"
    return hits[0]


def test_length_summary_plots_elapsed_time_not_frame_count(
    tmp_path, metric_spy):
    """A run of 8-frame tracks lasts 7 intervals, so the summary must plot
    7*dt -- not 8*dt, which is what scaling the raw frame count gives."""
    rows = _write_run(tmp_path, ["a", "b"], n_frames=8)
    args = bulk.parse_args(["--summary-only", str(tmp_path),
                            "--min-track-length", "3",
                            "--pixel-size-nm", str(PX_NM),
                            "--frame-interval-s", str(DT_S)])
    bulk.write_run_summary(tmp_path, rows, args)

    per_file, column, xlabel, scale = _metric(metric_spy, "length_summary.png")
    values = bulk._experiment_values(per_file, column, scale)
    assert values, "the length summary was handed no data"
    for _stem, arr in values:
        np.testing.assert_allclose(arr, 7 * DT_S)
    assert xlabel == "Track duration (s)"


def test_length_summary_stays_in_frames_when_uncalibrated(tmp_path, metric_spy):
    rows = _write_run(tmp_path, ["a"], n_frames=8)
    args = bulk.parse_args(["--summary-only", str(tmp_path),
                            "--min-track-length", "3"])
    bulk.write_run_summary(tmp_path, rows, args)

    per_file, column, xlabel, scale = _metric(metric_spy, "length_summary.png")
    for _stem, arr in bulk._experiment_values(per_file, column, scale):
        np.testing.assert_allclose(arr, 8.0)
    assert xlabel == "Track length (frames)"


def test_length_summary_works_on_a_run_written_before_duration_existed(
    tmp_path, metric_spy):
    """--summary-only over an older folder has only the pixel columns to work
    from; the duration must be recomputed rather than the figure left empty."""
    rows = _write_run(tmp_path, ["a"], n_frames=12)
    props_path = tmp_path / "a" / "track_analysis.csv"
    assert "duration_s" not in pd.read_csv(props_path).columns
    args = bulk.parse_args(["--summary-only", str(tmp_path),
                            "--min-track-length", "3",
                            "--pixel-size-nm", str(PX_NM),
                            "--frame-interval-s", str(DT_S)])
    bulk.write_run_summary(tmp_path, rows, args)

    per_file, column, _xlabel, scale = _metric(metric_spy, "length_summary.png")
    for _stem, arr in bulk._experiment_values(per_file, column, scale):
        np.testing.assert_allclose(arr, 11 * DT_S)


def test_length_vs_speed_summary_uses_the_same_duration(tmp_path):
    """length_vs_speed_summary must not disagree with length_summary."""
    props = pd.DataFrame({
        "track_id": [0, 1],
        "length": [8, 8],
        "total_displacement": [10.0, 20.0],
        "net_displacement": [5.0, 10.0],
        "mean_speed": [1.0, 2.0],
        "motility_class": ["diffusive", "directed"],
    })
    cal = bulk.Calibration(pixel_size_nm=PX_NM, frame_interval_s=DT_S)
    frames = bulk._length_speed_metrics(
        [("a", bulk._props_in_working_units(props, cal))], cal)
    assert frames, "no experiment survived"
    for _stem, df in frames:
        np.testing.assert_allclose(df["_len"], 7 * DT_S)
