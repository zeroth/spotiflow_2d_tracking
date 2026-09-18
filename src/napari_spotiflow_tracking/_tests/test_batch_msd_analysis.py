"""The msd analysis: registration, settings, tables and figure selection."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.analyses import base, get_analysis
from napari_spotiflow_tracking.batch.analyses.msd import MsdAnalysis
from napari_spotiflow_tracking.batch.config import ConfigError, load_config
from napari_spotiflow_tracking.batch.validation import validate


def synthetic_movie(path, n_frames=40, size=64, n_spots=6, seed=0):
    """Bright Gaussian spots doing a slow random walk — LoG finds them every frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    starts = rng.uniform(12, size - 12, size=(n_spots, 2))
    stack = np.full((n_frames, size, size), 100.0)
    pos = starts.copy()
    for t in range(n_frames):
        pos = pos + rng.normal(0, 0.4, size=pos.shape)
        for cy, cx in pos:
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


def msd_config(tmp_path, movie_dir, **analysis_settings):
    settings = {"max_lag": 10, "min_length": 5}
    settings.update(analysis_settings)
    return load_config(data={
        "inputs": [str(movie_dir)],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"msd": settings},
    })


def test_msd_is_registered():
    analysis = get_analysis("msd")
    assert isinstance(analysis, MsdAnalysis)
    assert analysis.requires_tracks is True


def test_tracking_preset_enables_msd():
    cfg = load_config(data={"modality": "tracking", "inputs": ["x.tif"],
                            "output_dir": "out"})
    assert "msd" in cfg.analyses
    validate(cfg)


def test_msd_requires_tracking():
    cfg = load_config(data={"inputs": ["x.tif"], "output_dir": "out",
                            "tracking": {"enabled": False},
                            "analyses": {"msd": {}}})
    with pytest.raises(ConfigError, match="analyses.msd needs tracking"):
        validate(cfg)


@pytest.mark.parametrize("bad, message", [
    ({"max_lag": 0}, "max_lag"),
    ({"min_length": 2}, "min_length"),
    ({"max_tracks_plot": 0}, "max_tracks_plot"),
    ({"sigma_nm": -1}, "sigma_nm"),
])
def test_invalid_settings(bad, message):
    cfg = load_config(data={"inputs": ["x.tif"], "output_dir": "out",
                            "analyses": {"msd": bad}})
    with pytest.raises(ConfigError, match=message):
        validate(cfg)


def test_run_writes_tables_and_summary_record(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    result = runner.run(msd_config(tmp_path, tmp_path / "g"))
    assert result.exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    curves = pd.read_csv(msd_dir / "msd_curves.csv")
    fits = pd.read_csv(msd_dir / "msd_fits.csv")
    table = pd.read_csv(msd_dir / "track_analysis.csv")
    assert list(curves.columns) == ["track_id", "lag", "msd", "sem", "n_pairs"]
    assert list(fits.columns) == ["track_id", "length", "n_lags", "D", "alpha",
                                  "motility_class"]
    assert len(fits) > 0 and curves["lag"].max() <= 10
    assert {"D", "alpha", "motility_class"} <= set(table.columns)
    row = result.rows[0]
    assert row["msd.n_fitted"] == len(fits)
    assert row["msd.diffusive"] + row["msd.constrained"] + row["msd.directed"] == len(fits)


def test_tables_none_writes_nothing(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", tables="none", figures="none")
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert not msd_dir.exists() or not list(msd_dir.glob("*.csv"))


def test_selected_table_only(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", tables=["msd_fits"], figures="none")
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "msd_fits.csv").exists()
    assert not (msd_dir / "msd_curves.csv").exists()


def test_movie_without_qualifying_tracks_still_succeeds(tmp_path):
    path = tmp_path / "g" / "empty.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), np.full((8, 32, 32), 100, np.uint16))
    result = runner.run(msd_config(tmp_path, tmp_path / "g"))
    assert result.exit_code == 0
    assert result.rows[0]["status"] == "OK"
    assert result.rows[0]["msd.n_fitted"] == 0


def test_list_outputs_shows_msd(capsys):
    from napari_spotiflow_tracking.batch import cli
    assert cli.main(["list-outputs", "msd"]) == 0
    out = capsys.readouterr().out
    assert "msd_fits" in out and "msd_curves" in out


CORE_FIGURES = ["msd_combined", "msd_regimes", "motility_classes", "d_vs_alpha"]


def test_core_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=CORE_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    for name in CORE_FIGURES:
        png = msd_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 5000, name


def test_unselected_figure_is_not_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["motility_classes"])
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "motility_classes.png").exists()
    assert not (msd_dir / "msd_combined.png").exists()


def test_figures_respect_output_formats(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["motility_classes"])
    cfg.outputs.formats = ["png", "svg"]
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "motility_classes.png").exists()
    assert (msd_dir / "motility_classes.svg").exists()


def test_figures_only_run_skips_track_table(tmp_path):
    """A figures-only run must not build (or write) track_analysis.csv."""
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", tables="none",
                     figures=["motility_classes"])
    result = runner.run(cfg)
    assert result.exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "motility_classes.png").exists()
    assert not (msd_dir / "track_analysis.csv").exists()


def test_core_figures_are_written_when_calibrated(tmp_path):
    """Calibrated run: the only path that exercises the 4*sigma^2 subtraction
    in plot_msd_regimes and the physical (µm, s) unit strings."""
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = load_config(data={
        "inputs": [str(tmp_path / "g")],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "calibration": {"pixel_size_nm": 100, "frame_interval_s": 0.5},
        "analyses": {"msd": {"max_lag": 10, "min_length": 5, "figures": CORE_FIGURES}},
    })
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    for name in CORE_FIGURES:
        png = msd_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 5000, name


def test_empty_fits_write_no_figure(tmp_path):
    path = tmp_path / "g" / "empty.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), np.full((8, 32, 32), 100, np.uint16))
    cfg = msd_config(tmp_path, tmp_path / "g", figures=CORE_FIGURES)
    assert runner.run(cfg).exit_code == 0
    assert not list((tmp_path / "out" / "g" / "empty" / "msd").glob("*.png"))


EXPLANATORY_FIGURES = ["msd_construction", "msd_archetypes", "msd_shape_comparison"]


def test_explanatory_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=6)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=EXPLANATORY_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    for name in EXPLANATORY_FIGURES:
        png = msd_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 5000, name


def test_archetypes_figure_is_full_width(tmp_path):
    """The archetype figure exports at a fixed manuscript width (three panels)."""
    from PIL import Image
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=7)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["msd_archetypes"])
    cfg.outputs.dpi = 100
    runner.run(cfg)
    with Image.open(tmp_path / "out" / "g" / "m" / "msd" / "msd_archetypes.png") as im:
        assert im.width > im.height * 2


def test_figure_module_uses_no_pyplot():
    import napari_spotiflow_tracking.batch.analyses.msd.figures_movie as fm
    source = Path(fm.__file__).read_text(encoding="utf-8")
    assert "pyplot" not in source and "matplotlib.use" not in source


def test_motility_classes_uses_theme_colours(tmp_path):
    """The bars must be the theme's class colours, not matplotlib defaults."""
    from matplotlib.colors import to_hex
    from napari_spotiflow_tracking.batch import theme
    from napari_spotiflow_tracking.batch.analyses.msd import compute, figures_movie
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    tracked = pd.concat([
        pd.DataFrame({"frame": range(30), "y": np.cumsum(np.random.default_rng(0).normal(0, 1, 30)),
                      "x": np.cumsum(np.random.default_rng(1).normal(0, 1, 30)), "particle": 0}),
    ])
    _, fits = compute.compute_msd_analysis(tracked, Units(Calibration()), max_lag=10,
                                           min_length=5)
    captured = {}
    figures_movie.plot_motility_classes(fits, tmp_path / "motility_classes",
                                        ["png"], 40, _capture=captured)
    drawn = {to_hex(c).lower() for c in captured["colors"]}
    assert drawn <= {to_hex(c).lower() for c in theme.MOTILITY_COLORS.values()}


DISTRIBUTION_FIGURES = ["length_hist", "path_length_hist", "speed_hist",
                        "length_per_track", "speed_per_track",
                        "directed_displacement", "directed_velocity",
                        "directed_displacement_vs_velocity"]


def test_distribution_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=4)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=DISTRIBUTION_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    written = {p.stem for p in msd_dir.glob("*.png")}
    # The directed figures only exist when some track is classified directed.
    assert {"length_hist", "path_length_hist", "speed_hist", "length_per_track",
            "speed_per_track"} <= written
    assert written <= set(DISTRIBUTION_FIGURES)


def test_directed_figures_when_directed_tracks_exist(tmp_path, monkeypatch):
    """A movie of steadily drifting spots yields directed tracks."""
    import numpy as np
    rng = np.random.default_rng(2)
    size, n_frames = 96, 40
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    starts = rng.uniform(10, 40, size=(5, 2))
    for t in range(n_frames):
        for cy, cx in starts + t * 1.2:
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path = tmp_path / "g" / "drift.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))

    cfg = msd_config(tmp_path, tmp_path / "g", figures=DISTRIBUTION_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "drift" / "msd"
    fits = pd.read_csv(msd_dir / "msd_fits.csv")
    assert (fits["motility_class"] == "directed").any()
    assert (msd_dir / "directed_displacement.png").exists()
    assert (msd_dir / "directed_velocity.png").exists()
    assert (msd_dir / "directed_displacement_vs_velocity.png").exists()


def test_max_tracks_plot_limits_per_track_figures(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_spots=8, seed=5)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["length_per_track"],
                     max_tracks_plot=2)
    assert runner.run(cfg).exit_code == 0
    assert (tmp_path / "out" / "g" / "m" / "msd" / "length_per_track.png").exists()


def test_max_tracks_plot_caps_lines_drawn(tmp_path):
    """max_tracks_plot must actually cap the number of lines drawn, not just
    let the figure exist regardless of whether the setting is honoured."""
    from napari_spotiflow_tracking.batch.analyses.msd import figures_movie
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    tracked = pd.concat([
        pd.DataFrame({"frame": range(10), "y": np.arange(10, dtype=float) + i,
                      "x": np.arange(10, dtype=float) * 0.5, "particle": i})
        for i in range(5)
    ])
    captured = {}
    figures_movie.plot_length_per_track(
        tracked, Units(Calibration()), max_tracks=2,
        stem=tmp_path / "length_per_track", formats=["png"], dpi=40,
        _capture=captured)
    assert (tmp_path / "length_per_track.png").exists()
    assert captured["n_lines"] == 2
