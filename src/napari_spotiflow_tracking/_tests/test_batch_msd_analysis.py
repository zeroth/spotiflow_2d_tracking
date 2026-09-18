"""The msd analysis: registration, settings, tables and figure selection."""
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
