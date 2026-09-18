"""spotiflow-batch subcommands."""
import warnings

import numpy as np
import pytest
import tifffile
import yaml

from napari_spotiflow_tracking.batch import analyses, cli
from napari_spotiflow_tracking.batch.analyses import Analysis, register
from napari_spotiflow_tracking.batch.config import ConfigWarning, load_config
from napari_spotiflow_tracking.batch.outputs import OutputSpec
from napari_spotiflow_tracking.batch.validation import validate


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(analyses.base, "REGISTRY", {})


@pytest.mark.parametrize("modality", ["tracking", "papa", "bleaching"])
def test_init_config_is_valid_and_warning_free(modality, capsys):
    assert cli.main(["init-config", modality]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"# spotiflow-batch config — {modality} preset")
    data = yaml.safe_load(text)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        cfg = load_config(data=data)
    assert cfg.modality == modality
    validate(cfg)


def test_list_outputs(empty_registry, capsys):
    assert cli.main(["list-outputs"]) == 0
    assert "No analyses are registered yet" in capsys.readouterr().out

    class Demo(Analysis):
        name = "demo"
        catalogue = (OutputSpec("plot", "movie", "figure", "A plot of things"),)

    register(Demo())
    assert cli.main(["list-outputs", "demo"]) == 0
    out = capsys.readouterr().out
    assert "demo" in out and "plot" in out and "A plot of things" in out
    assert cli.main(["list-outputs", "nope"]) == 2


def test_validate_reports_movies(tmp_path, empty_registry, capsys):
    movie = tmp_path / "g" / "m.tif"
    movie.parent.mkdir()
    tifffile.imwrite(str(movie), np.zeros((2, 8, 8), np.uint16))
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": [str(movie.parent)],
                                      "output_dir": str(tmp_path / "out")}))
    assert cli.main(["validate", str(config)]) == 0
    assert "config OK: 1 movie(s) in 1 group(s)" in capsys.readouterr().out


def test_config_error_exit_code(tmp_path, empty_registry, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": [], "output_dir": ""}))
    assert cli.main(["validate", str(config)]) == 2
    assert "ERROR: invalid config" in capsys.readouterr().err
    assert cli.main(["run", str(config), "--set", "broken"]) == 2


def test_run_uses_set_overrides(tmp_path, empty_registry, monkeypatch):
    seen = {}

    class Result:
        exit_code = 0

    def fake_run(cfg):
        seen["cfg"] = cfg
        return Result()

    monkeypatch.setattr(cli, "run", fake_run)
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": ["x.tif"], "output_dir": "out"}))
    assert cli.main(["run", str(config), "--set", "tracking.memory=9",
                     "--set", "device=cpu"]) == 0
    assert seen["cfg"].tracking.memory == 9 and seen["cfg"].device == "cpu"


def test_missing_config_file_exit_code(tmp_path, empty_registry, capsys):
    assert cli.main(["validate", str(tmp_path / "nope.yaml")]) == 2
    assert "ERROR: cannot read config" in capsys.readouterr().err


def test_malformed_yaml_exit_code(tmp_path, empty_registry, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("inputs: [unclosed\noutput_dir: : :\n", encoding="utf-8")
    assert cli.main(["validate", str(config)]) == 2
    assert "ERROR: cannot read config" in capsys.readouterr().err
