"""Input expansion into movie jobs, and the per-movie context."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch.config import ConfigError, load_config
from napari_spotiflow_tracking.batch.context import MovieContext
from napari_spotiflow_tracking.batch.inputs import MovieJob, color_pins, expand_inputs


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _cfg(inputs):
    return load_config(data={"inputs": inputs, "output_dir": "out"})


def test_folder_input_uses_folder_name_and_pattern(tmp_path):
    _touch(tmp_path / "ctrl" / "b.tif")
    _touch(tmp_path / "ctrl" / "a.tif")
    _touch(tmp_path / "ctrl" / "notes.txt")
    jobs = expand_inputs(_cfg([str(tmp_path / "ctrl")]))
    assert [(j.group, j.movie_id) for j in jobs] == [("ctrl", "a"), ("ctrl", "b")]
    assert jobs[0].key == "ctrl/a"


def test_file_input_and_label(tmp_path):
    movie = _touch(tmp_path / "field 1.tif")
    jobs = expand_inputs(_cfg([{"path": str(movie), "label": "Drug"}]))
    assert jobs == [MovieJob(group="Drug", movie_id="field 1", path=movie)]
    assert expand_inputs(_cfg([str(movie)]))[0].group == "field 1"


def test_recursive_ids_include_subfolders(tmp_path):
    _touch(tmp_path / "exp" / "day1" / "m.tif")
    _touch(tmp_path / "exp" / "day2" / "m.tif")
    jobs = expand_inputs(_cfg([{"path": str(tmp_path / "exp"), "recursive": True}]))
    assert [j.movie_id for j in jobs] == ["day1__m", "day2__m"]


def test_errors_are_collected(tmp_path):
    (tmp_path / "empty").mkdir()
    _touch(tmp_path / "a" / "x.tif")
    _touch(tmp_path / "b" / "a" / "y.tif")
    with pytest.raises(ConfigError) as info:
        expand_inputs(_cfg([
            str(tmp_path / "missing"),
            str(tmp_path / "empty"),
            str(tmp_path / "a"),
            str(tmp_path / "b" / "a"),
            {"path": str(tmp_path / "a"), "label": "bad:name"},
        ]))
    text = str(info.value)
    assert "inputs[0]" in text and "does not exist" in text
    assert "inputs[1]" in text and "no files match" in text
    assert "inputs[3]" in text and "label 'a' is used by an earlier input" in text
    assert "inputs[4]" in text and "cannot contain" in text


def test_dotted_folder_name_is_label(tmp_path):
    _touch(tmp_path / "field_1_20_7.5_l_20" / "m.tif")
    jobs = expand_inputs(_cfg([str(tmp_path / "field_1_20_7.5_l_20")]))
    assert jobs[0].group == "field_1_20_7.5_l_20"


def test_color_pins(tmp_path):
    movie = _touch(tmp_path / "m.tif")
    cfg = _cfg([{"path": str(movie), "label": "A", "color": "#123456"}])
    assert color_pins(cfg) == {"A": "#123456"}


def test_context_image_selection(tmp_path):
    raw = np.zeros((2, 4, 4))
    pre = np.ones((2, 4, 4))
    ctx = MovieContext(job=MovieJob("g", "m", tmp_path / "m.tif"), out_dir=tmp_path,
                       config=_cfg([]), raw=raw, preprocessed=pre,
                       spots=pd.DataFrame())
    assert ctx.image("raw") is raw
    assert ctx.image("preprocessed") is pre
    with pytest.raises(ValueError):
        ctx.image("denoised")
    assert ctx.analysis_dir("msd") == tmp_path / "msd"
    assert (tmp_path / "msd").is_dir()
