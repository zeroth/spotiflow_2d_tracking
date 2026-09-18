"""End-to-end runner on tiny synthetic movies with LoG detection (no GPU, no model)."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from napari_spotiflow_tracking.batch import analyses, runner
from napari_spotiflow_tracking.batch.analyses import Analysis, register
from napari_spotiflow_tracking.batch.config import load_config
from napari_spotiflow_tracking.batch.outputs import OutputSpec


def _movie(path, n_frames=6, size=48):
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    for t in range(n_frames):
        for cy, cx in ((12, 12), (34, 30)):
            stack[t] += 1000 * np.exp(-((yy - cy) ** 2 + (xx - (cx + 0.5 * t)) ** 2) / 8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


class Recorder(Analysis):
    name = "recorder"
    requires_tracks = True
    catalogue = (OutputSpec("count", "movie", "table", "Spot count"),)

    def __init__(self):
        self.movies, self.groups, self.compared = [], [], []
        self.hook_cfgs = []

    def default_settings(self):
        return {"size": 3}

    def run_movie(self, ctx, settings, select):
        if "broken" in ctx.job.movie_id:
            raise RuntimeError("boom")
        self.movies.append(ctx.job.key)
        if select.table("count"):
            pd.DataFrame({"n": [len(ctx.spots)]}).to_csv(ctx.analysis_dir(self.name) / "count.csv",
                                                         index=False)
        return {"n_particles": int(ctx.tracks["particle"].nunique())}

    def summarize_group(self, group, movie_dirs, settings, select, out_dir, cfg):
        self.groups.append((group, sorted(d.name for d in movie_dirs)))
        self.hook_cfgs.append(cfg)

    def compare_groups(self, groups, settings, select, out_dir, cfg):
        self.compared.append(sorted(groups))
        self.hook_cfgs.append(cfg)


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(analyses.base, "REGISTRY", {})
    return register(Recorder())


def _cfg(tmp_path, inputs, **extra):
    data = {
        "inputs": inputs,
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"recorder": {}},
    }
    data.update(extra)
    return load_config(data=data)


def test_single_group_run(tmp_path, recorder):
    _movie(tmp_path / "ctrl" / "m1.tif")
    _movie(tmp_path / "ctrl" / "m2.tif")
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "ctrl")]))
    out = tmp_path / "out"

    assert result.exit_code == 0
    assert recorder.movies == ["ctrl/m1", "ctrl/m2"]
    assert recorder.groups == [("ctrl", ["m1", "m2"])]
    assert recorder.compared == []
    movie = out / "ctrl" / "m1"
    for name in ("spots.csv", "tracks.csv", "movie_status.json", "recorder/count.csv"):
        assert (movie / name).exists(), name
    summary = pd.read_csv(out / "run_summary.csv")
    assert summary.columns[:9].tolist() == runner.SUMMARY_COLUMNS
    assert summary["status"].tolist() == ["OK", "OK"]
    assert summary["n_frames"].tolist() == [6, 6]
    assert (summary["n_spots"] > 0).all()
    assert summary["recorder.n_particles"].tolist() == [2, 2]
    assert (out / "resolved_config.yaml").exists()
    assert (out / "run.log").exists()
    assert (out / "run_overview.png").exists()


def test_failure_is_isolated_and_sets_exit_code(tmp_path, recorder):
    _movie(tmp_path / "a" / "good.tif")
    _movie(tmp_path / "a" / "broken.tif")
    _movie(tmp_path / "b" / "other.tif")
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "a"), str(tmp_path / "b")]))

    assert result.exit_code == 1
    rows = {r["movie"]: r for r in result.rows}
    assert rows["broken"]["status"] == "FAIL"
    assert rows["broken"]["error"] == "RuntimeError: boom"
    assert rows["good"]["status"] == "OK" and rows["other"]["status"] == "OK"
    assert recorder.groups == [("a", ["good"]), ("b", ["other"])]
    assert recorder.compared == [["a", "b"]]
    assert not (tmp_path / "out" / "a" / "broken" / "movie_status.json").exists()
    assert "boom" in (tmp_path / "out" / "run.log").read_text(encoding="utf-8")


def test_skip_existing_skips_finished_movies(tmp_path, recorder):
    _movie(tmp_path / "g" / "m.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")], skip_existing=True)
    runner.run(cfg)
    recorder.movies.clear()
    result = runner.run(cfg)
    assert recorder.movies == []
    assert result.rows[0]["status"] == "SKIP"
    assert result.rows[0]["n_spots"] > 0
    assert recorder.groups[-1] == ("g", ["m"])


def test_skip_existing_reuses_spots_when_analysis_did_not_finish(tmp_path, recorder, monkeypatch):
    _movie(tmp_path / "g" / "m.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")], skip_existing=True)
    runner.run(cfg)
    (tmp_path / "out" / "g" / "m" / runner.STATUS_FILE).unlink()

    def no_detection(*args, **kwargs):
        raise AssertionError("detection should be reused from spots.csv")

    monkeypatch.setattr(runner, "detect_stack", no_detection)
    result = runner.run(cfg)
    assert result.rows[0]["status"] == "OK"
    assert recorder.movies[-1] == "g/m"


def test_non_3d_movie_fails_cleanly(tmp_path, recorder):
    path = tmp_path / "g" / "flat.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), np.zeros((16, 16), np.uint16))
    result = runner.run(_cfg(tmp_path, [str(path.parent)]))
    assert result.rows[0]["status"] == "FAIL"
    assert "(T, Y, X)" in result.rows[0]["error"]


def test_save_preprocessed(tmp_path, recorder):
    _movie(tmp_path / "g" / "m.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")],
               preprocessing={"steps": [{"gaussian_blur": {"sigma": 1}}]},
               outputs={"formats": ["png"], "dpi": 40, "save_preprocessed": True})
    runner.run(cfg)
    saved = tifffile.imread(str(tmp_path / "out" / "g" / "m" / "preprocessed.tif"))
    assert saved.shape == (6, 48, 48)


def test_status_file_is_json_row(tmp_path, recorder):
    _movie(tmp_path / "g" / "m.tif")
    runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))
    row = json.loads((tmp_path / "out" / "g" / "m" / runner.STATUS_FILE).read_text())
    assert row["status"] == "OK" and row["movie"] == "m"


def test_hook_failure_sets_exit_code(tmp_path, recorder, monkeypatch):
    _movie(tmp_path / "g" / "m.tif")

    def bad_summary(*args, **kwargs):
        raise ValueError("summary broke")

    monkeypatch.setattr(recorder, "summarize_group", bad_summary)
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))
    assert result.failed_hooks == ["recorder: group summary for 'g'"]
    assert result.exit_code == 1


def test_skip_existing_reprocesses_when_status_file_is_corrupt(tmp_path, recorder):
    _movie(tmp_path / "g" / "m.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")], skip_existing=True)
    runner.run(cfg)
    status_path = tmp_path / "out" / "g" / "m" / runner.STATUS_FILE
    status_path.write_text("{not valid json", encoding="utf-8")
    recorder.movies.clear()

    result = runner.run(cfg)
    assert recorder.movies == ["g/m"]
    assert result.rows[0]["status"] == "OK"
    assert result.exit_code == 0


def test_status_write_failure_is_isolated(tmp_path, recorder, monkeypatch):
    _movie(tmp_path / "g" / "ok.tif")
    _movie(tmp_path / "g" / "flaky.tif")
    real_replace = runner.os.replace

    def flaky_replace(src, dst):
        dst = Path(dst)
        if dst.name == runner.STATUS_FILE and "flaky" in dst.parts:
            raise OSError("status file is locked")
        return real_replace(src, dst)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))

    rows = {r["movie"]: r for r in result.rows}
    assert rows["flaky"]["status"] == "FAIL"
    assert rows["flaky"]["error"].startswith("OSError")
    assert rows["ok"]["status"] == "OK"
    assert (tmp_path / "out" / "run_summary.csv").exists()
    assert result.exit_code == 1


def test_failed_rerun_invalidates_status_file(tmp_path, recorder, monkeypatch):
    _movie(tmp_path / "g" / "m.tif")
    status_path = tmp_path / "out" / "g" / "m" / runner.STATUS_FILE
    assert runner.run(_cfg(tmp_path, [str(tmp_path / "g")])).exit_code == 0
    assert status_path.exists()

    real_detect = runner.detect_stack

    def broken_detect(*args, **kwargs):
        raise RuntimeError("new parameters broke detection")

    monkeypatch.setattr(runner, "detect_stack", broken_detect)
    failed = runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))
    assert failed.rows[0]["status"] == "FAIL"
    assert not status_path.exists()

    monkeypatch.setattr(runner, "detect_stack", real_detect)
    recorder.movies.clear()
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "g")], skip_existing=True))
    assert result.rows[0]["status"] == "OK"
    assert recorder.movies == ["g/m"]


def test_status_file_that_cannot_be_removed_fails_movie(tmp_path, recorder, monkeypatch):
    _movie(tmp_path / "g" / "m.tif")
    runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))
    recorder.movies.clear()
    real_unlink = Path.unlink

    def locked_unlink(self, *args, **kwargs):
        if self.name == runner.STATUS_FILE:
            raise PermissionError("status file is locked")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    result = runner.run(_cfg(tmp_path, [str(tmp_path / "g")]))
    assert result.rows[0]["status"] == "FAIL"
    assert result.rows[0]["error"] == "PermissionError: status file is locked"
    assert recorder.movies == []


def test_hooks_receive_cfg_and_dropped_groups_are_logged(tmp_path, recorder, caplog):
    _movie(tmp_path / "a" / "m1.tif")
    _movie(tmp_path / "b" / "m2.tif")
    _movie(tmp_path / "c" / "broken.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / d) for d in "abc"])
    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        runner.run(cfg)
    assert recorder.groups == [("a", ["m1"]), ("b", ["m2"])]
    assert recorder.compared == [["a", "b"]]
    assert len(recorder.hook_cfgs) == 3
    assert all(seen is cfg for seen in recorder.hook_cfgs)
    dropped = [r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING and "'c'" in r.getMessage()]
    assert len(dropped) == 1 and "no movie succeeded" in dropped[0]


def test_mask_and_fit_columns_are_written_by_default(tmp_path, recorder):
    _movie(tmp_path / "g" / "m.tif")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")])
    cfg.detection.fit.enabled = True
    cfg.outputs.save_masks = True
    assert runner.run(cfg).exit_code == 0
    movie = tmp_path / "out" / "g" / "m"
    spots = pd.read_csv(movie / "spots.csv")
    assert {"amp", "x_fit", "y_fit", "sigma_x", "sigma_y", "bg",
            "area", "mean_intensity", "max_intensity", "spot_radius"} <= set(spots.columns)
    mask = tifffile.imread(str(movie / "mask.tif"))
    assert mask.shape[0] == 6 and mask.any()


def test_resolved_config_holds_merged_settings_and_effective_device(tmp_path, recorder,
                                                                     monkeypatch):
    _movie(tmp_path / "g" / "m.tif")
    monkeypatch.setattr(runner, "_get_device", lambda use_gpu: "cpu")
    cfg = _cfg(tmp_path, [str(tmp_path / "g")], device="cuda",
               analyses={"recorder": {}, "unknown_one": {"x": 1}})
    with pytest.warns(UserWarning, match="unknown analysis 'analyses.unknown_one'"):
        runner.run(cfg)
    resolved = yaml.safe_load((tmp_path / "out" / "resolved_config.yaml").read_text(
        encoding="utf-8"))
    assert resolved["device"] == "cpu"
    assert resolved["analyses"]["recorder"]["size"] == 3
    assert resolved["analyses"]["unknown_one"] == {"x": 1}
    assert cfg.device == "cuda"  # the caller's config object is not mutated
