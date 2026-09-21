"""Per-group (``settings``) and per-file (``per_file``) config overrides.

Resolution order (later wins, deep-merged): dataclass defaults < modality
preset < global YAML < ``--set`` < group ``settings`` < matching ``per_file``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.analyses import enabled_analyses
from napari_spotiflow_tracking.batch.config import ConfigError, load_config
from napari_spotiflow_tracking.batch.inputs import MovieJob, expand_inputs
from napari_spotiflow_tracking.batch.overrides import (
    ALLOWED_SECTIONS,
    config_for,
    matches_for,
    validate_all,
    validate_override_sections,
)
from napari_spotiflow_tracking.batch.validation import validate


def _job(group, movie_id, filename=None):
    filename = filename or f"{movie_id}.tif"
    return MovieJob(group=group, movie_id=movie_id, path=Path(f"/data/{group}/{filename}"))


def _spot_movie(path, amps, n_frames=4, size=32):
    """Two Gaussian spots of different amplitude, so a stricter LoG threshold
    drops the dim one but keeps the bright one."""
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    for (cy, cx), amp in zip(((8, 8), (22, 22)), amps):
        for t in range(n_frames):
            stack[t] += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


def _no_analysis_cfg(inputs, output_dir, **extra):
    data = {
        "inputs": inputs,
        "output_dir": str(output_dir),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log", "fit": {"enabled": False}},
        "tracking": {"enabled": False},
        "outputs": {"formats": ["png"], "dpi": 40, "save_masks": False},
        "analyses": {"msd": {"enabled": False}},
    }
    data.update(extra)
    return load_config(data=data)


# ── 1. Layer precedence ──────────────────────────────────────────────────

def test_per_file_beats_group_beats_global_beats_set():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "settings": {"detection": {"spotiflow": {"prob_thresh": 0.45}}},
            "per_file": {"cell03.tif": {"detection": {"spotiflow": {"prob_thresh": 0.6}}}},
        }],
        "output_dir": "out",
        "detection": {"spotiflow": {"prob_thresh": 0.5}},
    }, overrides=["detection.spotiflow.prob_thresh=0.4"])
    resolved = config_for(_job("A", "cell03", "cell03.tif"), cfg)
    assert resolved.detection.spotiflow.prob_thresh == 0.6


def test_group_settings_beat_global_for_every_movie_without_a_file_match():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "settings": {"detection": {"spotiflow": {"prob_thresh": 0.45}}},
        }],
        "output_dir": "out",
        "detection": {"spotiflow": {"prob_thresh": 0.5}},
    })
    resolved = config_for(_job("A", "cell01"), cfg)
    assert resolved.detection.spotiflow.prob_thresh == 0.45


def test_global_only_value_survives_group_and_file_layers():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "settings": {"detection": {"spotiflow": {"prob_thresh": 0.45}}},
        }],
        "output_dir": "out",
        "tracking": {"memory": 7},
    })
    resolved = config_for(_job("A", "cell01"), cfg)
    assert resolved.tracking.memory == 7


def test_set_beats_global_yaml_but_loses_to_per_file():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {"cell03.tif": {"detection": {"spotiflow": {"prob_thresh": 0.6}}}},
        }],
        "output_dir": "out",
        "detection": {"spotiflow": {"prob_thresh": 0.5}},
    }, overrides=["detection.spotiflow.prob_thresh=0.4"])
    unmatched = config_for(_job("A", "cell01", "cell01.tif"), cfg)
    assert unmatched.detection.spotiflow.prob_thresh == 0.4  # --set beats global YAML
    matched = config_for(_job("A", "cell03", "cell03.tif"), cfg)
    assert matched.detection.spotiflow.prob_thresh == 0.6  # per_file still wins over --set


def test_config_for_returns_the_same_object_with_no_overrides():
    cfg = load_config(data={"inputs": [{"path": "D:/x", "label": "A"}], "output_dir": "out"})
    assert config_for(_job("A", "m1"), cfg) is cfg


def test_config_for_unknown_group_returns_cfg_unchanged():
    cfg = load_config(data={"inputs": [{"path": "D:/x", "label": "A"}], "output_dir": "out"})
    assert config_for(_job("Other", "m1"), cfg) is cfg


def test_two_input_entries_each_keep_their_own_settings():
    """The per-entry property must not get pinned to whichever entry happens
    to be first/last: each input's settings apply only to its own movies."""
    cfg = load_config(data={
        "inputs": [
            {"path": "D:/a", "label": "A",
             "settings": {"detection": {"spotiflow": {"prob_thresh": 0.3}}}},
            {"path": "D:/b", "label": "B",
             "settings": {"detection": {"spotiflow": {"prob_thresh": 0.7}}}},
        ],
        "output_dir": "out",
    })
    a = config_for(_job("A", "m1"), cfg)
    b = config_for(_job("B", "m2"), cfg)
    assert a.detection.spotiflow.prob_thresh == 0.3
    assert b.detection.spotiflow.prob_thresh == 0.7


def test_empty_settings_and_per_file_are_a_no_op():
    cfg = load_config(data={
        "inputs": [{"path": "D:/x", "label": "A", "settings": {}, "per_file": {}}],
        "output_dir": "out",
    })
    assert config_for(_job("A", "m1"), cfg) is cfg


# ── 2. Deep merging ───────────────────────────────────────────────────────

def test_per_file_merge_is_deep_and_leaves_siblings_alone():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {"cell03.tif": {"detection": {"spotiflow": {"prob_thresh": 0.6}}}},
        }],
        "output_dir": "out",
        "detection": {"method": "log", "spotiflow": {"min_distance": 5}},
    })
    resolved = config_for(_job("A", "cell03", "cell03.tif"), cfg)
    assert resolved.detection.spotiflow.prob_thresh == 0.6
    assert resolved.detection.spotiflow.min_distance == 5
    assert resolved.detection.method == "log"


# ── 3. Matching rules ─────────────────────────────────────────────────────

def test_matches_by_filename_movie_id_and_glob():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {
                "cell03.tif": {"tracking": {"memory": 1}},
                "cell04": {"tracking": {"memory": 2}},
                "low_snr_*": {"tracking": {"memory": 3}},
            },
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    jobs = [_job("A", "cell03", "cell03.tif"),
            _job("A", "cell04", "cell04.tif"),
            _job("A", "low_snr_5", "low_snr_5.tif")]
    result = matches_for(spec, jobs)
    assert result["cell03"]["tracking"]["memory"] == 1
    assert result["cell04"]["tracking"]["memory"] == 2
    assert result["low_snr_5"]["tracking"]["memory"] == 3


def test_exact_match_beats_a_glob():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {
                "cell03.tif": {"tracking": {"memory": 1}},
                "cell*": {"tracking": {"memory": 99}},
            },
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    result = matches_for(spec, [_job("A", "cell03", "cell03.tif")])
    assert result["cell03"]["tracking"]["memory"] == 1


def test_two_globs_matching_one_movie_is_a_config_error():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {
                "cell0*": {"tracking": {"memory": 1}},
                "*03": {"tracking": {"memory": 2}},
            },
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    with pytest.raises(ConfigError) as info:
        matches_for(spec, [_job("A", "cell03", "cell03.tif")])
    text = str(info.value)
    assert "cell03" in text and "cell0*" in text and "*03" in text


def test_unmatched_per_file_key_is_a_config_error():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {"typo_cell.tif": {"tracking": {"memory": 1}}},
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    with pytest.raises(ConfigError, match="typo_cell.tif"):
        matches_for(spec, [_job("A", "cell03", "cell03.tif")])


def test_matching_is_case_sensitive_on_every_platform():
    """fnmatch.fnmatch normalizes case on Windows (via os.path.normcase) but
    not on Linux, so plain fnmatch would classify 'Cell03.tif' differently
    per OS -- an exact match on Windows, a glob (or no match) on Linux.
    fnmatchcase must make this identical everywhere: a different-case key
    matches nothing, not even as a glob."""
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {"Cell03.tif": {"tracking": {"memory": 1}}},
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    exact = matches_for(spec, [_job("A", "Cell03", "Cell03.tif")])
    assert exact["Cell03"]["tracking"]["memory"] == 1
    with pytest.raises(ConfigError, match="Cell03.tif"):
        matches_for(spec, [_job("A", "cell03", "cell03.tif")])


def test_ambiguity_error_does_not_hide_an_unmatched_key_in_the_same_entry():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {
                "cell0*": {"tracking": {"memory": 1}},
                "*03": {"tracking": {"memory": 2}},
                "typo.tif": {"tracking": {"memory": 3}},
            },
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    with pytest.raises(ConfigError) as info:
        matches_for(spec, [_job("A", "cell03", "cell03.tif")])
    text = str(info.value)
    assert "cell03" in text and "cell0*" in text and "*03" in text  # the ambiguity
    assert "typo.tif" in text  # the unmatched key -- not hidden by the ambiguity


def test_recursive_input_matches_by_combined_movie_id():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A", "recursive": True,
            "per_file": {"day1__cell03": {"tracking": {"memory": 9}}},
        }],
        "output_dir": "out",
    })
    spec = cfg.inputs[0]
    result = matches_for(spec, [_job("A", "day1__cell03", "cell03.tif")])
    assert result["day1__cell03"]["tracking"]["memory"] == 9


# ── 4. Allowed sections ───────────────────────────────────────────────────

def test_validate_override_sections_allows_the_six_sections():
    section = {name: {} for name in ALLOWED_SECTIONS}
    assert validate_override_sections(section, "inputs[0].settings") == []


@pytest.mark.parametrize("bad_key", ["outputs", "device", "skip_existing", "output_dir", "inputs"])
def test_validate_override_sections_rejects_run_level_keys(bad_key):
    problems = validate_override_sections({bad_key: {}}, "inputs[0].settings")
    assert len(problems) == 1
    assert f"inputs[0].settings.{bad_key}" in problems[0]


@pytest.mark.parametrize("bad_key", ["outputs", "device", "skip_existing", "output_dir", "inputs"])
def test_validate_rejects_disallowed_section_in_settings(bad_key):
    cfg = load_config(data={
        "inputs": [{"path": "D:/x", "label": "A", "settings": {bad_key: {}}}],
        "output_dir": "out",
    })
    with pytest.raises(ConfigError, match=rf"inputs\[0\]\.settings\.{bad_key}"):
        validate(cfg)


@pytest.mark.parametrize("bad_key", ["outputs", "device", "skip_existing", "output_dir", "inputs"])
def test_validate_rejects_disallowed_section_in_per_file(bad_key):
    cfg = load_config(data={
        "inputs": [{"path": "D:/x", "label": "A", "per_file": {"m.tif": {bad_key: {}}}}],
        "output_dir": "out",
    })
    with pytest.raises(ConfigError, match=rf"inputs\[0\]\.per_file\.m\.tif\.{bad_key}"):
        validate(cfg)


def test_validate_passes_with_only_allowed_sections():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "settings": {
                "preprocessing": {"steps": []},
                "detection": {"spotiflow": {"prob_thresh": 0.4}},
                "tracking": {"memory": 2},
                "calibration": {"pixel_size_nm": 100},
                "measure_on": "raw",
                "analyses": {"msd": {"min_length": 10}},
            },
            "per_file": {"m.tif": {"tracking": {"memory": 3}}},
        }],
        "output_dir": "out",
    })
    validate(cfg)


# ── 5. Analyses settings override per file ────────────────────────────────

def test_analyses_settings_override_per_file():
    cfg = load_config(data={
        "inputs": [{
            "path": "D:/x", "label": "A",
            "per_file": {"cell03.tif": {"analyses": {"msd": {"min_length": 20}}}},
        }],
        "output_dir": "out",
        "analyses": {"msd": {}},
    })
    [(_, other_settings)] = enabled_analyses(config_for(_job("A", "cell01", "cell01.tif"), cfg))
    [(_, match_settings)] = enabled_analyses(config_for(_job("A", "cell03", "cell03.tif"), cfg))
    assert other_settings["min_length"] == 5   # msd's own default, untouched
    assert match_settings["min_length"] == 20


# ── 6. End to end through runner.run ──────────────────────────────────────

def test_end_to_end_per_file_override_changes_spot_count_and_resolved_config(tmp_path):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g"), "label": "g",
          "per_file": {"b.tif": {"detection": {"log": {"threshold": 0.3}}}}}],
        tmp_path / "out",
    )
    result = runner.run(cfg)
    assert result.exit_code == 0

    spots_a = pd.read_csv(tmp_path / "out" / "g" / "a" / "spots.csv")
    spots_b = pd.read_csv(tmp_path / "out" / "g" / "b" / "spots.csv")
    assert len(spots_a) > len(spots_b) > 0  # b's stricter threshold drops the dim spot

    resolved_a = yaml.safe_load(
        (tmp_path / "out" / "g" / "a" / "resolved_config.yaml").read_text(encoding="utf-8"))
    resolved_b = yaml.safe_load(
        (tmp_path / "out" / "g" / "b" / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved_a["detection"]["log"]["threshold"] == 0.09  # dataclass default, untouched
    assert resolved_b["detection"]["log"]["threshold"] == 0.3


# ── 7. Heterogeneity warning ───────────────────────────────────────────────

def test_heterogeneity_warning_fires_when_group_detection_settings_differ(tmp_path, caplog):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g"), "label": "g",
          "per_file": {"b.tif": {"detection": {"log": {"threshold": 0.3}}}}}],
        tmp_path / "out",
    )
    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        runner.run(cfg)
    hits = [r.getMessage() for r in caplog.records if "do not all share" in r.getMessage()]
    assert len(hits) == 1 and "'g'" in hits[0]


def test_heterogeneity_warning_does_not_fire_when_settings_match(tmp_path, caplog):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg([str(tmp_path / "g")], tmp_path / "out")
    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        runner.run(cfg)
    hits = [r.getMessage() for r in caplog.records if "do not all share" in r.getMessage()]
    assert hits == []


def test_heterogeneity_warning_fires_for_a_calibration_difference(tmp_path, caplog):
    """calibration is overridable and changes what unit a group's pooled
    figures/tables are labelled in -- a difference here must warn too, not
    just preprocessing/detection."""
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g"), "label": "g",
          "per_file": {"b.tif": {"calibration": {"pixel_size_nm": 100.0}}}}],
        tmp_path / "out",
    )
    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        runner.run(cfg)
    hits = [r.getMessage() for r in caplog.records if "do not all share" in r.getMessage()]
    assert len(hits) == 1
    assert "calibration" in hits[0]


def test_heterogeneity_warning_is_scoped_to_its_own_group(tmp_path, caplog):
    """A bug that pooled every movie_cfg across the whole run (instead of
    grouping by job.group first) would also warn for the homogeneous group
    just because a *different* group's movies differ."""
    _spot_movie(tmp_path / "g1" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g1" / "b.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g2" / "c.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g2" / "d.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g1"), "label": "g1",
          "per_file": {"b.tif": {"detection": {"log": {"threshold": 0.3}}}}},
         {"path": str(tmp_path / "g2"), "label": "g2"}],
        tmp_path / "out",
    )
    with caplog.at_level(logging.WARNING, logger="napari_spotiflow_tracking.batch"):
        runner.run(cfg)
    hits = [r.getMessage() for r in caplog.records if "do not all share" in r.getMessage()]
    assert len(hits) == 1
    assert "'g1'" in hits[0] and "'g2'" not in hits[0]


# ── 8. Regression: no settings/per_file behaves exactly as before ─────────

def test_no_overrides_behaves_like_before(tmp_path):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    cfg = _no_analysis_cfg([str(tmp_path / "g")], tmp_path / "out")
    result = runner.run(cfg)
    assert result.exit_code == 0
    movie = tmp_path / "out" / "g" / "a"
    assert (movie / "resolved_config.yaml").exists()
    assert (movie / "spots.csv").exists()
    resolved = yaml.safe_load((movie / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["detection"]["log"]["threshold"] == 0.09
    run_resolved = yaml.safe_load(
        (tmp_path / "out" / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert run_resolved["detection"]["log"]["threshold"] == 0.09


# ── Detector cache keyed by the (now overridable) detection settings ──────

class _FakeSpotiflowModel:
    """Stands in for a real Spotiflow model: predict() finds nothing, so the
    detector-cache tests below don't need a real model file."""

    def __init__(self, name):
        self.name = name

    def predict(self, image_2d, prob_thresh=None, min_distance=2, verbose=False):
        return np.empty((0, 2)), None


def test_detector_cache_is_keyed_by_detection_method(tmp_path, monkeypatch):
    """Reproduces the CRITICAL bug: a single ``detector: dict`` cache slot
    shared by every movie meant movie b inherited movie a's LoG sentinel
    (``None``) as its "model" once detection became overridable, and calling
    ``None.predict(...)`` blew up with an AttributeError."""
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    calls = []

    def fake_load_detector(detection, device):
        if detection.method != "spotiflow":
            calls.append(("log", None))
            return None
        calls.append(("spotiflow", detection.spotiflow.model))
        return _FakeSpotiflowModel(detection.spotiflow.model)

    monkeypatch.setattr(runner, "load_detector", fake_load_detector)
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g"), "label": "g",
          "per_file": {"b.tif": {"detection": {"method": "spotiflow"}}}}],
        tmp_path / "out",
    )
    result = runner.run(cfg)

    rows = {r["movie"]: r for r in result.rows}
    assert rows["a"]["status"] == "OK" and rows["b"]["status"] == "OK"
    assert result.exit_code == 0
    assert calls.count(("log", None)) == 1
    assert calls.count(("spotiflow", "general")) == 1


def test_detector_cache_reloads_for_each_distinct_spotiflow_model(tmp_path, monkeypatch):
    """The silent variant of the bug: with every movie on ``method:
    spotiflow``, the first-loaded model was cached and reused for a
    different per-file ``spotiflow.model`` -- wrong detections, no warning,
    and a resolved_config.yaml that lied about which model ran."""
    _spot_movie(tmp_path / "g" / "a.tif", amps=(3000, 1500))
    _spot_movie(tmp_path / "g" / "b.tif", amps=(3000, 1500))
    calls = []

    def fake_load_detector(detection, device):
        calls.append(detection.spotiflow.model)
        return _FakeSpotiflowModel(detection.spotiflow.model)

    monkeypatch.setattr(runner, "load_detector", fake_load_detector)
    cfg = _no_analysis_cfg(
        [{"path": str(tmp_path / "g"), "label": "g",
          "per_file": {"b.tif": {"detection": {"spotiflow": {"model": "hybiss"}}}}}],
        tmp_path / "out",
        detection={"method": "spotiflow", "fit": {"enabled": False}},
    )
    result = runner.run(cfg)

    assert result.exit_code == 0
    assert sorted(calls) == ["general", "hybiss"]
    resolved_b = yaml.safe_load(
        (tmp_path / "out" / "g" / "b" / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved_b["detection"]["spotiflow"]["model"] == "hybiss"


# ── validate_all / fail-before-first-movie ─────────────────────────────────

def test_validate_all_raises_for_unmatched_per_file_key(tmp_path):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(1000, 1000))
    cfg = load_config(data={
        "inputs": [{"path": str(tmp_path / "g"), "label": "g",
                   "per_file": {"missing.tif": {"tracking": {"memory": 2}}}}],
        "output_dir": str(tmp_path / "out"),
    })
    jobs = expand_inputs(cfg)
    with pytest.raises(ConfigError, match="missing.tif"):
        validate_all(cfg, jobs)


def test_runner_fails_before_the_first_movie_loads_for_a_bad_per_file_key(tmp_path):
    _spot_movie(tmp_path / "g" / "a.tif", amps=(1000, 1000))
    cfg = load_config(data={
        "inputs": [{"path": str(tmp_path / "g"), "label": "g",
                   "per_file": {"missing.tif": {"tracking": {"memory": 2}}}}],
        "output_dir": str(tmp_path / "out"),
    })
    with pytest.raises(ConfigError, match="missing.tif"):
        runner.run(cfg)
    assert not (tmp_path / "out").exists()
