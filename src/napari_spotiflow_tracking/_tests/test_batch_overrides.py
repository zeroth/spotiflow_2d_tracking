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
