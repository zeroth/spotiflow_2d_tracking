"""Config loading: presets, precedence, --set overrides, unknown keys."""
import warnings

import pytest
import yaml

from napari_spotiflow_tracking.batch.config import (
    ConfigError,
    ConfigWarning,
    PipelineConfig,
    config_to_dict,
    deep_merge,
    load_config,
    load_preset,
    parse_override,
    write_resolved_config,
)
from napari_spotiflow_tracking.batch.validation import validate


def _write(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_without_file_use_tracking_preset():
    cfg = load_config(data={})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.modality == "tracking"
    assert cfg.tracking.search_range == 5.0
    assert cfg.tracking.memory == 3
    names = [next(iter(s)) for s in cfg.preprocessing.steps]
    assert names == ["walking_average", "subtract", "gaussian_blur"]


def test_each_preset_loads():
    for modality in ("tracking", "papa", "bleaching"):
        assert isinstance(load_preset(modality), dict)


def test_papa_preset_disables_tracking():
    cfg = load_config(data={"modality": "papa"})
    assert cfg.tracking.enabled is False
    assert cfg.preprocessing.steps == []


def test_bleaching_preset_tracking_settings():
    cfg = load_config(data={"modality": "bleaching"})
    assert (cfg.tracking.search_range, cfg.tracking.memory) == (3.0, 10)


def test_unknown_modality_is_config_error():
    with pytest.raises(ConfigError, match="unknown modality"):
        load_config(data={"modality": "sted"})


def test_user_file_overrides_preset(tmp_path):
    path = _write(tmp_path, {"modality": "bleaching", "tracking": {"memory": 4}})
    cfg = load_config(path)
    assert cfg.tracking.memory == 4
    assert cfg.tracking.search_range == 3.0  # untouched preset value


def test_set_overrides_beat_file(tmp_path):
    path = _write(tmp_path, {"tracking": {"memory": 4}})
    cfg = load_config(path, overrides=["tracking.memory=7", "outputs.formats=[pdf]"])
    assert cfg.tracking.memory == 7
    assert cfg.outputs.formats == ["pdf"]


def test_set_can_change_modality():
    cfg = load_config(overrides=["modality=papa"])
    assert cfg.modality == "papa"
    assert cfg.tracking.enabled is False


def test_parse_override_types():
    assert parse_override("a.b=3") == (["a", "b"], 3)
    assert parse_override("a=true") == (["a"], True)
    assert parse_override("a=0.4") == (["a"], 0.4)
    assert parse_override("a=D:/data") == (["a"], "D:/data")
    assert parse_override("a=") == (["a"], None)
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")
    with pytest.raises(ConfigError):
        parse_override("=3")


def test_lists_are_replaced_not_merged():
    merged = deep_merge({"a": [1, 2], "b": {"c": 1, "d": 2}}, {"a": [3], "b": {"c": 5}})
    assert merged == {"a": [3], "b": {"c": 5, "d": 2}}


def test_unknown_keys_warn_and_are_ignored():
    with pytest.warns(ConfigWarning, match="tracking.serch_range"):
        cfg = load_config(data={"tracking": {"serch_range": 9}})
    assert cfg.tracking.search_range == 5.0


def test_inputs_accept_plain_strings_and_mappings():
    cfg = load_config(data={"inputs": ["D:/a.tif", {"path": "D:/b", "label": "B"}]})
    assert cfg.inputs[0].path == "D:/a.tif"
    assert cfg.inputs[0].label is None
    assert cfg.inputs[1].label == "B"
    assert cfg.inputs[1].pattern == "*.tif"


def test_section_must_be_mapping():
    with pytest.raises(ConfigError, match="tracking must be a mapping"):
        load_config(data={"tracking": 3})


def test_calibration_is_physical():
    cfg = load_config(data={"calibration": {"pixel_size_nm": 100, "frame_interval_s": 0.1}})
    assert cfg.calibration.is_physical
    assert not load_config(data={}).calibration.is_physical


def test_resolved_config_round_trips(tmp_path):
    cfg = load_config(data={"modality": "bleaching", "output_dir": "D:/out"})
    out = tmp_path / "resolved.yaml"
    write_resolved_config(cfg, out)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        again = load_config(out)
    assert config_to_dict(again) == config_to_dict(cfg)


@pytest.mark.parametrize("data, message", [
    ({"tracking": {"memory": "five"}}, "tracking.memory must be int, got 'five'"),
    ({"tracking": {"memory": True}}, "tracking.memory must be int, got True"),
    ({"skip_existing": "no"}, "skip_existing must be bool, got 'no'"),
    ({"outputs": {"formats": "png"}}, "outputs.formats must be list of str, got 'png'"),
    ({"outputs": {"formats": ["png", 3]}}, "outputs.formats must be list of str"),
    ({"outputs": {"dpi": "high"}}, "outputs.dpi must be int, got 'high'"),
    ({"device": 0}, "device must be str, got 0"),
    ({"calibration": {"pixel_size_nm": "100nm"}},
     "calibration.pixel_size_nm must be float or null, got '100nm'"),
])
def test_scalar_types_are_checked(data, message):
    with pytest.raises(ConfigError) as info:
        load_config(data=data)
    assert message in str(info.value)


def test_fitting_and_masks_are_on_by_default():
    cfg = load_config(data={})
    assert cfg.detection.fit.enabled is True
    assert cfg.outputs.save_masks is True


@pytest.mark.parametrize("modality", ["tracking", "bleaching"])  # papa opts out deliberately
def test_every_preset_fits(modality):
    cfg = load_config(data={"modality": modality})
    assert cfg.detection.fit.enabled is True


def test_papa_preset_opts_out_of_fitting_and_masks():
    """PAPA only counts spots; it never reads the fit columns or the mask, so
    the preset turns off Task 10's fit-and-mask defaults (both keys are
    required together: validation needs masks off when fitting is off)."""
    cfg = load_config(data={"modality": "papa", "inputs": ["x.tif"], "output_dir": "out"})
    assert cfg.detection.fit.enabled is False
    assert cfg.outputs.save_masks is False
    validate(cfg)


def test_tracking_has_no_min_track_length():
    """The minimum length belongs to the analysis, not to tracking."""
    cfg = load_config(data={})
    assert not hasattr(cfg.tracking, "min_track_length")


def test_min_track_length_in_a_config_warns_and_is_ignored():
    with pytest.warns(ConfigWarning, match="tracking.min_track_length"):
        load_config(data={"tracking": {"min_track_length": 10}})


def test_optional_accepts_null_and_int_becomes_float():
    cfg = load_config(data={"detection": {"spotiflow": {"prob_thresh": None}},
                            "tracking": {"search_range": 3}})
    assert cfg.detection.spotiflow.prob_thresh is None
    assert cfg.tracking.search_range == 3.0 and isinstance(cfg.tracking.search_range, float)
    cfg = load_config(data={"calibration": {"pixel_size_nm": 100}})
    assert isinstance(cfg.calibration.pixel_size_nm, float)
