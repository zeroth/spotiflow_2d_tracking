"""Registry behaviour and whole-config validation."""
import pytest

from napari_spotiflow_tracking.batch import analyses
from napari_spotiflow_tracking.batch.analyses import (
    Analysis,
    enabled_analyses,
    measure_on_for,
    register,
)
from napari_spotiflow_tracking.batch.config import ConfigError, ConfigWarning, load_config
from napari_spotiflow_tracking.batch.outputs import OutputSpec
from napari_spotiflow_tracking.batch.validation import validate


class Demo(Analysis):
    name = "demo"
    requires_tracks = True
    catalogue = (OutputSpec("plot", "movie", "figure", "A plot"),)

    def default_settings(self):
        return {"size": 3}

    def validate(self, settings):
        return [] if settings["size"] > 0 else ["size must be positive"]


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setattr(analyses.base, "REGISTRY", {})
    return register(Demo())


def _cfg(**data):
    base = {"inputs": ["D:/x.tif"], "output_dir": "D:/out"}
    base.update(data)
    return load_config(data=base)


def test_valid_default_config_passes():
    validate(_cfg())


def test_enabled_analyses_merge_defaults_and_skip_disabled(demo):
    cfg = _cfg(analyses={"demo": {"size": 5}})
    [(analysis, settings)] = enabled_analyses(cfg)
    assert analysis is demo and settings == {"size": 5}
    assert enabled_analyses(_cfg(analyses={"demo": None}))[0][1] == {"size": 3}
    assert enabled_analyses(_cfg(analyses={"demo": {"enabled": False}})) == []


def test_register_rejects_duplicates(demo):
    with pytest.raises(ValueError, match="already registered"):
        register(Demo())


def test_collects_all_errors(demo):
    cfg = _cfg(
        inputs=[],
        output_dir="",
        device="gpu",
        measure_on="denoised",
        detection={"method": "dog", "fit": {"enabled": False}},
        outputs={"formats": ["png", "tiff"], "save_masks": True},
        preprocessing={"steps": [{"sharpen": {}}]},
        tracking={"enabled": False},
        analyses={"demo": {"size": 0}},
    )
    with pytest.raises(ConfigError) as info:
        validate(cfg)
    text = str(info.value)
    for fragment in [
        "inputs: at least one input is required",
        "output_dir is required",
        "device must be cuda or cpu",
        "measure_on must be raw or preprocessed",
        "detection.method must be spotiflow or log",
        "outputs.formats: unsupported 'tiff'",
        "outputs.save_masks needs detection.fit.enabled",
        "unknown step 'sharpen'",
        "analyses.demo needs tracking",
        "analyses.demo: size must be positive",
    ]:
        assert fragment in text


def test_unknown_analysis_warns():
    with pytest.warns(ConfigWarning, match="unknown analysis 'analyses.msdd'"):
        validate(_cfg(analyses={"msdd": {}}))


def test_unknown_output_name_warns(demo):
    with pytest.warns(ConfigWarning, match="unknown output 'plots'"):
        validate(_cfg(analyses={"demo": {"figures": ["plots"]}}))


def test_fit_does_not_require_torchimize(monkeypatch):
    """The scipy fallback means a missing torchimize is not a config error."""
    monkeypatch.setattr("napari_spotiflow_tracking.batch.validation.torchimize_available",
                        lambda: False)
    validate(_cfg(detection={"fit": {"enabled": True}}))


def test_save_masks_still_needs_fitting():
    with pytest.raises(ConfigError, match="outputs.save_masks needs detection.fit.enabled"):
        validate(_cfg(detection={"fit": {"enabled": False}},
                      outputs={"save_masks": True}))


def test_analysis_section_must_be_mapping(demo):
    with pytest.raises(ConfigError, match="analyses.demo must be a mapping"):
        validate(_cfg(analyses={"demo": [1, 2]}))


def test_measure_on_for_resolution_order(demo):
    cfg = _cfg(measure_on="preprocessed")
    assert measure_on_for(demo, {}, cfg) == "preprocessed"
    assert measure_on_for(demo, {"measure_on": "raw"}, cfg) == "raw"
    assert measure_on_for(demo, {"measure_on": None}, cfg) == "preprocessed"

    class Blind(Analysis):
        name = "blind"
        default_measure_on = "none"

    assert measure_on_for(Blind(), {"measure_on": "raw"}, cfg) is None


def test_analysis_measure_on_is_validated(demo):
    with pytest.raises(ConfigError, match="analyses.demo.measure_on must be raw or preprocessed"):
        validate(_cfg(analyses={"demo": {"measure_on": "denoised"}}))
    validate(_cfg(tracking={"enabled": True},
                  analyses={"demo": {"measure_on": "preprocessed"}}))
    # a disabled analysis is not checked
    validate(_cfg(analyses={"demo": {"enabled": False, "measure_on": "denoised"}}))
