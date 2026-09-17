"""Registry behaviour and whole-config validation."""
import pytest

from napari_spotiflow_tracking.batch import analyses
from napari_spotiflow_tracking.batch.analyses import Analysis, enabled_analyses, register
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
        detection={"method": "dog"},
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


def test_fit_needs_torchimize(monkeypatch):
    monkeypatch.setattr("napari_spotiflow_tracking.batch.validation.torchimize_available",
                        lambda: False)
    with pytest.raises(ConfigError, match="torchimize"):
        validate(_cfg(detection={"fit": {"enabled": True}}))


def test_analysis_section_must_be_mapping(demo):
    with pytest.raises(ConfigError, match="analyses.demo must be a mapping"):
        validate(_cfg(analyses={"demo": [1, 2]}))
