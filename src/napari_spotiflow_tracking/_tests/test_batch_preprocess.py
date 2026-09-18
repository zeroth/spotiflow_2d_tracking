"""Preprocessing steps must equal the direct logic-function calls."""
import numpy as np
import pytest

from napari_spotiflow_tracking import _preprocessing as pp
from napari_spotiflow_tracking.batch import preprocess
from napari_spotiflow_tracking.batch.config import ConfigError, ConfigWarning


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    return rng.integers(900, 1300, size=(5, 24, 24)).astype(np.uint16)


def test_empty_chain_returns_input(stack):
    assert preprocess.run_steps(stack, []) is stack


def test_crop(stack):
    out = preprocess.run_steps(stack, [{"crop": {"y": [2, 10], "x": [4, 20]}}])
    np.testing.assert_array_equal(out, pp.crop_image(stack, (2, 10, 4, 20)))


def test_subtract_mean_rounds_stack_mean(stack):
    out = preprocess.run_steps(stack, [{"subtract": {"value": "mean"}}])
    expected = pp.apply_calculation(stack, "subtract", int(round(float(np.mean(stack)))))
    np.testing.assert_array_equal(out, expected)


def test_add_number(stack):
    out = preprocess.run_steps(stack, [{"add": {"value": 7}}])
    np.testing.assert_array_equal(out, pp.apply_calculation(stack, "add", 7))


def test_chain_order_matches_manual(stack):
    steps = [
        {"walking_average": {"window": 3}},
        {"subtract": {"value": "mean"}},
        {"gaussian_blur": {"sigma": 1.0}},
    ]
    manual = pp.walking_average(stack, window=3)
    manual = pp.apply_calculation(manual, "subtract", int(round(float(np.mean(manual)))))
    manual = pp.gaussian_blur(manual, 1.0)
    np.testing.assert_array_equal(preprocess.run_steps(stack, steps), manual)


def test_bare_name_uses_defaults(stack):
    out = preprocess.run_steps(stack, ["gaussian_blur"])
    np.testing.assert_array_equal(out, pp.gaussian_blur(stack, 1.0))


def test_background_high_pass(stack):
    out = preprocess.run_steps(stack, [{"background": {"method": "high_pass", "sigma": 5}}])
    expected = pp.remove_background_high_pass_stack(stack, sigma=5.0, device="cpu")
    np.testing.assert_array_equal(out, expected)


def test_background_erosion_is_per_frame(stack):
    out = preprocess.run_steps(stack, [{"background": {"method": "erosion", "disk_size": 3}}])
    expected = np.stack([pp.remove_background_erosion(f, disk_size=3) for f in stack])
    np.testing.assert_array_equal(out, expected)


def test_n2v_passes_parameters(stack, monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen.update(kwargs)
        return image + 1

    monkeypatch.setattr(preprocess, "denoise_n2v", fake)
    out = preprocess.run_steps(stack, [{"n2v": {"n_epochs": 3, "patch_size": 16}}])
    assert seen == {"model_path": None, "n_epochs": 3, "patch_size": 16,
                    "mode": "2D", "train_frames": None}
    np.testing.assert_array_equal(out, stack + 1)


@pytest.mark.parametrize("entry, message", [
    ({"sharpen": {}}, "unknown step 'sharpen'"),
    ({"crop": {"y": [0, 5]}}, "missing x"),
    ({"subtract": {"value": "median"}}, "number or 'mean'"),
    ({"background": {"method": "rolling_ball"}}, "high_pass or erosion"),
    ({"a": {}, "b": {}}, "one-key mapping"),
    (5, "one-key mapping"),
    ({"gaussian_blur": 2}, "parameters must be a mapping"),
])
def test_invalid_steps(entry, message):
    with pytest.raises(ConfigError, match=message):
        preprocess.parse_step(entry, 0)


def test_unknown_parameter_warns():
    with pytest.warns(ConfigWarning, match="unknown parameter 'sigmaa'"):
        name, params = preprocess.parse_step({"gaussian_blur": {"sigmaa": 3}}, 2)
    assert (name, params) == ("gaussian_blur", {"sigma": 1.0})


def test_validate_steps_collects_all_errors():
    errors = preprocess.validate_steps([{"sharpen": {}}, "gaussian_blur", {"crop": {}}])
    assert len(errors) == 2
    assert errors[0].startswith("preprocessing.steps[0]")
    assert errors[1].startswith("preprocessing.steps[2]")
    assert preprocess.validate_steps("gaussian_blur") == ["preprocessing.steps must be a list"]


@pytest.mark.parametrize("params", [
    {"y": [1, 2, 3], "x": [0, 4]},
    {"y": [0, 4], "x": 5},
    {"y": [0, "a"], "x": [0, 4]},
    {"y": [0, True], "x": [0, 4]},
])
def test_crop_bounds_must_be_two_numbers(params):
    with pytest.raises(ConfigError, match="must be a list of two numbers"):
        preprocess.parse_step({"crop": params}, 0)
