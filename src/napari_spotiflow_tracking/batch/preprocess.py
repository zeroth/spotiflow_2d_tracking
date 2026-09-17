"""Ordered preprocessing chain mirroring the Pre-Processing widget.

A step is a name or a one-key mapping ``{name: {param: value}}``. Each wraps
one function from ``_preprocessing`` with the same arguments the widget uses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .._preprocessing import (
    apply_calculation,
    crop_image,
    denoise_n2v,
    gaussian_blur,
    remove_background_erosion,
    remove_background_high_pass_stack,
    walking_average,
)
from .config import ConfigError, ConfigWarning


@dataclass(frozen=True)
class Step:
    func: Callable[[np.ndarray, dict, str], np.ndarray]
    params: dict = field(default_factory=dict)  # accepted keys → defaults
    required: tuple = ()


def _crop(stack, p, device):
    (y0, y1), (x0, x1) = p["y"], p["x"]
    return crop_image(stack, (int(y0), int(y1), int(x0), int(x1)))


def _scalar(op: str):
    def run(stack, p, device):
        value = p["value"]
        if value == "mean":
            value = round(float(np.mean(stack)))
        return apply_calculation(stack, op, int(value))
    return run


def _walking_average(stack, p, device):
    return walking_average(stack, window=int(p["window"]))


def _gaussian_blur(stack, p, device):
    return gaussian_blur(stack, float(p["sigma"]))


def _background(stack, p, device):
    if p["method"] == "high_pass":
        return remove_background_high_pass_stack(stack, sigma=float(p["sigma"]), device=device)
    return np.stack([remove_background_erosion(frame, disk_size=int(p["disk_size"]))
                     for frame in stack])


def _n2v(stack, p, device):
    return denoise_n2v(
        stack,
        model_path=p["model_path"],
        n_epochs=int(p["n_epochs"]),
        patch_size=int(p["patch_size"]),
        mode="2D",
        train_frames=p["train_frames"],
    )


STEPS = {
    "crop": Step(_crop, {"y": None, "x": None}, required=("y", "x")),
    "add": Step(_scalar("add"), {"value": None}, required=("value",)),
    "subtract": Step(_scalar("subtract"), {"value": None}, required=("value",)),
    "walking_average": Step(_walking_average, {"window": 3}),
    "gaussian_blur": Step(_gaussian_blur, {"sigma": 1.0}),
    "background": Step(_background, {"method": "high_pass", "sigma": 10.0, "disk_size": 10}),
    "n2v": Step(_n2v, {"model_path": None, "n_epochs": 100, "patch_size": 64,
                       "train_frames": None}),
}


def parse_step(entry: Any, index: int) -> tuple:
    """Return ``(name, params with defaults)`` or raise :class:`ConfigError`."""
    where = f"preprocessing.steps[{index}]"
    if isinstance(entry, str):
        name, params = entry, {}
    elif isinstance(entry, dict) and len(entry) == 1:
        name, params = next(iter(entry.items()))
        params = {} if params is None else params
    else:
        raise ConfigError(
            f"{where} must be a step name or a one-key mapping like "
            "{gaussian_blur: {sigma: 1.0}}"
        )
    if name not in STEPS:
        raise ConfigError(f"{where}: unknown step '{name}'; choose one of: {', '.join(STEPS)}")
    if not isinstance(params, dict):
        raise ConfigError(f"{where}.{name}: parameters must be a mapping")
    step = STEPS[name]
    for key in params:
        if key not in step.params:
            warnings.warn(f"{where}.{name}: unknown parameter '{key}' is ignored",
                          ConfigWarning, stacklevel=2)
    merged = {**step.params, **{k: v for k, v in params.items() if k in step.params}}
    missing = [key for key in step.required if merged.get(key) is None]
    if missing:
        raise ConfigError(f"{where}.{name}: missing {', '.join(missing)}")
    if name in ("add", "subtract"):
        value = merged["value"]
        if value != "mean" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"{where}.{name}: value must be a number or 'mean'")
    if name == "background" and merged["method"] not in ("high_pass", "erosion"):
        raise ConfigError(f"{where}.background: method must be high_pass or erosion")
    return name, merged


def validate_steps(steps: Any) -> list:
    if not isinstance(steps, list):
        return ["preprocessing.steps must be a list"]
    errors = []
    for index, entry in enumerate(steps):
        try:
            parse_step(entry, index)
        except ConfigError as exc:
            errors.append(str(exc))
    return errors


def run_steps(stack: np.ndarray, steps: Sequence, device: str = "cpu") -> np.ndarray:
    out = stack
    for index, entry in enumerate(steps):
        name, params = parse_step(entry, index)
        out = STEPS[name].func(out, params, device)
    return out
