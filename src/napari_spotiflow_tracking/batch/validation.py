"""Whole-config validation, run before any movie is read."""

from __future__ import annotations

import warnings

from .analyses import base
from .config import FIGURE_FORMATS, MODALITIES, ConfigError, ConfigWarning, PipelineConfig
from .outputs import OutputSelector
from .preprocess import validate_steps


def validate(cfg: PipelineConfig) -> None:
    errors = []
    if cfg.modality not in MODALITIES:
        errors.append(f"modality must be one of: {', '.join(MODALITIES)}")
    if not cfg.inputs:
        errors.append("inputs: at least one input is required")
    for index, spec in enumerate(cfg.inputs):
        if not spec.path:
            errors.append(f"inputs[{index}].path is required")
    if not cfg.output_dir:
        errors.append("output_dir is required")
    if cfg.device not in ("cuda", "cpu"):
        errors.append("device must be cuda or cpu")
    if cfg.measure_on not in ("raw", "preprocessed"):
        errors.append("measure_on must be raw or preprocessed")
    if cfg.detection.method not in ("spotiflow", "log"):
        errors.append("detection.method must be spotiflow or log")
    if cfg.outputs.save_masks and not cfg.detection.fit.enabled:
        errors.append("outputs.save_masks needs detection.fit.enabled "
                      "(masks come from the Gaussian fit)")
    for fmt in cfg.outputs.formats:
        if fmt not in FIGURE_FORMATS:
            errors.append(f"outputs.formats: unsupported '{fmt}'; "
                          f"choose from {', '.join(FIGURE_FORMATS)}")
    errors.extend(validate_steps(cfg.preprocessing.steps))

    for name, section in cfg.analyses.items():
        analysis = base.get_analysis(name)
        if analysis is None:
            warnings.warn(f"unknown analysis 'analyses.{name}' is ignored; known: "
                          f"{', '.join(sorted(base.REGISTRY)) or 'none'}",
                          ConfigWarning, stacklevel=2)
            continue
        if section is not None and not isinstance(section, dict):
            errors.append(f"analyses.{name} must be a mapping")
            continue
        settings = base.settings_for(analysis, section)
        if not settings.pop("enabled", True):
            continue
        if analysis.requires_tracks and not cfg.tracking.enabled:
            errors.append(f"analyses.{name} needs tracking; set tracking.enabled: true")
        if settings.get("measure_on") is not None and \
                settings["measure_on"] not in ("raw", "preprocessed"):
            errors.append(f"analyses.{name}.measure_on must be raw or preprocessed")
        errors.extend(f"analyses.{name}: {problem}" for problem in analysis.validate(settings))
        try:
            OutputSelector(settings, analysis.catalogue, name)
        except ConfigError as exc:
            errors.append(str(exc))

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
