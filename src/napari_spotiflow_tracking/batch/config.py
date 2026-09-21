"""Batch pipeline configuration.

A run is described by a YAML file. Values are resolved in this order, later
winning: dataclass defaults < ``presets/<modality>.yaml`` < user YAML <
``--set key.path=value`` overrides. Unknown keys warn and are ignored so a
config written for a newer version still runs.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Union, get_args, get_origin,
                    get_type_hints)

import yaml

PRESET_DIR = Path(__file__).parent / "presets"
MODALITIES = ("tracking", "papa", "bleaching")
FIGURE_FORMATS = ("png", "pdf", "svg")


class ConfigError(ValueError):
    """The config cannot be run."""


class ConfigWarning(UserWarning):
    """Something in the config is ignored (unknown key, unknown output name)."""


@dataclass
class InputSpec:
    path: str = ""
    label: Optional[str] = None
    pattern: str = "*.tif"
    recursive: bool = False
    color: Optional[str] = None
    # Per-movie overrides (see batch/overrides.py): ``settings`` applies to every
    # movie in this entry, ``per_file`` to movies matching its keys. Both are
    # restricted to overrides.ALLOWED_SECTIONS; validated in validation.validate
    # and batch/overrides.validate_all (which needs the expanded jobs).
    settings: Dict[str, Any] = field(default_factory=dict)
    per_file: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Calibration:
    pixel_size_nm: Optional[float] = None
    frame_interval_s: Optional[float] = None

    @property
    def is_physical(self) -> bool:
        return self.pixel_size_nm is not None and self.frame_interval_s is not None


@dataclass
class SpotiflowSettings:
    model: str = "general"
    prob_thresh: Optional[float] = 0.5
    min_distance: int = 2


@dataclass
class LogSettings:
    min_sigma: float = 2.0
    max_sigma: float = 10.0
    num_sigma: int = 10
    threshold: float = 0.09


@dataclass
class FitSettings:
    # On by default: every spot is fitted with a 2D Gaussian, using the batched
    # torchimize backend when importable and a per-frame scipy fallback otherwise
    # (see batch/detect.py:fit_backend). spots.csv always carries the fit and
    # region-property columns as a result.
    enabled: bool = True
    window: int = 2
    max_radius: Optional[float] = 2.0


@dataclass
class DetectionSettings:
    method: str = "spotiflow"
    spotiflow: SpotiflowSettings = field(default_factory=SpotiflowSettings)
    log: LogSettings = field(default_factory=LogSettings)
    fit: FitSettings = field(default_factory=FitSettings)


@dataclass
class TrackingSettings:
    enabled: bool = True
    search_range: float = 3.0
    memory: int = 10


@dataclass
class OutputSettings:
    formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    dpi: int = 300
    save_preprocessed: bool = False
    # On by default: mask.tif is written alongside spots.csv unless set to false.
    # Requires detection.fit.enabled (the mask comes from the Gaussian fit).
    save_masks: bool = True


@dataclass
class PreprocessingSettings:
    steps: List[Any] = field(default_factory=list)


@dataclass
class PipelineConfig:
    modality: str = "tracking"
    inputs: List[InputSpec] = field(default_factory=list)
    output_dir: str = ""
    skip_existing: bool = False
    device: str = "cuda"
    calibration: Calibration = field(default_factory=Calibration)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    # Unset means "each analysis's own default" (Analysis.default_measure_on);
    # an explicit "raw"/"preprocessed" here overrides every analysis that does
    # not set its own analyses.<name>.measure_on.
    measure_on: Optional[str] = None
    outputs: OutputSettings = field(default_factory=OutputSettings)
    analyses: Dict[str, Any] = field(default_factory=dict)


def deep_merge(base: dict, override: dict) -> dict:
    """Copy of ``base`` with ``override`` merged in; nested dicts merge, lists replace."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_preset(modality: str) -> dict:
    path = PRESET_DIR / f"{modality}.yaml"
    if modality not in MODALITIES or not path.exists():
        raise ConfigError(
            f"unknown modality '{modality}'; choose one of: {', '.join(MODALITIES)}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_override(text: str) -> tuple:
    """``"a.b=3"`` → ``(["a", "b"], 3)``; the value is parsed as YAML."""
    if "=" not in text:
        raise ConfigError(f"--set expects key.path=value, got '{text}'")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"--set has an empty key: '{text}'")
    value = yaml.safe_load(raw) if raw.strip() else None
    return key.split("."), value


def _apply_override(data: dict, keys: list, value: Any) -> None:
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _type_name(hint: Any) -> str:
    if hint in (bool, int, float, str):
        return hint.__name__
    args = get_args(hint)
    if get_origin(hint) is list:
        return f"list of {_type_name(args[0])}"
    if get_origin(hint) is Union:
        inner = [arg for arg in args if arg is not type(None)]
        return f"{_type_name(inner[0])} or null"
    return str(hint)


def _check_type(hint: Any, value: Any, sub: str) -> Any:
    """Return ``value`` (an int converted for a float field) or raise :class:`ConfigError`.

    Only scalars, ``List[scalar]`` and ``Optional[scalar]`` are checked; ``Any``
    and containers of ``Any`` are left to the code that reads them.
    """
    def fail():
        raise ConfigError(f"{sub} must be {_type_name(hint)}, got {value!r}")

    if hint is bool:
        if not isinstance(value, bool):
            fail()
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            fail()
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail()
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            fail()
        return value
    origin, args = get_origin(hint), get_args(hint)
    if origin is list and args and args[0] in (bool, int, float, str):
        if not isinstance(value, list):
            fail()
        try:
            return [_check_type(args[0], item, sub) for item in value]
        except ConfigError:
            fail()
    if origin is Union:
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1 and len(args) == 2:
            if value is None:
                return None
            try:
                return _check_type(inner[0], value, sub)
            except ConfigError:
                fail()
    return value


def _build(cls, data: Any, where: str):
    """Instantiate dataclass ``cls`` from a mapping, warning on unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'} must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    for key in data:
        if key not in known:
            warnings.warn(
                f"unknown config key '{_join(where, key)}' is ignored",
                ConfigWarning,
                stacklevel=3,
            )
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        sub = _join(where, f.name)
        hint = hints[f.name]
        if isinstance(hint, type) and is_dataclass(hint):
            kwargs[f.name] = _build(hint, value, sub)
        elif cls is PipelineConfig and f.name == "inputs":
            if not isinstance(value, list):
                raise ConfigError(f"{sub} must be a list")
            kwargs[f.name] = [
                _build(InputSpec, {"path": item} if isinstance(item, str) else item,
                       f"{sub}[{i}]")
                for i, item in enumerate(value)
            ]
        elif cls is PipelineConfig and f.name == "analyses":
            if not isinstance(value, dict):
                raise ConfigError(f"{sub} must be a mapping")
            kwargs[f.name] = copy.deepcopy(value)
        else:
            kwargs[f.name] = _check_type(hint, value, sub)
    return cls(**kwargs)


def load_config(
    path: Optional[Path] = None,
    overrides: Sequence[str] = (),
    data: Optional[dict] = None,
) -> PipelineConfig:
    """Resolve a config from a YAML file and/or a dict plus ``--set`` overrides."""
    user: dict = {}
    if path is not None:
        try:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a YAML mapping")
        user = loaded or {}
    if data is not None:
        user = deep_merge(user, data)
    for text in overrides:
        keys, value = parse_override(text)
        _apply_override(user, keys, value)
    modality = user.get("modality", "tracking")
    merged = deep_merge(load_preset(modality), user)
    merged["modality"] = modality
    return _build(PipelineConfig, merged, "")


def config_to_dict(cfg: PipelineConfig) -> dict:
    return dataclasses.asdict(cfg)


def write_resolved_config(cfg: PipelineConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
