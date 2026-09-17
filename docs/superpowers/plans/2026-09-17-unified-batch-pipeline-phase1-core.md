# Unified Batch Pipeline — Phase 1 (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `spotiflow-batch`, a config-driven batch command that loads movies from files/folders, runs an ordered preprocessing chain, detects spots, optionally fits and tracks them, runs any registered analyses, and writes a run summary — with no analyses registered yet (those are phases 2–4).

**Architecture:** A new subpackage `napari_spotiflow_tracking.batch`. `config.py` turns YAML + modality preset + `--set` overrides into dataclasses; `validation.py` checks them; `inputs.py` expands inputs into movie jobs; `preprocess.py` and `detect.py` wrap the existing Qt-free logic modules; `analyses/base.py` defines the plugin hooks and registry; `runner.py` orchestrates movies, isolates failures and calls group/comparison hooks; `outputs.py` and `theme.py` give every figure and table one saving path and one look; `cli.py` exposes `run`, `validate`, `list-outputs`, `init-config`.

**Tech Stack:** Python ≥ 3.9, numpy, pandas, tifffile, PyYAML, matplotlib, trackpy, spotiflow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`

## Global Constraints

- Code runs on Python ≥ 3.9: every new module starts with `from __future__ import annotations`; no `match`; use `typing.Optional/List/Dict` in dataclass fields (resolved with `get_type_hints`).
- No Qt imports anywhere under `batch/`. Heavy optional imports (`spotiflow` model loading, `torchimize`, `careamics`) stay lazy, inside functions.
- The existing scripts (`scripts/*.py`) are not modified.
- Unknown config keys and unknown output names → `ConfigWarning` (a `UserWarning`), never an error.
- Precedence: built-in dataclass defaults < `presets/<modality>.yaml` < user YAML < `--set` overrides.
- Exit codes: `0` success, `1` any movie or hook failed, `2` config error.
- Figures are saved through temp file + `os.replace`; a locked target is written as `<name>_new.<ext>` with a logged warning.
- Batch code builds figures with `matplotlib.figure.Figure`, never `pyplot`, and never calls `matplotlib.use` — the package also runs inside napari and alongside Qt widget tests.
- Tests live flat in `src/napari_spotiflow_tracking/_tests/` as `test_batch_*.py`, run with `.venv/Scripts/python.exe -m pytest` (always the workspace `.venv`), need no GPU and no real data.
- Commit only the files a task names (`git add <paths>`); the working tree has unrelated uncommitted changes that must stay out of these commits.

## Spec clarifications applied in this plan

Found while reading the code; the spec is updated in Task 9:

1. `apply_calculation` supports only `add` and `subtract`, so the arithmetic steps are `add` and `subtract` (no multiply/divide).
2. Detection fitting uses the existing stack-wide `fit_all_frames_gpu` + `make_gpu_spot_mask` path (as `bulk_track_analysis.py` does). It needs `torchimize`; there is no `backend` key.
3. Analysis settings are plain dicts merged over each analysis's `default_settings()`; a section is enabled unless it says `enabled: false`.
4. `skip_existing`: a movie whose `movie_status.json` exists is skipped entirely; without it, existing `spots.csv`/`tracks.csv` are reused and analyses re-run.
5. Phase-1 presets contain no `analyses` sections (no analyses are registered yet); phases 2–4 add them.
6. Semantic palettes: matplotlib `tab:orange`/`tab:green` are indistinguishable under protanopia (CIEDE2000 1.3), so step counts, motility classes and PAPA groups use `#56b4e9`, `#eda100`, `#4a3aa7` (+ `#7f7f7f` for unclassified), and the group palette is reordered so neighbours stay ≥ 15 apart under every simulated colour-vision deficiency.

## File Structure

```
pyproject.toml                                    modify: pyyaml + tifffile deps, console script, preset package data
src/napari_spotiflow_tracking/batch/
  __init__.py                                     package marker + public names
  config.py                                       dataclasses, preset loading, merge, --set, build, dump
  presets/tracking.yaml, papa.yaml, bleaching.yaml
  theme.py                                        rc style context, palettes, axis labels
  outputs.py                                      OutputSpec, OutputSelector, save_figure, save_table
  preprocess.py                                   step registry, parse/validate/run steps
  detect.py                                       load_detector, detect_stack, fit_spots, track_spots
  inputs.py                                       MovieJob, input_label, expand_inputs
  context.py                                      MovieContext
  analyses/__init__.py                            re-exports registry API
  analyses/base.py                                Analysis base class + registry
  validation.py                                   validate(cfg)
  runner.py                                       run(cfg) → RunResult
  cli.py                                          argparse entry point
src/napari_spotiflow_tracking/_tests/
  test_batch_config.py, test_batch_theme_outputs.py, test_batch_preprocess.py,
  test_batch_detect.py, test_batch_inputs.py, test_batch_validation.py,
  test_batch_runner.py, test_batch_cli.py
```

---

### Task 1: Package skeleton, config dataclasses, presets, overrides

**Files:**
- Modify: `pyproject.toml`
- Create: `src/napari_spotiflow_tracking/batch/__init__.py`
- Create: `src/napari_spotiflow_tracking/batch/config.py`
- Create: `src/napari_spotiflow_tracking/batch/presets/tracking.yaml`
- Create: `src/napari_spotiflow_tracking/batch/presets/papa.yaml`
- Create: `src/napari_spotiflow_tracking/batch/presets/bleaching.yaml`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `batch.config`):
  - `ConfigError(ValueError)`, `ConfigWarning(UserWarning)`
  - `MODALITIES = ("tracking", "papa", "bleaching")`, `FIGURE_FORMATS = ("png", "pdf", "svg")`
  - dataclasses `InputSpec(path: str, label: Optional[str], pattern: str="*.tif", recursive: bool=False, color: Optional[str])`, `Calibration(pixel_size_nm, frame_interval_s)` with property `is_physical -> bool`, `SpotiflowSettings(model="general", prob_thresh=0.5, min_distance=2)`, `LogSettings(min_sigma=2.0, max_sigma=10.0, num_sigma=10, threshold=0.09)`, `FitSettings(enabled=False, window=2, max_radius=2.0)`, `DetectionSettings(method="spotiflow", spotiflow, log, fit)`, `TrackingSettings(enabled=True, search_range=3.0, memory=10, min_track_length=5)`, `OutputSettings(formats=["png","svg"], dpi=300, save_preprocessed=False, save_masks=False)`, `PreprocessingSettings(steps: List[Any]=[])`, `PipelineConfig(modality, inputs, output_dir, skip_existing, device, calibration, preprocessing, detection, tracking, measure_on, outputs, analyses: Dict[str, Any])`
  - `deep_merge(base: dict, override: dict) -> dict`
  - `load_preset(modality: str) -> dict`
  - `parse_override(text: str) -> tuple[list[str], Any]`
  - `load_config(path: Optional[Path]=None, overrides: Sequence[str]=(), data: Optional[dict]=None) -> PipelineConfig`
  - `config_to_dict(cfg: PipelineConfig) -> dict`
  - `write_resolved_config(cfg: PipelineConfig, path: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking/_tests/test_batch_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking.batch'`

- [ ] **Step 3: Update `pyproject.toml`**

In `[project] dependencies` add `"pyyaml",` and `"tifffile",` after `"scipy",`. Add after the `[project.entry-points."napari.manifest"]` table:

```toml
[project.scripts]
spotiflow-batch = "napari_spotiflow_tracking.batch.cli:main"
```

Replace the package-data table with:

```toml
[tool.setuptools.package-data]
napari_spotiflow_tracking = ["napari.yaml", "batch/presets/*.yaml"]
```

- [ ] **Step 4: Create the package and presets**

`src/napari_spotiflow_tracking/batch/__init__.py`:

```python
"""Config-driven batch processing: ``spotiflow-batch run config.yaml``."""
```

`src/napari_spotiflow_tracking/batch/presets/tracking.yaml`:

```yaml
# Tracking / MSD preset: the defaults of scripts/bulk_track_analysis.py.
preprocessing:
  steps:
    - walking_average: {window: 4}
    - subtract: {value: mean}
    - gaussian_blur: {sigma: 1.0}
detection:
  method: spotiflow
  spotiflow: {model: general, prob_thresh: 0.5, min_distance: 2}
tracking:
  enabled: true
  search_range: 5.0
  memory: 3
  min_track_length: 5
```

`src/napari_spotiflow_tracking/batch/presets/papa.yaml`:

```yaml
# PAPA preset: the defaults of scripts/papa_analysis.py (no preprocessing, no tracking).
preprocessing:
  steps: []
detection:
  method: spotiflow
  spotiflow: {model: general, prob_thresh: null, min_distance: 1}
tracking:
  enabled: false
```

`src/napari_spotiflow_tracking/batch/presets/bleaching.yaml`:

```yaml
# Photobleaching-step preset: settings of scripts/bleaching_trace_analysis.py.
preprocessing:
  steps:
    - subtract: {value: mean}
    - gaussian_blur: {sigma: 1.0}
detection:
  method: spotiflow
  spotiflow: {model: general, prob_thresh: 0.5, min_distance: 2}
tracking:
  enabled: true
  search_range: 3.0
  memory: 10
  min_track_length: 10
measure_on: raw
```

- [ ] **Step 5: Write `config.py`**

`src/napari_spotiflow_tracking/batch/config.py`:

```python
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
from typing import Any, Dict, List, Optional, Sequence, get_type_hints

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
    enabled: bool = False
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
    min_track_length: int = 5


@dataclass
class OutputSettings:
    formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    dpi: int = 300
    save_preprocessed: bool = False
    save_masks: bool = False


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
    measure_on: str = "raw"
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
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(
    path: Optional[Path] = None,
    overrides: Sequence[str] = (),
    data: Optional[dict] = None,
) -> PipelineConfig:
    """Resolve a config from a YAML file and/or a dict plus ``--set`` overrides."""
    user: dict = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
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
```

- [ ] **Step 6: Reinstall and run the tests**

Run: `.venv/Scripts/python.exe -m pip install -e . -q` then `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_config.py -v`
Expected: all 15 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/napari_spotiflow_tracking/batch/__init__.py src/napari_spotiflow_tracking/batch/config.py src/napari_spotiflow_tracking/batch/presets src/napari_spotiflow_tracking/_tests/test_batch_config.py
git commit -m "feat(batch): config dataclasses, modality presets and --set overrides"
```

---

### Task 2: Theme and output saving/selection

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/theme.py`
- Create: `src/napari_spotiflow_tracking/batch/outputs.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py`

**Interfaces:**
- Consumes: `ConfigError`, `ConfigWarning`, `Calibration` from `batch.config`.
- Produces:
  - `batch.theme`: `STYLE: dict`, `style()` context manager, `GROUP_COLORS: tuple[str, ...]`, `STEP_COLORS: dict[int, str]`, `MOTILITY_COLORS: dict[str, str]`, `PAPA_COLORS: dict[str, str]`, `NEUTRAL: str`, `group_colors(labels: Sequence[str], pinned: Optional[Mapping[str, Optional[str]]] = None) -> dict[str, str]`, `axis_label(kind: str, calibration: Calibration) -> str` (kinds: `time`, `length`, `msd`, `intensity`).
  - `batch.outputs`: `OutputSpec(name, level, kind, description)` frozen dataclass; `OutputSelector(settings: Mapping, catalogue: Sequence[OutputSpec], analysis: str)` with `.figure(name) -> bool`, `.table(name) -> bool`; `save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]`; `save_table(df: pd.DataFrame, path: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py`:

```python
"""Theme palettes/style and output selection/saving."""
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from skimage.color import deltaE_ciede2000, rgb2lab

from napari_spotiflow_tracking.batch import outputs, theme
from napari_spotiflow_tracking.batch.config import Calibration, ConfigError, ConfigWarning
from napari_spotiflow_tracking.batch.outputs import OutputSelector, OutputSpec

# Machado et al. (2009) full-severity simulation matrices, applied in linear RGB.
_CVD = {
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def _lab(color, mode):
    rgb = np.array(to_rgb(color))
    if mode != "normal":
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        sim = np.clip(lin @ _CVD[mode].T, 0, 1)
        rgb = np.where(sim <= 0.0031308, 12.92 * sim, 1.055 * sim ** (1 / 2.4) - 0.055)
    return rgb2lab(rgb.reshape(1, 1, 3))[0, 0]


def _min_delta(pairs):
    return min(
        deltaE_ciede2000(_lab(a, m), _lab(b, m))
        for m in ("normal", "deutan", "protan", "tritan")
        for a, b in pairs
    )


def test_semantic_palette_separates_under_cvd():
    colors = [theme.STEP_COLORS[1], theme.STEP_COLORS[2], theme.STEP_COLORS[3], theme.NEUTRAL]
    assert _min_delta(itertools.combinations(colors, 2)) >= 20


def test_semantic_palettes_share_colors():
    assert theme.MOTILITY_COLORS["constrained"] == theme.STEP_COLORS[1]
    assert theme.MOTILITY_COLORS["diffusive"] == theme.STEP_COLORS[2]
    assert theme.MOTILITY_COLORS["directed"] == theme.STEP_COLORS[3]
    assert theme.PAPA_COLORS["group1"] == theme.STEP_COLORS[1]
    assert theme.PAPA_COLORS["group2"] == theme.STEP_COLORS[2]


def test_group_palette_neighbours_and_first_four_separate():
    g = theme.GROUP_COLORS
    assert len(g) == 8
    assert _min_delta(zip(g, g[1:])) >= 15
    assert _min_delta(itertools.combinations(g[:4], 2)) >= 12


def test_group_colors_follow_input_order_and_pins():
    colors = theme.group_colors(["A", "B", "C"], pinned={"B": "#000000", "C": None})
    assert colors == {"A": theme.GROUP_COLORS[0], "B": "#000000", "C": theme.GROUP_COLORS[2]}


def test_style_does_not_leak():
    before = dict(matplotlib.rcParams)
    with theme.style():
        assert matplotlib.rcParams["axes.spines.top"] is False
    assert dict(matplotlib.rcParams) == before


def test_axis_labels_follow_calibration():
    px = Calibration()
    um = Calibration(pixel_size_nm=100, frame_interval_s=0.1)
    assert theme.axis_label("time", px) == "Time (frames)"
    assert theme.axis_label("time", um) == "Time (s)"
    assert theme.axis_label("length", um) == "Distance (µm)"
    assert theme.axis_label("msd", px) == "MSD (px²)"
    assert theme.axis_label("intensity", um) == "Intensity (a.u.)"
    with pytest.raises(ValueError):
        theme.axis_label("colour", px)


CATALOGUE = [
    OutputSpec("traces", "movie", "figure", "Traces"),
    OutputSpec("sheet", "movie", "figure", "Sheet"),
    OutputSpec("steps", "movie", "table", "Steps"),
]


def test_selector_all_none_and_lists():
    sel = OutputSelector({}, CATALOGUE, "demo")
    assert sel.figure("traces") and sel.figure("sheet") and sel.table("steps")
    sel = OutputSelector({"figures": "none", "tables": ["steps"]}, CATALOGUE, "demo")
    assert not sel.figure("traces") and sel.table("steps")
    sel = OutputSelector({"figures": "sheet"}, CATALOGUE, "demo")
    assert sel.figure("sheet") and not sel.figure("traces")


def test_selector_warns_on_unknown_name():
    with pytest.warns(ConfigWarning, match="analyses.demo.figures: unknown output 'trace'"):
        sel = OutputSelector({"figures": ["trace", "sheet"]}, CATALOGUE, "demo")
    assert sel.figure("sheet") and not sel.figure("traces")


def test_selector_rejects_bad_type():
    with pytest.raises(ConfigError, match="analyses.demo.tables"):
        OutputSelector({"tables": 3}, CATALOGUE, "demo")


def test_save_figure_writes_each_format(tmp_path):
    fig = Figure()
    fig.subplots().plot([0, 1], [0, 1])
    written = outputs.save_figure(fig, tmp_path / "sub" / "plot", ["png", "svg"], dpi=50)
    assert [p.name for p in written] == ["plot.png", "plot.svg"]
    assert all(p.exists() for p in written)
    assert not list((tmp_path / "sub").glob("*.tmp.*"))


def test_save_figure_falls_back_when_target_locked(tmp_path, monkeypatch):
    real_replace = os.replace

    def locked(src, dst):
        if str(dst).endswith("plot.png"):
            raise OSError(22, "Invalid argument")
        return real_replace(src, dst)

    monkeypatch.setattr(outputs.os, "replace", locked)
    written = outputs.save_figure(Figure(), tmp_path / "plot", ["png"], dpi=50)
    assert [p.name for p in written] == ["plot_new.png"]


def test_save_table_creates_parent(tmp_path):
    path = outputs.save_table(pd.DataFrame({"a": [1, 2]}), tmp_path / "x" / "t.csv")
    assert pd.read_csv(path)["a"].tolist() == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py -v`
Expected: FAIL — `ImportError: cannot import name 'outputs'`

- [ ] **Step 3: Write `theme.py`**

`src/napari_spotiflow_tracking/batch/theme.py`:

```python
"""One look for every batch figure: rc style, palettes and axis labels.

Palettes were checked with CIEDE2000 under Machado (2009) deutan, protan and
tritan simulation (see test_batch_theme_outputs). matplotlib's tab:orange and
tab:green collapse to ΔE 1.3 under protanopia, so they are not used.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional, Sequence

import matplotlib

from .config import Calibration

#: Categorical hues for groups, assigned in input order. Ordered so every
#: neighbouring pair and the first four stay apart under colour-vision deficiency.
GROUP_COLORS = ("#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6",
                "#e87ba4", "#008300", "#eda100", "#e34948")

_SKY, _AMBER, _INDIGO = "#56b4e9", "#eda100", "#4a3aa7"
NEUTRAL = "#7f7f7f"
#: Monomer / dimer / trimer.
STEP_COLORS = {1: _SKY, 2: _AMBER, 3: _INDIGO}
MOTILITY_COLORS = {"constrained": _SKY, "diffusive": _AMBER,
                   "directed": _INDIGO, "unclassified": NEUTRAL}
PAPA_COLORS = {"group1": _SKY, "group2": _AMBER, "ratio": _INDIGO}
#: Raw data drawn behind a fit.
RAW_TRACE = "#9a9a9a"

# Type sized for a figure printed at column width (bulk_track_analysis.py).
STYLE = {
    "font.size": 17,
    "axes.labelsize": 19,
    "axes.titlesize": 15,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "lines.linewidth": 2.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}


@contextmanager
def style():
    """Apply :data:`STYLE` for the duration of the block only."""
    with matplotlib.rc_context(STYLE):
        yield


def group_colors(
    labels: Sequence[str],
    pinned: Optional[Mapping[str, Optional[str]]] = None,
) -> dict:
    """Colour per group label: the input-order slot, unless pinned in the config."""
    pinned = pinned or {}
    return {
        label: pinned.get(label) or GROUP_COLORS[i % len(GROUP_COLORS)]
        for i, label in enumerate(labels)
    }


def axis_label(kind: str, calibration: Calibration) -> str:
    physical = calibration.is_physical
    labels = {
        "time": "Time (s)" if physical else "Time (frames)",
        "length": "Distance (µm)" if physical else "Distance (px)",
        "msd": "MSD (µm²)" if physical else "MSD (px²)",
        "intensity": "Intensity (a.u.)",
    }
    if kind not in labels:
        raise ValueError(f"unknown axis kind '{kind}'; choose one of: {', '.join(labels)}")
    return labels[kind]
```

- [ ] **Step 4: Write `outputs.py`**

`src/napari_spotiflow_tracking/batch/outputs.py`:

```python
"""Output catalogues, user selection, and lock-safe saving."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .config import ConfigError, ConfigWarning

log = logging.getLogger("napari_spotiflow_tracking.batch")


@dataclass(frozen=True)
class OutputSpec:
    """One figure or table an analysis can write."""

    name: str
    level: str  # "movie" | "group" | "comparison"
    kind: str  # "figure" | "table"
    description: str


class OutputSelector:
    """Which catalogue outputs the user asked for (``figures``/``tables`` keys)."""

    def __init__(self, settings: Mapping, catalogue: Sequence[OutputSpec], analysis: str):
        self._chosen = {
            kind: self._resolve(settings.get(f"{kind}s", "all"), kind, catalogue, analysis)
            for kind in ("figure", "table")
        }

    @staticmethod
    def _resolve(value, kind: str, catalogue, analysis: str) -> set:
        names = {spec.name for spec in catalogue if spec.kind == kind}
        key = f"analyses.{analysis}.{kind}s"
        if value is None or value == "none":
            return set()
        if value == "all":
            return names
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be 'all', 'none' or a list of names")
        for name in value:
            if name not in names:
                warnings.warn(
                    f"{key}: unknown output '{name}' is ignored; "
                    f"available: {', '.join(sorted(names)) or 'none'}",
                    ConfigWarning,
                    stacklevel=3,
                )
        return {name for name in value if name in names}

    def figure(self, name: str) -> bool:
        return name in self._chosen["figure"]

    def table(self, name: str) -> bool:
        return name in self._chosen["table"]


def _replace(tmp: Path, target: Path) -> Path:
    """Move ``tmp`` onto ``target``; if the target is locked, beside it as ``*_new``."""
    try:
        os.replace(tmp, target)
        return target
    except OSError:
        fallback = target.with_name(f"{target.stem}_new{target.suffix}")
        os.replace(tmp, fallback)
        log.warning("%s is locked (open in a viewer?); wrote %s instead",
                    target, fallback.name)
        return fallback


def save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> list:
    """Save ``fig`` as ``<stem>.<ext>`` for every format."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = stem.with_name(f"{stem.name}.{ext}")
        tmp = stem.with_name(f"{stem.name}.tmp.{ext}")
        fig.savefig(tmp, dpi=dpi)
        written.append(_replace(tmp, target))
    return written


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(tmp, index=False)
    return _replace(tmp, path)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/theme.py src/napari_spotiflow_tracking/batch/outputs.py src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py
git commit -m "feat(batch): colourblind-checked theme and lock-safe output saving"
```

---

### Task 3: Preprocessing step chain

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/preprocess.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_preprocess.py`

**Interfaces:**
- Consumes: `ConfigError`, `ConfigWarning` from `batch.config`; `crop_image`, `apply_calculation`, `walking_average`, `gaussian_blur`, `remove_background_high_pass_stack`, `remove_background_erosion`, `denoise_n2v` from `napari_spotiflow_tracking._preprocessing`.
- Produces (in `batch.preprocess`):
  - `STEPS: dict[str, Step]` with names `crop`, `add`, `subtract`, `walking_average`, `gaussian_blur`, `background`, `n2v`
  - `parse_step(entry: Any, index: int) -> tuple[str, dict]`
  - `validate_steps(steps: Any) -> list[str]` (error strings)
  - `run_steps(stack: np.ndarray, steps: Sequence, device: str = "cpu") -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_preprocess.py -v`
Expected: FAIL — `ImportError: cannot import name 'preprocess'`

- [ ] **Step 3: Write `preprocess.py`**

`src/napari_spotiflow_tracking/batch/preprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_preprocess.py -v`
Expected: all 18 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/preprocess.py src/napari_spotiflow_tracking/_tests/test_batch_preprocess.py
git commit -m "feat(batch): ordered preprocessing chain over the widget operations"
```

---

### Task 4: Detection, fitting and tracking stages

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/detect.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_detect.py`

**Interfaces:**
- Consumes: `DetectionSettings`, `FitSettings`, `TrackingSettings` from `batch.config`; `detect_spots`, `detect_spots_log`, `load_model` from `_segmentation`; `fit_all_frames_gpu`, `make_gpu_spot_mask` from `_fitting`; `track_blobs` from `_tracking`.
- Produces (in `batch.detect`):
  - `SPOT_COLUMNS = ["frame", "y", "x"]`
  - `load_detector(detection: DetectionSettings, device: str)` → Spotiflow model or `None` for LoG
  - `detect_stack(stack: np.ndarray, detection: DetectionSettings, model) -> pd.DataFrame` (columns `frame:int, y:float, x:float`)
  - `fit_spots(stack, spots: pd.DataFrame, fit: FitSettings, device: str) -> tuple[pd.DataFrame, np.ndarray]`
  - `track_spots(spots: pd.DataFrame, tracking: TrackingSettings) -> pd.DataFrame` (adds `particle`, keeps extra spot columns, drops tracks shorter than `min_track_length`, sorted by `particle, frame`)
  - `torchimize_available() -> bool`

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_detect.py`:

```python
"""Detection (LoG, fake Spotiflow), optional fit, and tracking stages."""
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch import detect
from napari_spotiflow_tracking.batch.config import (
    DetectionSettings,
    FitSettings,
    TrackingSettings,
)


def _blob_stack(n_frames=4, size=48, centres=((12, 12), (34, 30))):
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    for t in range(n_frames):
        for cy, cx in centres:
            stack[t] += 1000 * np.exp(-((yy - cy) ** 2 + (xx - (cx + t)) ** 2) / (2 * 2.0 ** 2))
    return stack.astype(np.uint16)


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        t = len(self.calls) - 1
        return np.array([[5.0, 6.0 + t], [20.0, 21.0]]), None


def test_log_detection_finds_blobs():
    settings = DetectionSettings(method="log")
    spots = detect.detect_stack(_blob_stack(), settings, model=None)
    assert list(spots.columns) == ["frame", "y", "x"]
    assert spots["frame"].dtype.kind == "i"
    first = spots[spots.frame == 0][["y", "x"]].to_numpy()
    for centre in ((12, 12), (34, 30)):
        assert np.min(np.hypot(*(first - centre).T)) < 2.0


def test_spotiflow_detection_uses_model_and_settings():
    model = FakeModel()
    settings = DetectionSettings()
    settings.spotiflow.prob_thresh = 0.4
    settings.spotiflow.min_distance = 3
    spots = detect.detect_stack(np.zeros((3, 32, 32), np.uint16), settings, model)
    assert len(spots) == 6
    assert spots["frame"].tolist() == [0, 0, 1, 1, 2, 2]
    assert model.calls[0] == {"prob_thresh": 0.4, "min_distance": 3, "verbose": False}


def test_empty_detection_has_columns():
    class Empty:
        def predict(self, image, **kwargs):
            return np.empty((0, 2)), None

    spots = detect.detect_stack(np.zeros((2, 8, 8)), DetectionSettings(), Empty())
    assert list(spots.columns) == ["frame", "y", "x"]
    assert len(spots) == 0


def test_load_detector_returns_none_for_log():
    assert detect.load_detector(DetectionSettings(method="log"), "cpu") is None


def test_tracking_links_filters_and_keeps_columns():
    rows = []
    for t in range(5):
        rows.append({"frame": t, "y": 10.0, "x": 10.0 + 0.5 * t, "amp": 1.0 + t})
        rows.append({"frame": t, "y": 40.0, "x": 40.0, "amp": 9.0})
    rows.append({"frame": 0, "y": 80.0, "x": 80.0, "amp": 0.0})  # one-frame spot
    spots = pd.DataFrame(rows)
    tracks = detect.track_spots(spots, TrackingSettings(search_range=3, memory=0,
                                                       min_track_length=3))
    assert tracks["particle"].nunique() == 2
    assert len(tracks) == 10
    assert "amp" in tracks.columns
    first = tracks[tracks.particle == tracks.particle.iloc[0]]
    assert first["frame"].tolist() == [0, 1, 2, 3, 4]


def test_tracking_empty_spots():
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    tracks = detect.track_spots(empty, TrackingSettings())
    assert "particle" in tracks.columns and len(tracks) == 0


def test_fit_empty_spots_returns_empty_mask():
    stack = np.zeros((2, 8, 8), np.uint16)
    empty = pd.DataFrame({"frame": pd.Series(dtype=int), "y": pd.Series(dtype=float),
                          "x": pd.Series(dtype=float)})
    spots, mask = detect.fit_spots(stack, empty, FitSettings(enabled=True), "cpu")
    assert len(spots) == 0 and mask.shape == stack.shape and not mask.any()


def test_fit_spots_adds_fit_columns():
    pytest.importorskip("torchimize")
    stack = _blob_stack(n_frames=2)
    spots = detect.detect_stack(stack, DetectionSettings(method="log"), model=None)
    fitted, mask = detect.fit_spots(stack, spots, FitSettings(enabled=True), "cpu")
    assert {"frame", "y", "x", "sigma_x", "sigma_y", "spot_radius"} <= set(fitted.columns)
    assert mask.shape == stack.shape and mask.any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_detect.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect'`

- [ ] **Step 3: Write `detect.py`**

`src/napari_spotiflow_tracking/batch/detect.py`:

```python
"""Detection, optional 2D Gaussian fit + masks, and trackpy linking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._segmentation import detect_spots, detect_spots_log
from .._tracking import track_blobs
from .config import DetectionSettings, FitSettings, TrackingSettings

SPOT_COLUMNS = ["frame", "y", "x"]


def torchimize_available() -> bool:
    try:
        import torchimize  # noqa: F401
    except ImportError:
        return False
    return True


def load_detector(detection: DetectionSettings, device: str):
    """Spotiflow model for ``method: spotiflow``; ``None`` for LoG."""
    if detection.method != "spotiflow":
        return None
    from .._segmentation import load_model

    return load_model(detection.spotiflow.model, device=device)


def detect_stack(stack: np.ndarray, detection: DetectionSettings, model) -> pd.DataFrame:
    frames, ys, xs = [], [], []
    for t in range(stack.shape[0]):
        if detection.method == "log":
            log = detection.log
            points = detect_spots_log(stack[t], min_sigma=log.min_sigma,
                                      max_sigma=log.max_sigma, num_sigma=log.num_sigma,
                                      threshold=log.threshold)
        else:
            sf = detection.spotiflow
            points, _ = detect_spots(stack[t], model, prob_thresh=sf.prob_thresh,
                                     min_distance=sf.min_distance)
        points = np.asarray(points)
        if len(points) == 0:
            continue
        frames.extend([t] * len(points))
        ys.extend(points[:, 0].astype(float))
        xs.extend(points[:, 1].astype(float))
    return pd.DataFrame({
        "frame": np.asarray(frames, dtype=np.int64),
        "y": np.asarray(ys, dtype=float),
        "x": np.asarray(xs, dtype=float),
    })


def fit_spots(stack: np.ndarray, spots: pd.DataFrame, fit: FitSettings, device: str):
    """Batched Gaussian fit + labelled mask, as scripts/bulk_track_analysis.py does."""
    if len(spots) == 0:
        return spots.copy(), np.zeros(stack.shape, dtype=np.int32)
    from .._fitting import fit_all_frames_gpu, make_gpu_spot_mask

    fitted = fit_all_frames_gpu(stack, spots.rename(columns={"frame": "time"}),
                                win=fit.window, device=device)
    mask, props = make_gpu_spot_mask(stack.shape, fitted, intensity_stack=stack,
                                     max_radius=fit.max_radius)
    return props.rename(columns={"time": "frame"}).reset_index(drop=True), mask


def track_spots(spots: pd.DataFrame, tracking: TrackingSettings) -> pd.DataFrame:
    if len(spots) == 0:
        return spots.assign(particle=pd.Series(dtype=np.int64))
    tracked = track_blobs(spots[SPOT_COLUMNS].copy(), search_range=tracking.search_range,
                          memory=tracking.memory)
    extra = spots.drop(columns=SPOT_COLUMNS)
    if len(extra.columns):
        tracked = tracked.join(extra)  # trackpy keeps the input index
    lengths = tracked.groupby("particle")["frame"].transform("size")
    tracked = tracked[lengths >= tracking.min_track_length]
    return tracked.sort_values(["particle", "frame"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_detect.py -v`
Expected: 7 PASS; `test_fit_spots_adds_fit_columns` PASS if torchimize is installed, otherwise SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/detect.py src/napari_spotiflow_tracking/_tests/test_batch_detect.py
git commit -m "feat(batch): detection, optional Gaussian fit and tracking stages"
```

---

### Task 5: Input expansion and movie context

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/inputs.py`
- Create: `src/napari_spotiflow_tracking/batch/context.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_inputs.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `InputSpec`, `ConfigError` from `batch.config`.
- Produces:
  - `batch.inputs`: `MovieJob(group: str, movie_id: str, path: Path)` frozen dataclass with property `key -> "group/movie_id"`; `input_label(spec: InputSpec) -> str`; `expand_inputs(cfg: PipelineConfig) -> list[MovieJob]`; `color_pins(cfg: PipelineConfig) -> dict[str, Optional[str]]`
  - `batch.context`: `MovieContext(job, out_dir, config, raw, preprocessed, spots, tracks=None, mask=None)` with `image(measure_on: str) -> np.ndarray` and `analysis_dir(name: str) -> Path`

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_inputs.py`:

```python
"""Input expansion into movie jobs, and the per-movie context."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch.config import ConfigError, load_config
from napari_spotiflow_tracking.batch.context import MovieContext
from napari_spotiflow_tracking.batch.inputs import MovieJob, color_pins, expand_inputs


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _cfg(inputs):
    return load_config(data={"inputs": inputs, "output_dir": "out"})


def test_folder_input_uses_folder_name_and_pattern(tmp_path):
    _touch(tmp_path / "ctrl" / "b.tif")
    _touch(tmp_path / "ctrl" / "a.tif")
    _touch(tmp_path / "ctrl" / "notes.txt")
    jobs = expand_inputs(_cfg([str(tmp_path / "ctrl")]))
    assert [(j.group, j.movie_id) for j in jobs] == [("ctrl", "a"), ("ctrl", "b")]
    assert jobs[0].key == "ctrl/a"


def test_file_input_and_label(tmp_path):
    movie = _touch(tmp_path / "field 1.tif")
    jobs = expand_inputs(_cfg([{"path": str(movie), "label": "Drug"}]))
    assert jobs == [MovieJob(group="Drug", movie_id="field 1", path=movie)]
    assert expand_inputs(_cfg([str(movie)]))[0].group == "field 1"


def test_recursive_ids_include_subfolders(tmp_path):
    _touch(tmp_path / "exp" / "day1" / "m.tif")
    _touch(tmp_path / "exp" / "day2" / "m.tif")
    jobs = expand_inputs(_cfg([{"path": str(tmp_path / "exp"), "recursive": True}]))
    assert [j.movie_id for j in jobs] == ["day1__m", "day2__m"]


def test_errors_are_collected(tmp_path):
    (tmp_path / "empty").mkdir()
    _touch(tmp_path / "a" / "x.tif")
    _touch(tmp_path / "b" / "a" / "y.tif")
    with pytest.raises(ConfigError) as info:
        expand_inputs(_cfg([
            str(tmp_path / "missing"),
            str(tmp_path / "empty"),
            str(tmp_path / "a"),
            str(tmp_path / "b" / "a"),
            {"path": str(tmp_path / "a"), "label": "bad:name"},
        ]))
    text = str(info.value)
    assert "inputs[0]" in text and "does not exist" in text
    assert "inputs[1]" in text and "no files match" in text
    assert "inputs[3]" in text and "label 'a' is used by an earlier input" in text
    assert "inputs[4]" in text and "cannot contain" in text


def test_dotted_folder_name_is_label(tmp_path):
    _touch(tmp_path / "field_1_20_7.5_l_20" / "m.tif")
    jobs = expand_inputs(_cfg([str(tmp_path / "field_1_20_7.5_l_20")]))
    assert jobs[0].group == "field_1_20_7.5_l_20"


def test_color_pins(tmp_path):
    movie = _touch(tmp_path / "m.tif")
    cfg = _cfg([{"path": str(movie), "label": "A", "color": "#123456"}])
    assert color_pins(cfg) == {"A": "#123456"}


def test_context_image_selection(tmp_path):
    raw = np.zeros((2, 4, 4))
    pre = np.ones((2, 4, 4))
    ctx = MovieContext(job=MovieJob("g", "m", tmp_path / "m.tif"), out_dir=tmp_path,
                       config=_cfg([]), raw=raw, preprocessed=pre,
                       spots=pd.DataFrame())
    assert ctx.image("raw") is raw
    assert ctx.image("preprocessed") is pre
    with pytest.raises(ValueError):
        ctx.image("denoised")
    assert ctx.analysis_dir("msd") == tmp_path / "msd"
    assert (tmp_path / "msd").is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_inputs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking.batch.context'`

- [ ] **Step 3: Write `inputs.py`**

`src/napari_spotiflow_tracking/batch/inputs.py`:

```python
"""Expand config inputs (files and folders) into per-movie jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, InputSpec, PipelineConfig

#: Labels become folder names, so they must be valid on Windows too.
_FORBIDDEN = set('<>:"/\\|?*')


@dataclass(frozen=True)
class MovieJob:
    group: str
    movie_id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.group}/{self.movie_id}"


def input_label(spec: InputSpec) -> str:
    if spec.label:
        return spec.label
    path = Path(spec.path)
    # is_file, not suffix: folders like "field_1_20_7.5_l_20" contain dots.
    return path.stem if path.is_file() else path.name


def expand_inputs(cfg: PipelineConfig) -> list:
    jobs, labels, errors = [], set(), []
    for index, spec in enumerate(cfg.inputs):
        where = f"inputs[{index}]"
        path = Path(spec.path)
        label = input_label(spec)
        if _FORBIDDEN & set(label):
            errors.append(f"{where}: label '{label}' cannot contain any of {''.join(sorted(_FORBIDDEN))}")
            continue
        if label in labels:
            errors.append(f"{where}: label '{label}' is used by an earlier input; "
                          "give each input a unique label")
            continue
        labels.add(label)
        if path.is_file():
            movies, base = [path], path.parent
        elif path.is_dir():
            found = path.rglob(spec.pattern) if spec.recursive else path.glob(spec.pattern)
            movies, base = sorted(p for p in found if p.is_file()), path
            if not movies:
                errors.append(f"{where}: no files match '{spec.pattern}' in {path}")
        else:
            errors.append(f"{where}: {path} does not exist")
            continue
        for movie in movies:
            movie_id = "__".join(movie.relative_to(base).with_suffix("").parts)
            jobs.append(MovieJob(group=label, movie_id=movie_id, path=movie))
    if errors:
        raise ConfigError("invalid inputs:\n  - " + "\n  - ".join(errors))
    return jobs


def color_pins(cfg: PipelineConfig) -> dict:
    return {input_label(spec): spec.color for spec in cfg.inputs if spec.color}
```

- [ ] **Step 4: Write `context.py`**

`src/napari_spotiflow_tracking/batch/context.py`:

```python
"""Everything an analysis gets for one movie."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .inputs import MovieJob


@dataclass
class MovieContext:
    job: MovieJob
    out_dir: Path
    config: PipelineConfig
    raw: np.ndarray
    preprocessed: np.ndarray
    spots: pd.DataFrame
    tracks: Optional[pd.DataFrame] = None
    mask: Optional[np.ndarray] = None

    def image(self, measure_on: str) -> np.ndarray:
        if measure_on == "raw":
            return self.raw
        if measure_on == "preprocessed":
            return self.preprocessed
        raise ValueError(f"measure_on must be raw or preprocessed, got '{measure_on}'")

    def analysis_dir(self, name: str) -> Path:
        path = self.out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_inputs.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/inputs.py src/napari_spotiflow_tracking/batch/context.py src/napari_spotiflow_tracking/_tests/test_batch_inputs.py
git commit -m "feat(batch): expand file/folder inputs into grouped movie jobs"
```

---

### Task 6: Analysis registry and config validation

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/__init__.py`
- Create: `src/napari_spotiflow_tracking/batch/analyses/base.py`
- Create: `src/napari_spotiflow_tracking/batch/validation.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_validation.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `ConfigError`, `ConfigWarning`, `MODALITIES`, `FIGURE_FORMATS`, `deep_merge` (Task 1); `OutputSpec`, `OutputSelector` (Task 2); `validate_steps` (Task 3); `torchimize_available` (Task 4); `MovieContext` (Task 5).
- Produces:
  - `batch.analyses.base`: class `Analysis` with class attributes `name: str`, `requires_tracks: bool`, `default_measure_on: str`, `catalogue: tuple[OutputSpec, ...]`, and methods `default_settings() -> dict`, `validate(settings: dict) -> list[str]`, `run_movie(ctx: MovieContext, settings: dict, select: OutputSelector) -> dict`, `summarize_group(group: str, movie_dirs: list[Path], settings: dict, select: OutputSelector, out_dir: Path) -> None`, `compare_groups(groups: dict[str, list[Path]], settings: dict, select: OutputSelector, out_dir: Path) -> None`; `REGISTRY: dict[str, Analysis]`; `register(analysis: Analysis) -> Analysis`; `get_analysis(name: str) -> Optional[Analysis]`; `enabled_analyses(cfg: PipelineConfig) -> list[tuple[Analysis, dict]]`
  - `batch.analyses` re-exports the same names.
  - `batch.validation`: `validate(cfg: PipelineConfig) -> None` (raises `ConfigError` listing every problem; warns for soft problems)

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_validation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking.batch.analyses'`

- [ ] **Step 3: Write `analyses/base.py` and `analyses/__init__.py`**

`src/napari_spotiflow_tracking/batch/analyses/base.py`:

```python
"""Analysis plugin base class and registry.

An analysis declares what it needs and what it can write, then implements up
to three hooks. The runner calls ``run_movie`` once per movie, then
``summarize_group`` per group and ``compare_groups`` when there is more than
one group. Group-level hooks read the per-movie outputs from disk, so they
also work for movies skipped with ``skip_existing``. Hooks create their
``out_dir`` only when they write something.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import PipelineConfig, deep_merge


class Analysis:
    name: str = ""
    requires_tracks: bool = False
    default_measure_on: str = "raw"
    catalogue: tuple = ()

    def default_settings(self) -> dict:
        return {}

    def validate(self, settings: dict) -> list:
        """Problems with ``settings`` (already merged over defaults), as strings."""
        return []

    def run_movie(self, ctx, settings: dict, select) -> dict:
        """Write this movie's outputs; return a flat record for run_summary.csv."""
        raise NotImplementedError

    def summarize_group(self, group: str, movie_dirs: list, settings: dict, select,
                        out_dir: Path) -> None:
        return None

    def compare_groups(self, groups: dict, settings: dict, select, out_dir: Path) -> None:
        return None


REGISTRY: dict = {}


def register(analysis: Analysis) -> Analysis:
    if analysis.name in REGISTRY:
        raise ValueError(f"analysis '{analysis.name}' is already registered")
    REGISTRY[analysis.name] = analysis
    return analysis


def get_analysis(name: str) -> Optional[Analysis]:
    return REGISTRY.get(name)


def settings_for(analysis: Analysis, section) -> dict:
    return deep_merge(analysis.default_settings(), section or {})


def enabled_analyses(cfg: PipelineConfig) -> list:
    """``(analysis, merged settings)`` for every known, enabled section, in config order."""
    enabled = []
    for name, section in cfg.analyses.items():
        analysis = get_analysis(name)
        if analysis is None or not isinstance(section, (dict, type(None))):
            continue
        settings = settings_for(analysis, section)
        if settings.pop("enabled", True):
            enabled.append((analysis, settings))
    return enabled
```

Note: `enabled_analyses` and `get_analysis` read the module global `REGISTRY` at call time, so the test's `monkeypatch.setattr(analyses.base, "REGISTRY", {})` isolates registrations.

`src/napari_spotiflow_tracking/batch/analyses/__init__.py`:

```python
"""Registered batch analyses. Phases 2–4 import their modules here to register them."""

from . import base
from .base import Analysis, enabled_analyses, get_analysis, register, settings_for

__all__ = ["Analysis", "base", "enabled_analyses", "get_analysis", "register", "settings_for"]
```

- [ ] **Step 4: Write `validation.py`**

`src/napari_spotiflow_tracking/batch/validation.py`:

```python
"""Whole-config validation, run before any movie is read."""

from __future__ import annotations

import warnings

from .analyses import base
from .config import FIGURE_FORMATS, MODALITIES, ConfigError, ConfigWarning, PipelineConfig
from .detect import torchimize_available
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
    if cfg.detection.fit.enabled and not torchimize_available():
        errors.append("detection.fit.enabled needs the optional 'torchimize' package "
                      "(pip install torchimize)")
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
        errors.extend(f"analyses.{name}: {problem}" for problem in analysis.validate(settings))
        try:
            OutputSelector(settings, analysis.catalogue, name)
        except ConfigError as exc:
            errors.append(str(exc))

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_validation.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses src/napari_spotiflow_tracking/batch/validation.py src/napari_spotiflow_tracking/_tests/test_batch_validation.py
git commit -m "feat(batch): analysis plugin registry and whole-config validation"
```

---

### Task 7: Runner

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/runner.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_runner.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `write_resolved_config` (Task 1); `theme.style`, `theme.group_colors`, `save_figure`, `save_table`, `OutputSelector` (Task 2); `run_steps` (Task 3); `load_detector`, `detect_stack`, `fit_spots`, `track_spots` (Task 4); `expand_inputs`, `color_pins`, `MovieJob` (Task 5); `MovieContext` (Task 5); `enabled_analyses` (Task 6); `validate` (Task 6).
- Produces (in `batch.runner`):
  - `STATUS_FILE = "movie_status.json"`, `SUMMARY_COLUMNS = ["group", "movie", "path", "status", "n_frames", "n_spots", "n_tracks", "runtime_s", "error"]`
  - `RunResult(rows: list[dict], failed_hooks: list[str])` with property `exit_code -> int`
  - `movie_dir(output_dir: Path, job: MovieJob) -> Path`
  - `run(cfg: PipelineConfig) -> RunResult`

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_runner.py`:

```python
"""End-to-end runner on tiny synthetic movies with LoG detection (no GPU, no model)."""
import json

import numpy as np
import pandas as pd
import pytest
import tifffile

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

    def run_movie(self, ctx, settings, select):
        if "broken" in ctx.job.movie_id:
            raise RuntimeError("boom")
        self.movies.append(ctx.job.key)
        if select.table("count"):
            pd.DataFrame({"n": [len(ctx.spots)]}).to_csv(ctx.analysis_dir(self.name) / "count.csv",
                                                         index=False)
        return {"n_particles": int(ctx.tracks["particle"].nunique())}

    def summarize_group(self, group, movie_dirs, settings, select, out_dir):
        self.groups.append((group, sorted(d.name for d in movie_dirs)))

    def compare_groups(self, groups, settings, select, out_dir):
        self.compared.append(sorted(groups))


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
        "tracking": {"enabled": True, "search_range": 3, "memory": 0, "min_track_length": 3},
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'runner'`

- [ ] **Step 3: Write `runner.py`**

`src/napari_spotiflow_tracking/batch/runner.py`:

```python
"""Run a validated config over every movie, then group and comparison hooks."""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure  # no pyplot: never switch the backend napari uses

from .._preprocessing import _get_device
from . import theme
from .analyses import enabled_analyses
from .config import PipelineConfig, write_resolved_config
from .context import MovieContext
from .detect import detect_stack, fit_spots, load_detector, track_spots
from .inputs import MovieJob, color_pins, expand_inputs
from .outputs import OutputSelector, save_figure, save_table
from .preprocess import run_steps
from .validation import validate

log = logging.getLogger("napari_spotiflow_tracking.batch")

STATUS_FILE = "movie_status.json"
SUMMARY_COLUMNS = ["group", "movie", "path", "status", "n_frames", "n_spots",
                   "n_tracks", "runtime_s", "error"]


@dataclass
class RunResult:
    rows: list = field(default_factory=list)
    failed_hooks: list = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        failed = self.failed_hooks or any(r["status"] == "FAIL" for r in self.rows)
        return 1 if failed else 0


def movie_dir(output_dir: Path, job: MovieJob) -> Path:
    return Path(output_dir) / job.group / job.movie_id


def _resolve_device(requested: str) -> str:
    device = _get_device(use_gpu=requested == "cuda")
    if device != requested:
        log.warning("device '%s' is not available; using '%s'", requested, device)
    return device


def _build_context(job: MovieJob, cfg: PipelineConfig, out_dir: Path, device: str,
                   detector: dict) -> MovieContext:
    raw = tifffile.imread(str(job.path))
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) movie, got shape {raw.shape}")
    pre = run_steps(raw, cfg.preprocessing.steps, device)
    if cfg.outputs.save_preprocessed:
        tifffile.imwrite(str(out_dir / "preprocessed.tif"), pre, bigtiff=True)

    spots_csv, tracks_csv = out_dir / "spots.csv", out_dir / "tracks.csv"
    reusable = spots_csv.exists() and (not cfg.tracking.enabled or tracks_csv.exists())
    mask = None
    if cfg.skip_existing and reusable:
        log.info("%s: reusing spots/tracks on disk", job.key)
        spots = pd.read_csv(spots_csv)
        tracks = pd.read_csv(tracks_csv) if cfg.tracking.enabled else None
    else:
        if "model" not in detector:
            detector["model"] = load_detector(cfg.detection, device)
        spots = detect_stack(pre, cfg.detection, detector["model"])
        if cfg.detection.fit.enabled:
            spots, mask = fit_spots(pre, spots, cfg.detection.fit, device)
            if cfg.outputs.save_masks:
                tifffile.imwrite(str(out_dir / "mask.tif"), mask, bigtiff=True)
        save_table(spots, spots_csv)
        tracks = None
        if cfg.tracking.enabled:
            tracks = track_spots(spots, cfg.tracking)
            save_table(tracks, tracks_csv)
    return MovieContext(job=job, out_dir=out_dir, config=cfg, raw=raw, preprocessed=pre,
                        spots=spots, tracks=tracks, mask=mask)


def _run_movie(job: MovieJob, cfg: PipelineConfig, analyses: list, device: str,
               detector: dict) -> dict:
    out_dir = movie_dir(Path(cfg.output_dir), job)
    status_path = out_dir / STATUS_FILE
    if cfg.skip_existing and status_path.exists():
        row = json.loads(status_path.read_text(encoding="utf-8"))
        row["status"] = "SKIP"
        log.info("%s: already finished, skipped", job.key)
        return row

    start = time.perf_counter()
    row = {"group": job.group, "movie": job.movie_id, "path": str(job.path),
           "status": "OK", "n_frames": 0, "n_spots": 0, "n_tracks": 0,
           "runtime_s": 0.0, "error": ""}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = _build_context(job, cfg, out_dir, device, detector)
        row["n_frames"] = int(ctx.raw.shape[0])
        row["n_spots"] = int(len(ctx.spots))
        if ctx.tracks is not None:
            row["n_tracks"] = int(ctx.tracks["particle"].nunique())
        for analysis, settings in analyses:
            select = OutputSelector(settings, analysis.catalogue, analysis.name)
            record = analysis.run_movie(ctx, settings, select) or {}
            for key, value in record.items():
                row[f"{analysis.name}.{key}"] = value
    except Exception as exc:
        log.error("%s failed:\n%s", job.key, traceback.format_exc())
        row["status"] = "FAIL"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_s"] = round(time.perf_counter() - start, 2)
    if row["status"] == "OK":
        status_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    log.info("%s: %s (%d spots, %d tracks, %.1f s)", job.key, row["status"],
             row["n_spots"], row["n_tracks"], row["runtime_s"])
    return row


def _run_hooks(cfg: PipelineConfig, analyses: list, jobs: list, rows: list) -> list:
    out = Path(cfg.output_dir)
    done = {(r["group"], r["movie"]) for r in rows if r["status"] in ("OK", "SKIP")}
    groups: dict = {}
    for job in jobs:
        if (job.group, job.movie_id) in done:
            groups.setdefault(job.group, []).append(movie_dir(out, job))
    failed = []
    for analysis, settings in analyses:
        select = OutputSelector(settings, analysis.catalogue, analysis.name)
        for group, dirs in groups.items():
            try:
                analysis.summarize_group(group, dirs, settings, select,
                                         out / group / "_summary" / analysis.name)
            except Exception:
                label = f"{analysis.name}: group summary for '{group}'"
                log.error("%s failed:\n%s", label, traceback.format_exc())
                failed.append(label)
        if len(groups) > 1:
            try:
                analysis.compare_groups(groups, settings, select,
                                        out / "_comparison" / analysis.name)
            except Exception:
                label = f"{analysis.name}: group comparison"
                log.error("%s failed:\n%s", label, traceback.format_exc())
                failed.append(label)
    return failed


def _write_summary(cfg: PipelineConfig, rows: list) -> None:
    out = Path(cfg.output_dir)
    table = pd.DataFrame(rows)
    extra = [c for c in table.columns if c not in SUMMARY_COLUMNS]
    save_table(table.reindex(columns=SUMMARY_COLUMNS + extra), out / "run_summary.csv")

    labels = list(dict.fromkeys(r["group"] for r in rows))
    colors = theme.group_colors(labels, color_pins(cfg))
    metrics = ["n_spots", "n_tracks"] if cfg.tracking.enabled else ["n_spots"]
    names = [f"{r['group']}/{r['movie']}" for r in rows]
    with theme.style():
        fig = Figure(figsize=(max(6.0, 0.6 * len(rows) + 2), 3.5 * len(metrics)))
        axes = fig.subplots(len(metrics), 1, squeeze=False, sharex=True)
        for ax, metric in zip(axes[:, 0], metrics):
            ax.bar(np.arange(len(rows)), [r[metric] for r in rows],
                   color=[colors[r["group"]] for r in rows])
            ax.set_ylabel("Spots" if metric == "n_spots" else "Tracks")
        axes[-1, 0].set_xticks(np.arange(len(rows)))
        axes[-1, 0].set_xticklabels(names, rotation=45, ha="right")
        fig.tight_layout()
        save_figure(fig, out / "run_overview", cfg.outputs.formats, cfg.outputs.dpi)


def run(cfg: PipelineConfig) -> RunResult:
    validate(cfg)
    jobs = expand_inputs(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        write_resolved_config(cfg, out / "resolved_config.yaml")
        analyses = enabled_analyses(cfg)
        device = _resolve_device(cfg.device)
        log.info("%d movies in %d groups; analyses: %s", len(jobs),
                 len({j.group for j in jobs}), ", ".join(a.name for a, _ in analyses) or "none")
        detector: dict = {}
        rows = [_run_movie(job, cfg, analyses, device, detector) for job in jobs]
        failed = _run_hooks(cfg, analyses, jobs, rows)
        _write_summary(cfg, rows)
        return RunResult(rows=rows, failed_hooks=failed)
    finally:
        log.removeHandler(handler)
        handler.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_runner.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/runner.py src/napari_spotiflow_tracking/_tests/test_batch_runner.py
git commit -m "feat(batch): runner with failure isolation, skip_existing and run summary"
```

---

### Task 8: Command-line interface

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/cli.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_cli.py`

**Interfaces:**
- Consumes: `load_config`, `config_to_dict`, `ConfigError`, `ConfigWarning`, `MODALITIES` (Task 1); `base.REGISTRY`, `get_analysis` (Task 6); `validate` (Task 6); `expand_inputs` (Task 5); `run` (Task 7).
- Produces (in `batch.cli`): `main(argv: Optional[Sequence[str]] = None) -> int`; `starter_config(modality: str) -> str`; `format_catalogues(name: Optional[str] = None) -> str`.

- [ ] **Step 1: Write the failing tests**

`src/napari_spotiflow_tracking/_tests/test_batch_cli.py`:

```python
"""spotiflow-batch subcommands."""
import warnings

import numpy as np
import pytest
import tifffile
import yaml

from napari_spotiflow_tracking.batch import analyses, cli
from napari_spotiflow_tracking.batch.analyses import Analysis, register
from napari_spotiflow_tracking.batch.config import ConfigWarning, load_config
from napari_spotiflow_tracking.batch.outputs import OutputSpec
from napari_spotiflow_tracking.batch.validation import validate


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(analyses.base, "REGISTRY", {})


@pytest.mark.parametrize("modality", ["tracking", "papa", "bleaching"])
def test_init_config_is_valid_and_warning_free(modality, capsys):
    assert cli.main(["init-config", modality]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"# spotiflow-batch config — {modality} preset")
    data = yaml.safe_load(text)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        cfg = load_config(data=data)
    assert cfg.modality == modality
    validate(cfg)


def test_list_outputs(empty_registry, capsys):
    assert cli.main(["list-outputs"]) == 0
    assert "No analyses are registered yet" in capsys.readouterr().out

    class Demo(Analysis):
        name = "demo"
        catalogue = (OutputSpec("plot", "movie", "figure", "A plot of things"),)

    register(Demo())
    assert cli.main(["list-outputs", "demo"]) == 0
    out = capsys.readouterr().out
    assert "demo" in out and "plot" in out and "A plot of things" in out
    assert cli.main(["list-outputs", "nope"]) == 2


def test_validate_reports_movies(tmp_path, empty_registry, capsys):
    movie = tmp_path / "g" / "m.tif"
    movie.parent.mkdir()
    tifffile.imwrite(str(movie), np.zeros((2, 8, 8), np.uint16))
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": [str(movie.parent)],
                                      "output_dir": str(tmp_path / "out")}))
    assert cli.main(["validate", str(config)]) == 0
    assert "config OK: 1 movie(s) in 1 group(s)" in capsys.readouterr().out


def test_config_error_exit_code(tmp_path, empty_registry, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": [], "output_dir": ""}))
    assert cli.main(["validate", str(config)]) == 2
    assert "ERROR: invalid config" in capsys.readouterr().err
    assert cli.main(["run", str(config), "--set", "broken"]) == 2


def test_run_uses_set_overrides(tmp_path, empty_registry, monkeypatch):
    seen = {}

    class Result:
        exit_code = 0

    def fake_run(cfg):
        seen["cfg"] = cfg
        return Result()

    monkeypatch.setattr(cli, "run", fake_run)
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"inputs": ["x.tif"], "output_dir": "out"}))
    assert cli.main(["run", str(config), "--set", "tracking.memory=9",
                     "--set", "device=cpu"]) == 0
    assert seen["cfg"].tracking.memory == 9 and seen["cfg"].device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Write `cli.py`**

`src/napari_spotiflow_tracking/batch/cli.py`:

```python
"""``spotiflow-batch``: run, validate, list-outputs, init-config."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .analyses import base
from .config import MODALITIES, ConfigError, config_to_dict, load_config
from .inputs import expand_inputs
from .runner import run
from .validation import validate

SECTION_HELP = {
    "skip_existing": "Skip movies that already finished in output_dir (resume an interrupted run).",
    "device": "cuda or cpu; cuda falls back to cpu when no GPU is available.",
    "calibration": "Set both to report lengths in µm and times in s.",
    "preprocessing": "Steps run top to bottom. Available: crop, add, subtract, "
                     "walking_average, gaussian_blur, background, n2v.",
    "detection": "method: spotiflow or log. fit.enabled adds a 2D Gaussian fit and "
                 "spot masks (needs torchimize).",
    "tracking": "trackpy linking; tracks shorter than min_track_length are dropped.",
    "measure_on": "raw or preprocessed: the movie analyses read intensities from, "
                  "unless an analysis overrides it.",
    "outputs": "formats: any of png, pdf, svg.",
    "analyses": "Enabled analyses and their settings; `spotiflow-batch list-outputs` "
                "lists figure and table names.",
}


def starter_config(modality: str) -> str:
    resolved = config_to_dict(load_config(data={"modality": modality}))
    lines = [
        f"# spotiflow-batch config — {modality} preset",
        "# Check: spotiflow-batch validate this.yaml   Run: spotiflow-batch run this.yaml",
        "",
        f"modality: {modality}",
        "",
        "# One entry per group: a folder of movies or a single movie.",
        "inputs:",
        "  - {path: 'D:/path/to/movies', label: Control, pattern: '*.tif', recursive: false}",
        "output_dir: 'D:/path/to/results'",
        "",
    ]
    for key, value in resolved.items():
        if key in ("modality", "inputs", "output_dir"):
            continue
        if key in SECTION_HELP:
            lines.append(f"# {SECTION_HELP[key]}")
        lines.append(yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True).rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def format_catalogues(name: Optional[str] = None) -> str:
    if name is not None:
        analysis = base.get_analysis(name)
        if analysis is None:
            raise ConfigError(f"unknown analysis '{name}'; known: "
                              f"{', '.join(sorted(base.REGISTRY)) or 'none'}")
        chosen = [analysis]
    else:
        chosen = [base.REGISTRY[key] for key in sorted(base.REGISTRY)]
    if not chosen:
        return "No analyses are registered yet."
    blocks = []
    for analysis in chosen:
        lines = [f"{analysis.name}:"]
        for spec in analysis.catalogue:
            lines.append(f"  {spec.name:<28} {spec.kind:<6} {spec.level:<10} {spec.description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spotiflow-batch",
        description="Config-driven batch detection, tracking and analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "run a config"), ("validate", "check a config and its inputs")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("config", type=Path)
        cmd.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="KEY=VALUE",
                         help="override a config value, e.g. --set tracking.memory=5")
    cmd = sub.add_parser("list-outputs", help="list the figures and tables analyses can write")
    cmd.add_argument("analysis", nargs="?")
    cmd = sub.add_parser("init-config", help="print a commented starter config")
    cmd.add_argument("modality", choices=MODALITIES)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.formatwarning = lambda message, *a, **k: f"WARNING: {message}\n"
    try:
        if args.command == "init-config":
            sys.stdout.write(starter_config(args.modality))
            return 0
        if args.command == "list-outputs":
            print(format_catalogues(args.analysis))
            return 0
        cfg = load_config(args.config, args.overrides)
        if args.command == "validate":
            validate(cfg)
            jobs = expand_inputs(cfg)
            groups = {job.group for job in jobs}
            print(f"config OK: {len(jobs)} movie(s) in {len(groups)} group(s)")
            return 0
        return run(cfg).exit_code
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_cli.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Check the installed console command**

Run: `.venv/Scripts/python.exe -m pip install -e . -q` then `.venv/Scripts/spotiflow-batch.exe init-config bleaching`
Expected: prints the commented bleaching starter config, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/cli.py src/napari_spotiflow_tracking/_tests/test_batch_cli.py
git commit -m "feat(batch): spotiflow-batch run/validate/list-outputs/init-config"
```

---

### Task 9: Full suite, real-data smoke run, spec and CLAUDE.md update

**Files:**
- Modify: `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new in code.

- [ ] **Step 1: Run the whole test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests PASS (pre-existing and new); no new failures.

- [ ] **Step 2: Smoke-run on real data (manual check)**

Create `C:/Users/CELLPH~1/AppData/Local/Temp/claude/d--Code-Lab-spotiflow-tracking/2bfe210b-ac1c-4253-b4b7-8c3a300ef68d/scratchpad/smoke.yaml`:

```yaml
modality: bleaching
inputs:
  - {path: "D:/Data/Sudipta/Arpan/GoodDataset/field2.tif", label: field2}
output_dir: "C:/Users/CELLPH~1/AppData/Local/Temp/claude/d--Code-Lab-spotiflow-tracking/2bfe210b-ac1c-4253-b4b7-8c3a300ef68d/scratchpad/batch_smoke"
```

Run: `.venv/Scripts/spotiflow-batch.exe run <scratchpad>/smoke.yaml`
Expected: exit code 0; `batch_smoke/field2/field2/tracks.csv` exists. Its particle count should be close to the 1896 tracks found for `field2.tif` by `scripts/bleaching_trace_analysis.py` earlier today (same preprocessing, detector and linking; the preset's `min_track_length: 10` drops short tracks, so expect fewer). Record the number in the task report.

- [ ] **Step 3: Update the spec with the clarifications**

In `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`:

1. In the config schema block replace `- subtract: {value: mean}          # add | subtract | multiply | divide; number or "mean"` with `- subtract: {value: mean}          # add | subtract; number or "mean"`, and replace `- background: {method: high_pass, sigma: 10}   # or {method: erosion, radius: 5}` with `- background: {method: high_pass, sigma: 10}   # or {method: erosion, disk_size: 5}`, and `- n2v: {epochs: 10, patch: 64}` with `- n2v: {n_epochs: 10, patch_size: 64}`.
2. Replace `fit: {enabled: false, backend: auto, window: 2, max_radius: 2.0}` with `fit: {enabled: false, window: 2, max_radius: 2.0}   # needs torchimize`.
3. Under "### Analysis protocol", replace `config_type: type                          # dataclass for its config section` with `def default_settings(self) -> dict: ...        # merged under the user's section; "enabled: false" disables`.
4. In "## Theme", replace the semantic-palette bullet with: `Semantic palettes — step counts, motility classes and PAPA groups share #56b4e9 / #eda100 / #4a3aa7 (+ #7f7f7f unclassified); tab:orange/tab:green are not used because they collapse under protanopia.`
5. Add to "## Error handling": `- skip_existing: a movie with movie_status.json is skipped; otherwise existing spots.csv/tracks.csv are reused and analyses re-run.`

- [ ] **Step 4: Update `CLAUDE.md`**

Under `### Bulk / headless scripts`, before the existing code block, add:

````markdown
The config-driven pipeline `spotiflow-batch` (package `batch/`) is the unified
replacement being built in phases (spec:
`docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`):

```bash
spotiflow-batch init-config bleaching > my.yaml   # commented starter config
spotiflow-batch validate my.yaml
spotiflow-batch run my.yaml --set tracking.memory=5
```
````

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md
git add CLAUDE.md
git commit -m "docs(batch): record phase-1 clarifications and CLI usage"
```

---

## Self-review notes

- Spec coverage for phase 1: location/console command (T1, T8), YAML + `--set` + precedence (T1), presets (T1), ordered preprocessing incl. crop/add/subtract/walking average/blur/background/N2V (T3), both detectors + optional fit/masks (T4), optional tracking (T4, T7), `measure_on` (T1 field, T5 `MovieContext.image`), inputs as files/folders/groups/patterns/recursion/colours (T5), output catalogues + selection + unknown-name warnings (T2, T6), lock-safe saving (T2), theme + CVD check (T2), validation incl. dependency errors and unknown-key warnings (T1, T6), per-movie layout, `run_summary.csv`, `run_overview`, `run.log`, `resolved_config.yaml` (T7), failure isolation + exit codes (T7, T8), `skip_existing` (T7), group/comparison hooks (T6, T7), `validate`/`list-outputs`/`init-config` (T8). Analyses, group summaries content and docs page are phases 2–5.
- Names used across tasks: `OutputSelector(settings, catalogue, analysis)` (T2 → T6, T7), `settings_for` (T6 → validation), `movie_dir` (T7), `MovieJob.key` (T5 → T7 logs/tests), `enabled_analyses` returns `(Analysis, dict)` (T6 → T7).
