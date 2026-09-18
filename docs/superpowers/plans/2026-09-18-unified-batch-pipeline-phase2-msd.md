# Unified Batch Pipeline — Phase 2 (MSD + group summaries + comparison) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the MSD / motility analysis of `scripts/bulk_track_analysis.py` into the batch pipeline as the registered `msd` analysis — per-movie tables and figures, per-group summaries, and cross-group comparison (the old `--compare` mode) — drawn through the shared theme and verified against the old script.

**Architecture:** A new subpackage `napari_spotiflow_tracking.batch.analyses.msd`. `compute.py` holds the calibration-aware fitting and metrics (no plotting); `figures_movie.py`, `figures_group.py` and `figures_compare.py` hold the plots for the three levels; `analysis.py` binds them to the `Analysis` hooks and the output catalogue. Every figure is ported from the named function in `scripts/bulk_track_analysis.py` with the same content and layout, but drawn via `theme.style()`, the shared palettes and `matplotlib.figure.Figure`, and saved through `outputs.save_figure`. The old script is not modified.

**Tech Stack:** Python ≥ 3.9, numpy, pandas, scipy, matplotlib (Figure API only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md` (build order item 2)

## Global Constraints

- Code runs on Python ≥ 3.9: every new module starts with `from __future__ import annotations`; no `match`; `typing.Optional/List/Dict` in dataclass fields.
- No Qt imports anywhere under `batch/`. Heavy optional imports stay lazy, inside functions.
- **Never import `matplotlib.pyplot` and never call `matplotlib.use`** — build figures with `matplotlib.figure.Figure`, draw inside `with theme.style():`, save with `outputs.save_figure(fig, stem, cfg.outputs.formats, cfg.outputs.dpi)`.
- The existing scripts (`scripts/*.py`) are not modified.
- Unknown config keys and unknown output names → `ConfigWarning`, never an error.
- Exit codes: `0` success, `1` any movie or hook failed, `2` config error.
- Tests live flat in `src/napari_spotiflow_tracking/_tests/` as `test_batch_msd_*.py`, run with `.venv/Scripts/python.exe -m pytest` (always the workspace `.venv`), need no GPU and no real data. The one exception is the opt-in regression test of Task 9 (`@pytest.mark.slow`).
- Commit only the files a task names (`git add <paths>`); the working tree has unrelated uncommitted changes (`scripts/*.py`, untracked scratch files) that must stay out of every commit. Never `git add -A`/`git add .`, `git stash`, `git reset`, `git checkout -- <file>`.
- Work on branch `feat/batch-msd` (already checked out, based on `main` at the phase-1 merge).

## Porting contract (applies to every figure and metric task)

Each ported unit names its source as `bulk:<function>` (in `scripts/bulk_track_analysis.py`). Port rules:

1. **Same content and layout** — same series, axes, annotations, bins, limits, legends and titles as the source function. Read the source before writing; keep its comments where they explain a choice.
2. **Colours come from `batch.theme`**, never from the old `CLASS_COLORS`/`tab:` literals: motility classes → `theme.MOTILITY_COLORS[cls]` (keys `constrained`, `diffusive`, `directed`, `unclassified`), groups → `theme.group_colors(...)`, raw-data traces → `theme.RAW_TRACE`, a single-series neutral → `theme.NEUTRAL`. The old `LAG_RAMP` sequential ramp moves to `msd/figures_movie.py` as a module constant (it is one hue, light→dark, and stays as it is).
3. **Style** — wrap drawing in `with theme.style():`. Do not set font sizes, `axes.linewidth` or spine visibility per figure; the style provides them. Figure sizes stay as in the source (`FIG_H = 5.0`, `FIG_W = 7.0`, `FIG_W_PANEL = 4.5` — copy these constants into `msd/figures_movie.py`).
4. **Axis labels** come from `theme.axis_label(kind, cfg.calibration)` where a kind exists (Task 3 adds the missing kinds); anything else keeps the source's literal label.
5. **Saving** — never `fig.savefig`; call `outputs.save_figure(fig, out_dir / "<catalogue name>", formats, dpi)` which appends the extensions. Tables are written with `outputs.save_table(df, path)`.
6. **No `plt`** — replace `plt.subplots(...)` with `fig = Figure(figsize=...)` + `fig.subplots(...)`, and drop `plt.close(fig)` (there is no global figure registry to leak into).
7. **Guard empty input** — every figure function returns without writing when its input frame is empty, so a movie with no qualifying tracks produces no file and no exception.

---

## File Structure

```
src/napari_spotiflow_tracking/batch/
  theme.py                                   modify: more axis_label kinds (Task 3)
  analyses/__init__.py                       modify: import msd so it registers (Task 2)
  analyses/msd/__init__.py                   registers MsdAnalysis
  analyses/msd/units.py                      Units: calibration + sigma_nm → scales and unit strings
  analyses/msd/compute.py                    MSD fits, ensemble, class counts, directed metrics
  analyses/msd/analysis.py                   MsdAnalysis: catalogue, settings, the three hooks
  analyses/msd/figures_movie.py              per-movie figures
  analyses/msd/figures_group.py              per-group summary figures
  analyses/msd/figures_compare.py            cross-group comparison figures
  presets/tracking.yaml                      modify: enable msd (Task 2)
src/napari_spotiflow_tracking/_tests/
  test_batch_msd_compute.py                  Tasks 1
  test_batch_msd_analysis.py                 Tasks 2, 4, 5, 6
  test_batch_msd_group.py                    Tasks 7, 8
  test_batch_msd_regression.py               Task 9 (slow)
  test_batch_theme_outputs.py                modify: new axis kinds (Task 3)
examples/batch/two_conditions.yaml           modify: msd settings (Task 9)
README.md                                    modify: analyses section (Task 9)
```

---

### Task 1: Units and computation

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/__init__.py` (empty for now, one docstring line)
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/units.py`
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/compute.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_compute.py`

**Interfaces:**
- Consumes: `batch.config.Calibration` (fields `pixel_size_nm`, `frame_interval_s`, property `is_physical`); `_track_properties.compute_track_properties`, `msd_stats`, `msd_basic_fit`, `classify_motility`, `MOTILITY_CLASSES`, `add_physical_columns`.
- Produces:
  - `units.Units(calibration, sigma_nm)` — frozen dataclass with properties `is_physical`, `length_scale` (µm per px, else 1.0), `delta` (s per frame, else 1.0), `sigma` (µm, else 0.0), `length_unit`, `time_unit`, `msd_unit`, `d_unit`, and `describe() -> str`.
  - `compute.MSD_CURVE_COLS = ["track_id", "lag", "msd", "sem", "n_pairs"]`, `compute.MSD_FIT_COLS = ["track_id", "length", "n_lags", "D", "alpha", "motility_class"]`
  - `compute.compute_msd_analysis(tracked, units, max_lag, min_length) -> tuple[pd.DataFrame, pd.DataFrame]` (curves, fits)
  - `compute.ensemble_msd(curves) -> pd.DataFrame`
  - `compute.class_counts(fits) -> dict`
  - `compute.directed_metrics(props, units) -> pd.DataFrame`
  - `compute.track_table(tracked, image, fits, units) -> pd.DataFrame` — `compute_track_properties` + merged `D`/`alpha`/`motility_class` (missing → `"unclassified"`) + physical columns

Port sources: `bulk:compute_msd_analysis` (L247), `bulk:ensemble_msd` (L319), `bulk:class_counts` (L684), `bulk:_directed_metrics` (L1259), the `Calibration` dataclass (L173-220), and the `track_props` block of `bulk:analyze_folder` (L1949-1975). `Units` replaces the old `Calibration` because phase 1's `Calibration` holds only the two user-facing values; `sigma_nm` is an `msd` setting.

- [ ] **Step 1: Write the failing tests**

```python
"""MSD computation: fits, ensemble, class counts, directed metrics, units."""
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking.batch.analyses.msd import compute
from napari_spotiflow_tracking.batch.analyses.msd.units import Units
from napari_spotiflow_tracking.batch.config import Calibration


def brownian_tracks(n_tracks=6, n_frames=60, step_px=1.0, seed=0):
    """Tracks whose per-frame steps are N(0, step_px) in each axis → alpha ≈ 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        pos = np.cumsum(rng.normal(0, step_px, size=(n_frames, 2)), axis=0)
        for frame, (y, x) in enumerate(pos):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


def directed_tracks(n_tracks=4, n_frames=60, speed=2.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        drift = np.arange(n_frames)[:, None] * np.array([speed, speed])
        jitter = rng.normal(0, 0.05, size=(n_frames, 2))
        for frame, (y, x) in enumerate(drift + jitter):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


PX = Units(Calibration(), sigma_nm=35.0)
UM = Units(Calibration(pixel_size_nm=100, frame_interval_s=0.5), sigma_nm=35.0)


def test_units_pixel_mode():
    assert not PX.is_physical
    assert (PX.length_scale, PX.delta, PX.sigma) == (1.0, 1.0, 0.0)
    assert PX.length_unit == "px" and PX.time_unit == "frames"
    assert PX.msd_unit == "px²" and PX.d_unit == "px²/frame^α"
    assert "px" in PX.describe()


def test_units_physical_mode():
    assert UM.is_physical
    assert UM.length_scale == pytest.approx(0.1)      # 100 nm → µm
    assert UM.delta == pytest.approx(0.5)
    assert UM.sigma == pytest.approx(0.035)
    assert UM.length_unit == "µm" and UM.time_unit == "s"
    assert UM.msd_unit == "µm²" and UM.d_unit == "µm²/s^α"


def test_curves_and_fits_shape_and_columns():
    curves, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                                min_length=5)
    assert list(curves.columns) == compute.MSD_CURVE_COLS
    assert list(fits.columns) == compute.MSD_FIT_COLS
    assert set(fits["track_id"]) == set(range(6))
    assert curves["lag"].max() <= 10
    assert (curves["n_pairs"] > 0).all()


def test_brownian_tracks_are_classified_diffusive():
    _, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=15,
                                           min_length=5)
    assert fits["alpha"].median() == pytest.approx(1.0, abs=0.25)
    assert (fits["motility_class"] == "diffusive").mean() >= 0.8


def test_directed_tracks_are_classified_directed():
    _, fits = compute.compute_msd_analysis(directed_tracks(), PX, max_lag=15,
                                           min_length=5)
    assert fits["alpha"].median() > 1.5
    assert (fits["motility_class"] == "directed").all()


def test_short_tracks_are_dropped():
    tracks = brownian_tracks(n_tracks=2, n_frames=60)
    short = pd.DataFrame({"frame": [0, 1, 2], "y": [0.0, 1.0, 2.0],
                          "x": [0.0, 0.0, 1.0], "particle": [99, 99, 99]})
    _, fits = compute.compute_msd_analysis(pd.concat([tracks, short]), PX,
                                           max_lag=10, min_length=5)
    assert 99 not in set(fits["track_id"])


def test_empty_input_gives_empty_frames_with_columns():
    empty = pd.DataFrame({"frame": [], "y": [], "x": [], "particle": []})
    curves, fits = compute.compute_msd_analysis(empty, PX, max_lag=10, min_length=5)
    assert len(curves) == 0 and list(curves.columns) == compute.MSD_CURVE_COLS
    assert len(fits) == 0 and list(fits.columns) == compute.MSD_FIT_COLS


def test_physical_units_scale_d_by_pixel_and_frame():
    tracks = brownian_tracks(seed=3)
    _, px_fits = compute.compute_msd_analysis(tracks, PX, max_lag=10, min_length=5)
    _, um_fits = compute.compute_msd_analysis(tracks, UM, max_lag=10, min_length=5)
    # D scales as length²/time: (0.1 µm/px)² / (0.5 s/frame) = 0.02
    ratio = (um_fits["D"] / px_fits["D"]).median()
    assert ratio == pytest.approx(0.02, rel=0.25)
    assert um_fits["alpha"].median() == pytest.approx(px_fits["alpha"].median(), abs=0.1)


def test_ensemble_msd_averages_curves():
    curves, _ = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                             min_length=5)
    ens = compute.ensemble_msd(curves)
    assert {"lag", "msd", "sem", "n_tracks"} <= set(ens.columns)
    assert len(ens) == curves["lag"].nunique()
    assert ens["msd"].is_monotonic_increasing


def test_class_counts_covers_every_class():
    _, fits = compute.compute_msd_analysis(brownian_tracks(), PX, max_lag=10,
                                           min_length=5)
    counts = compute.class_counts(fits)
    # "unclassified" appears only when some track has one (source behaviour).
    assert {"constrained", "diffusive", "directed"} <= set(counts)
    assert set(counts) <= {"constrained", "diffusive", "directed", "unclassified"}
    assert sum(counts.values()) == len(fits)
    assert compute.class_counts(fits.iloc[:0]) == {"constrained": 0, "diffusive": 0,
                                                   "directed": 0}


def test_directed_metrics_only_directed_tracks():
    tracked = pd.concat([brownian_tracks(), directed_tracks()])
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=15, min_length=5)
    props = compute.track_table(tracked, None, fits, PX)
    directed = compute.directed_metrics(props, PX)
    assert len(directed) == (props["motility_class"] == "directed").sum()
    assert {"net_displacement", "mean_speed"} <= set(directed.columns)


def test_track_table_merges_fits_and_marks_unclassified():
    tracked = brownian_tracks(n_tracks=3, n_frames=60)
    short = pd.DataFrame({"frame": [0, 1], "y": [0.0, 1.0], "x": [0.0, 1.0],
                          "particle": [77, 77]})
    tracked = pd.concat([tracked, short])
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=10, min_length=5)
    table = compute.track_table(tracked, None, fits, PX)
    assert {"track_id", "length", "mean_speed", "D", "alpha", "motility_class"} <= set(table.columns)
    assert table.loc[table.track_id == 77, "motility_class"].item() == "unclassified"


def test_track_table_adds_physical_columns_only_when_calibrated():
    tracked = brownian_tracks(n_tracks=2, n_frames=40)
    _, fits = compute.compute_msd_analysis(tracked, UM, max_lag=10, min_length=5)
    table = compute.track_table(tracked, None, fits, UM)
    assert {"duration_s", "mean_speed_um_s"} <= set(table.columns)
    assert "length" in table.columns          # pixel columns stay beside them
    px_table = compute.track_table(tracked, None, fits, PX)
    assert "duration_s" not in px_table.columns


def test_track_table_samples_intensity_when_image_given():
    tracked = brownian_tracks(n_tracks=2, n_frames=20, step_px=0.2)
    tracked[["y", "x"]] = tracked[["y", "x"]].abs() + 5
    image = np.tile(np.arange(32, dtype=np.uint16), (20, 32, 1))
    _, fits = compute.compute_msd_analysis(tracked, PX, max_lag=5, min_length=5)
    table = compute.track_table(tracked, image, fits, PX)
    assert "mean_intensity" in table.columns and table["mean_intensity"].notna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_compute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking.batch.analyses.msd'`

- [ ] **Step 3: Write `units.py`**

Port `bulk:Calibration` (L173-220), replacing its `pixel_size_nm`/`frame_interval_s` fields with a `Calibration` reference:

```python
"""Scales and unit strings for the MSD analysis.

Phase 1's ``Calibration`` carries the two user-facing values; ``sigma_nm`` is an
``msd`` setting, and the derived scales live here so the figures and the fits
share one source. Ported from ``scripts/bulk_track_analysis.py`` (Calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import Calibration


@dataclass(frozen=True)
class Units:
    calibration: Calibration
    sigma_nm: float = 35.0

    @property
    def is_physical(self) -> bool:
        return self.calibration.is_physical
    ...
```

Fill in `length_scale` (`pixel_size_nm / 1000` else `1.0`), `delta` (`frame_interval_s` else `1.0`), `sigma` (`sigma_nm / 1000` else `0.0`), the four unit strings (`µm`/`px`, `s`/`frames`, `µm²`/`px²`, `µm²/s^α`/`px²/frame^α`) and `describe()` exactly as the source does.

- [ ] **Step 4: Write `compute.py`**

Port `bulk:compute_msd_analysis`, `bulk:ensemble_msd`, `bulk:class_counts`, `bulk:_directed_metrics` (renamed `directed_metrics`) and the `track_props` block of `bulk:analyze_folder` as `track_table(tracked, image, fits, units)`. Replace every `cal.` attribute access with the identically named `units.` property; keep the underlying calls to `_track_properties.msd_stats`, `msd_basic_fit`, `classify_motility`, `compute_track_properties` and `add_physical_columns` unchanged. No plotting, no file writing, no printing in this module.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_compute.py -v`
Expected: all 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/__init__.py src/napari_spotiflow_tracking/batch/analyses/msd/units.py src/napari_spotiflow_tracking/batch/analyses/msd/compute.py src/napari_spotiflow_tracking/_tests/test_batch_msd_compute.py
git commit -m "feat(batch-msd): calibration-aware MSD fits, ensemble and metrics"
```

---

### Task 2: The `msd` analysis — registration, catalogue, tables

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/__init__.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/__init__.py`
- Modify: `src/napari_spotiflow_tracking/batch/presets/tracking.yaml`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py`

**Interfaces:**
- Consumes: Task 1's `compute` and `Units`; `analyses.base.Analysis`, `register`, `measure_on_for`; `outputs.OutputSpec`, `save_table`; `context.MovieContext`.
- Produces:
  - `analysis.MsdAnalysis` with `name = "msd"`, `requires_tracks = True`, `default_measure_on = "preprocessed"`, `catalogue` (Task 2 adds the tables and the Task 4/5/6 figure specs are added by those tasks), `default_settings()`, `validate()`, `run_movie()`.
  - `msd/__init__.py` exports `MsdAnalysis` and registers an instance.
  - Table names written by `run_movie`: `msd_curves` → `msd_curves.csv`, `msd_fits` → `msd_fits.csv`, `track_analysis` → `track_analysis.csv`, all inside `<movie>/msd/`.
  - `run_movie` returns `{"n_fitted": int, "constrained": int, "diffusive": int, "directed": int}`.

Default settings (from the spec's config schema and the old CLI defaults):

```python
{"max_lag": 25, "sigma_nm": 35.0, "min_length": 5, "max_tracks_plot": 200,
 "figures": "all", "tables": "all"}
```

`validate` returns problems for: `max_lag` < 1, `min_length` < 3, `max_tracks_plot` < 1, `sigma_nm` < 0.

- [ ] **Step 1: Write the failing tests**

```python
"""The msd analysis: registration, settings, tables and figure selection."""
import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.analyses import base, get_analysis
from napari_spotiflow_tracking.batch.analyses.msd import MsdAnalysis
from napari_spotiflow_tracking.batch.config import ConfigError, load_config
from napari_spotiflow_tracking.batch.validation import validate


def synthetic_movie(path, n_frames=40, size=64, n_spots=6, seed=0):
    """Bright Gaussian spots doing a slow random walk — LoG finds them every frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    starts = rng.uniform(12, size - 12, size=(n_spots, 2))
    stack = np.full((n_frames, size, size), 100.0)
    pos = starts.copy()
    for t in range(n_frames):
        pos = pos + rng.normal(0, 0.4, size=pos.shape)
        for cy, cx in pos:
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


def msd_config(tmp_path, movie_dir, **analysis_settings):
    settings = {"max_lag": 10, "min_length": 5}
    settings.update(analysis_settings)
    return load_config(data={
        "inputs": [str(movie_dir)],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"msd": settings},
    })


def test_msd_is_registered():
    analysis = get_analysis("msd")
    assert isinstance(analysis, MsdAnalysis)
    assert analysis.requires_tracks is True


def test_tracking_preset_enables_msd():
    cfg = load_config(data={"modality": "tracking", "inputs": ["x.tif"],
                            "output_dir": "out"})
    assert "msd" in cfg.analyses
    validate(cfg)


def test_msd_requires_tracking():
    cfg = load_config(data={"inputs": ["x.tif"], "output_dir": "out",
                            "tracking": {"enabled": False},
                            "analyses": {"msd": {}}})
    with pytest.raises(ConfigError, match="analyses.msd needs tracking"):
        validate(cfg)


@pytest.mark.parametrize("bad, message", [
    ({"max_lag": 0}, "max_lag"),
    ({"min_length": 2}, "min_length"),
    ({"max_tracks_plot": 0}, "max_tracks_plot"),
    ({"sigma_nm": -1}, "sigma_nm"),
])
def test_invalid_settings(bad, message):
    cfg = load_config(data={"inputs": ["x.tif"], "output_dir": "out",
                            "analyses": {"msd": bad}})
    with pytest.raises(ConfigError, match=message):
        validate(cfg)


def test_run_writes_tables_and_summary_record(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    result = runner.run(msd_config(tmp_path, tmp_path / "g"))
    assert result.exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    curves = pd.read_csv(msd_dir / "msd_curves.csv")
    fits = pd.read_csv(msd_dir / "msd_fits.csv")
    table = pd.read_csv(msd_dir / "track_analysis.csv")
    assert list(curves.columns) == ["track_id", "lag", "msd", "sem", "n_pairs"]
    assert list(fits.columns) == ["track_id", "length", "n_lags", "D", "alpha",
                                  "motility_class"]
    assert len(fits) > 0 and curves["lag"].max() <= 10
    assert {"D", "alpha", "motility_class"} <= set(table.columns)
    row = result.rows[0]
    assert row["msd.n_fitted"] == len(fits)
    assert row["msd.diffusive"] + row["msd.constrained"] + row["msd.directed"] == len(fits)


def test_tables_none_writes_nothing(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", tables="none", figures="none")
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert not msd_dir.exists() or not list(msd_dir.glob("*.csv"))


def test_selected_table_only(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", tables=["msd_fits"], figures="none")
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "msd_fits.csv").exists()
    assert not (msd_dir / "msd_curves.csv").exists()


def test_movie_without_qualifying_tracks_still_succeeds(tmp_path):
    path = tmp_path / "g" / "empty.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), np.full((8, 32, 32), 100, np.uint16))
    result = runner.run(msd_config(tmp_path, tmp_path / "g"))
    assert result.exit_code == 0
    assert result.rows[0]["status"] == "OK"
    assert result.rows[0]["msd.n_fitted"] == 0


def test_list_outputs_shows_msd(capsys):
    from napari_spotiflow_tracking.batch import cli
    assert cli.main(["list-outputs", "msd"]) == 0
    out = capsys.readouterr().out
    assert "msd_fits" in out and "msd_curves" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -v`
Expected: FAIL — `ImportError: cannot import name 'MsdAnalysis'`

- [ ] **Step 3: Write `analysis.py`**

```python
"""The msd analysis: per-movie tables and figures, group summaries, comparison.

Ported from scripts/bulk_track_analysis.py: run_movie from analyze_folder,
summarize_group from write_run_summary, compare_groups from
write_group_comparison.
"""

from __future__ import annotations

from pathlib import Path

from ...config import PipelineConfig
from ...outputs import OutputSpec, save_table
from ..base import Analysis, measure_on_for, register
from . import compute
from .units import Units

CATALOGUE = (
    OutputSpec("msd_curves", "movie", "table", "Per-track MSD per lag, with SEM and pair count"),
    OutputSpec("msd_fits", "movie", "table", "D, alpha and motility class per track"),
    OutputSpec("track_analysis", "movie", "table", "Per-track geometry, intensity, D, alpha, class"),
)


class MsdAnalysis(Analysis):
    name = "msd"
    requires_tracks = True
    default_measure_on = "preprocessed"
    catalogue = CATALOGUE

    def default_settings(self) -> dict:
        return {"max_lag": 25, "sigma_nm": 35.0, "min_length": 5,
                "max_tracks_plot": 200, "figures": "all", "tables": "all"}

    def validate(self, settings: dict) -> list:
        problems = []
        if settings["max_lag"] < 1:
            problems.append("max_lag must be at least 1")
        if settings["min_length"] < 3:
            problems.append("min_length must be at least 3")
        if settings["max_tracks_plot"] < 1:
            problems.append("max_tracks_plot must be at least 1")
        if settings["sigma_nm"] < 0:
            problems.append("sigma_nm must not be negative")
        return problems

    def run_movie(self, ctx, settings: dict, select) -> dict:
        units = Units(ctx.config.calibration, settings["sigma_nm"])
        tracked = ctx.tracks
        curves, fits = compute.compute_msd_analysis(
            tracked, units, max_lag=settings["max_lag"],
            min_length=settings["min_length"],
        )
        measure_on = measure_on_for(self, settings, ctx.config)
        image = ctx.image(measure_on) if measure_on else None
        table = compute.track_table(tracked, image, fits, units)

        out_dir = ctx.analysis_dir(self.name)
        for name, frame in (("msd_curves", curves), ("msd_fits", fits),
                            ("track_analysis", table)):
            if select.table(name):
                save_table(frame, out_dir / f"{name}.csv")

        counts = compute.class_counts(fits)
        # Fixed keys: class_counts adds "unclassified" only when non-empty, and
        # run_summary.csv columns must not come and go between movies.
        return {"n_fitted": int(len(fits)),
                "constrained": counts.get("constrained", 0),
                "diffusive": counts.get("diffusive", 0),
                "directed": counts.get("directed", 0),
                "unclassified": counts.get("unclassified", 0)}
```

`ctx.analysis_dir` creates the folder, so guard it: only call it when at least one table or figure is selected (a `tables: none, figures: none` run must leave no empty `msd/` folder — the test asserts this).

- [ ] **Step 4: Register it**

`msd/__init__.py`:

```python
"""MSD / motility analysis (ported from scripts/bulk_track_analysis.py)."""

from ..base import register
from .analysis import MsdAnalysis

register(MsdAnalysis())

__all__ = ["MsdAnalysis"]
```

In `batch/analyses/__init__.py` add `from . import msd  # noqa: F401  (registers the msd analysis)` after the `base` import, and mention in its docstring that analysis modules must not import `batch.runner` or `batch.validation` (both import this package — an import cycle).

In `presets/tracking.yaml` add:

```yaml
analyses:
  msd: {}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 6: Run the other batch tests**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests -k batch -q`
Expected: PASS. The phase-1 test `test_batch_cli.py::test_list_outputs` asserts "No analyses are registered yet" for an empty registry — it monkeypatches `REGISTRY`, so it still holds. If any phase-1 test fails because `msd` is now registered by default, report it rather than editing that test.

- [ ] **Step 7: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/batch/analyses/msd/__init__.py src/napari_spotiflow_tracking/batch/analyses/__init__.py src/napari_spotiflow_tracking/batch/presets/tracking.yaml src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py
git commit -m "feat(batch-msd): register the msd analysis and write its per-movie tables"
```

---

### Task 3: Theme axis kinds for MSD

**Files:**
- Modify: `src/napari_spotiflow_tracking/batch/theme.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py`

**Interfaces:**
- Consumes: `batch.config.Calibration`.
- Produces: `theme.axis_label(kind, calibration)` additionally accepts `speed` (`Speed (µm/s)` / `Speed (px/frame)`), `diffusion` (`D (µm²/s^α)` / `D (px²/frame^α)`), `duration` (`Duration (s)` / `Duration (frames)`), `path_length` (`Path length (µm)` / `Path length (px)`), `displacement` (`Net displacement (µm)` / `Net displacement (px)`), `alpha` (`α` in both modes), `lag` (`Lag (s)` / `Lag (frames)`).

- [ ] **Step 1: Write the failing test**

Append to `src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py`:

```python
@pytest.mark.parametrize("kind, physical, expected", [
    ("speed", False, "Speed (px/frame)"),
    ("speed", True, "Speed (µm/s)"),
    ("diffusion", False, "D (px²/frame^α)"),
    ("diffusion", True, "D (µm²/s^α)"),
    ("duration", False, "Duration (frames)"),
    ("duration", True, "Duration (s)"),
    ("path_length", False, "Path length (px)"),
    ("path_length", True, "Path length (µm)"),
    ("displacement", False, "Net displacement (px)"),
    ("displacement", True, "Net displacement (µm)"),
    ("lag", False, "Lag (frames)"),
    ("lag", True, "Lag (s)"),
    ("alpha", False, "α"),
    ("alpha", True, "α"),
])
def test_msd_axis_kinds(kind, physical, expected):
    cal = Calibration(pixel_size_nm=100, frame_interval_s=0.5) if physical else Calibration()
    assert theme.axis_label(kind, cal) == expected
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py -k msd_axis -v`
Expected: FAIL — `ValueError: unknown axis kind 'speed'`

- [ ] **Step 3: Extend `axis_label`**

Add the seven kinds to the `labels` dict in `theme.axis_label`, following the existing pattern (`"<name>": "<physical>" if physical else "<pixel>"`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py -v`
Expected: all PASS (12 existing + 14 new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/theme.py src/napari_spotiflow_tracking/_tests/test_batch_theme_outputs.py
git commit -m "feat(batch): axis labels for speed, D, duration, lag and displacement"
```

---

### Task 4: Core MSD figures

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py`

**Interfaces:**
- Consumes: Task 1 `compute`/`Units`, Task 3 axis kinds, `theme.style`, `theme.MOTILITY_COLORS`, `outputs.save_figure`.
- Produces in `figures_movie.py` (every function returns `None` and writes nothing when its input is empty):
  - `plot_msd_combined(curves, fits, units, stem, formats, dpi)` — port of `bulk:_plot_msd_combined` (L616)
  - `plot_msd_regimes(curves, fits, units, stem, formats, dpi)` — `bulk:_plot_msd_regimes` (L749)
  - `plot_motility_classes(fits, stem, formats, dpi)` — `bulk:_plot_motility_classes` (L724)
  - `plot_d_vs_alpha(fits, units, stem, formats, dpi)` — `bulk:_plot_d_vs_alpha` (L931)
  - module constants `FIG_H = 5.0`, `FIG_W = 7.0`, `FIG_W_PANEL = 4.5`, `LAG_RAMP`, `MIN_TRACKS_PER_LAG = 3`
  - Catalogue entries added to `analysis.CATALOGUE`: `msd_combined`, `msd_regimes`, `motility_classes`, `d_vs_alpha` (all `movie`/`figure`), each written as `<movie>/msd/<name>.<ext>`.

`stem` is the output path without extension; `formats`/`dpi` come from `ctx.config.outputs`.

- [ ] **Step 1: Write the failing tests**

Append to `test_batch_msd_analysis.py`:

```python
CORE_FIGURES = ["msd_combined", "msd_regimes", "motility_classes", "d_vs_alpha"]


def test_core_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=CORE_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    for name in CORE_FIGURES:
        png = msd_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 5000, name


def test_unselected_figure_is_not_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["motility_classes"])
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "motility_classes.png").exists()
    assert not (msd_dir / "msd_combined.png").exists()


def test_figures_respect_output_formats(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif")
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["motility_classes"])
    cfg.outputs.formats = ["png", "svg"]
    runner.run(cfg)
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    assert (msd_dir / "motility_classes.png").exists()
    assert (msd_dir / "motility_classes.svg").exists()


def test_empty_fits_write_no_figure(tmp_path):
    path = tmp_path / "g" / "empty.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), np.full((8, 32, 32), 100, np.uint16))
    cfg = msd_config(tmp_path, tmp_path / "g", figures=CORE_FIGURES)
    assert runner.run(cfg).exit_code == 0
    assert not list((tmp_path / "out" / "g" / "empty" / "msd").glob("*.png"))


def test_figure_module_uses_no_pyplot():
    import napari_spotiflow_tracking.batch.analyses.msd.figures_movie as fm
    source = Path(fm.__file__).read_text(encoding="utf-8")
    assert "pyplot" not in source and "matplotlib.use" not in source


def test_motility_classes_uses_theme_colours(tmp_path):
    """The bars must be the theme's class colours, not matplotlib defaults."""
    from matplotlib.colors import to_hex
    from napari_spotiflow_tracking.batch import theme
    from napari_spotiflow_tracking.batch.analyses.msd import compute, figures_movie
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    tracked = pd.concat([
        pd.DataFrame({"frame": range(30), "y": np.cumsum(np.random.default_rng(0).normal(0, 1, 30)),
                      "x": np.cumsum(np.random.default_rng(1).normal(0, 1, 30)), "particle": 0}),
    ])
    _, fits = compute.compute_msd_analysis(tracked, Units(Calibration()), max_lag=10,
                                           min_length=5)
    captured = {}
    figures_movie.plot_motility_classes(fits, tmp_path / "motility_classes",
                                        ["png"], 40, _capture=captured)
    drawn = {to_hex(c).lower() for c in captured["colors"]}
    assert drawn <= {to_hex(c).lower() for c in theme.MOTILITY_COLORS.values()}
```

For the last test, give `plot_motility_classes` a keyword-only `_capture: dict | None = None` parameter that records `{"colors": [...]}` when passed — the drawn colours are otherwise not inspectable without parsing the PNG. Do not add capture hooks to the other figures.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -k "figure or colour" -v`
Expected: FAIL — `ModuleNotFoundError: ...msd.figures_movie`

- [ ] **Step 3: Port the four figures**

Create `figures_movie.py` with the module docstring naming its source, the constants above, and the four functions. Follow the porting contract: read each source function, keep content and layout, swap style/colour/label/save as the contract says. Signature shape:

```python
def plot_msd_combined(curves, fits, units, stem, formats, dpi) -> None:
    if curves.empty or fits.empty:
        return
    with theme.style():
        fig = Figure(figsize=(FIG_W, FIG_H))
        ax = fig.subplots()
        ...
        save_figure(fig, stem, formats, dpi)
```

- [ ] **Step 4: Call them from `run_movie`**

In `analysis.py`, add the four `OutputSpec`s to `CATALOGUE`, then after the tables:

```python
        figures = ctx.config.outputs.formats, ctx.config.outputs.dpi
        if select.figure("msd_combined"):
            figures_movie.plot_msd_combined(curves, fits, units, out_dir / "msd_combined", *figures)
        ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -v`
Expected: all PASS (12 from Task 2 + 6 new).

- [ ] **Step 6: Eyeball the figures**

Run the smoke config from Task 9's Step 2 (or reuse a `tmp_path` run) and open `msd_combined.png`, `msd_regimes.png`, `motility_classes.png` and `d_vs_alpha.png`. Compare each against the same figure from the old script's output for one real movie. Check: axis labels carry units, no clipped text, legend present, class colours match the theme. Record what you compared in your report; if a figure looks wrong, fix it before committing.

- [ ] **Step 7: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py
git commit -m "feat(batch-msd): MSD combined, regimes, class shares and D-vs-alpha figures"
```

---

### Task 5: Distribution and per-track figures

**Files:**
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py`

**Interfaces:**
- Produces in `figures_movie.py`, each with the signature `(..., stem, formats, dpi) -> None`:

| Function | Source | Catalogue name | Input |
|---|---|---|---|
| `plot_length_hist` | `bulk:_plot_length_hist` (L406) | `length_hist` | `table`, `units`, `min_length` |
| `plot_path_length_hist` | `bulk:_plot_path_length_hist` (L440) | `path_length_hist` | `table`, `units` |
| `plot_speed_hist` | `bulk:_plot_speed_hist` (L466) | `speed_hist` | `table`, `units` |
| `plot_length_per_track` | `bulk:_plot_length_per_track` (L491) | `length_per_track` | `tracked`, `units`, `max_tracks` |
| `plot_speed_per_track` | `bulk:_plot_speed_per_track` (L514) | `speed_per_track` | `tracked`, `units`, `max_tracks` |
| `plot_directed_hist` | `bulk:_plot_directed_hist` (L1289) | `directed_displacement`, `directed_velocity` | `directed`, `units`, `column`, `xlabel` |
| `plot_directed_joint` | `bulk:_plot_directed_joint` (L1323) | `directed_displacement_vs_velocity` | `directed`, `units` |

`plot_directed_hist` is called twice, once per column, exactly as `bulk:analyze_folder` does (L1996-2006) — check the source for which column and label goes with which output name.

- [ ] **Step 1: Write the failing test**

Append to `test_batch_msd_analysis.py`:

```python
DISTRIBUTION_FIGURES = ["length_hist", "path_length_hist", "speed_hist",
                        "length_per_track", "speed_per_track",
                        "directed_displacement", "directed_velocity",
                        "directed_displacement_vs_velocity"]


def test_distribution_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=4)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=DISTRIBUTION_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    written = {p.stem for p in msd_dir.glob("*.png")}
    # The directed figures only exist when some track is classified directed.
    assert {"length_hist", "path_length_hist", "speed_hist", "length_per_track",
            "speed_per_track"} <= written
    assert written <= set(DISTRIBUTION_FIGURES)


def test_directed_figures_when_directed_tracks_exist(tmp_path, monkeypatch):
    """A movie of steadily drifting spots yields directed tracks."""
    import numpy as np
    rng = np.random.default_rng(2)
    size, n_frames = 96, 40
    yy, xx = np.mgrid[:size, :size]
    stack = np.full((n_frames, size, size), 100.0)
    starts = rng.uniform(10, 40, size=(5, 2))
    for t in range(n_frames):
        for cy, cx in starts + t * 1.2:
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path = tmp_path / "g" / "drift.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))

    cfg = msd_config(tmp_path, tmp_path / "g", figures=DISTRIBUTION_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "drift" / "msd"
    fits = pd.read_csv(msd_dir / "msd_fits.csv")
    assert (fits["motility_class"] == "directed").any()
    assert (msd_dir / "directed_displacement.png").exists()
    assert (msd_dir / "directed_velocity.png").exists()
    assert (msd_dir / "directed_displacement_vs_velocity.png").exists()


def test_max_tracks_plot_limits_per_track_figures(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_spots=8, seed=5)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["length_per_track"],
                     max_tracks_plot=2)
    assert runner.run(cfg).exit_code == 0
    assert (tmp_path / "out" / "g" / "m" / "msd" / "length_per_track.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -k "distribution or directed or max_tracks" -v`
Expected: FAIL — the figures are not written.

- [ ] **Step 3: Port the seven functions and wire them in**

Add them to `figures_movie.py` per the porting contract, add their `OutputSpec`s to `CATALOGUE`, and call them from `run_movie` behind `select.figure(...)`. `directed` comes from `compute.directed_metrics(table, units)`; compute it once, and skip all three directed figures when it is empty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py
git commit -m "feat(batch-msd): length, speed and directed-motion figures"
```

---

### Task 6: Explanatory MSD figures

**Files:**
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py`

**Interfaces:**
- Produces:

| Function | Source | Catalogue name |
|---|---|---|
| `plot_msd_construction` | `bulk:_plot_msd_construction` (L1556) | `msd_construction` |
| `plot_msd_archetypes` | `bulk:_plot_msd_archetypes` (L1634) | `msd_archetypes` |
| `plot_msd_shape_comparison` | `bulk:_plot_msd_shape_comparison` (L1710) | `msd_shape_comparison` |

These three follow Figure 1 of Schirripa Spagnolo & Luin, *Int. J. Mol. Sci.* **25**, 8660 (2024); keep the source's comments recording that. `plot_msd_archetypes` uses the fixed export width `FIG_W_ARCHETYPES = 6800 / 300` — copy that constant and its comment too, and keep `bulk:_representative_track`, `_track_xy`, `_fit_curve`, `_draw_scale_bar` (L1510-1555) as private helpers in the same module.

- [ ] **Step 1: Write the failing test**

```python
EXPLANATORY_FIGURES = ["msd_construction", "msd_archetypes", "msd_shape_comparison"]


def test_explanatory_figures_are_written(tmp_path):
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=6)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=EXPLANATORY_FIGURES)
    assert runner.run(cfg).exit_code == 0
    msd_dir = tmp_path / "out" / "g" / "m" / "msd"
    for name in EXPLANATORY_FIGURES:
        png = msd_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 5000, name


def test_archetypes_figure_is_full_width(tmp_path):
    """The archetype figure exports at a fixed manuscript width (three panels)."""
    from PIL import Image
    synthetic_movie(tmp_path / "g" / "m.tif", n_frames=60, n_spots=8, seed=7)
    cfg = msd_config(tmp_path, tmp_path / "g", figures=["msd_archetypes"])
    cfg.outputs.dpi = 100
    runner.run(cfg)
    with Image.open(tmp_path / "out" / "g" / "m" / "msd" / "msd_archetypes.png") as im:
        assert im.width > im.height * 2
```

If Pillow is not installed in the venv, read the PNG header instead (bytes 16-24 are width and height, big-endian uint32) and keep the same assertion.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -k explanatory -v`
Expected: FAIL.

- [ ] **Step 3: Port the three figures and wire them in**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py -v`
Expected: all PASS.

- [ ] **Step 5: Check the catalogue is complete**

Run: `.venv/Scripts/python.exe -m napari_spotiflow_tracking.batch.cli list-outputs msd` (or `.venv/Scripts/spotiflow-batch.exe list-outputs msd`)
Expected: 15 movie-level figures and 3 movie-level tables, matching the 14 figures + 3 tables `bulk:analyze_folder` writes (its `directed_*` trio counts as three names here).

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/figures_movie.py src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/_tests/test_batch_msd_analysis.py
git commit -m "feat(batch-msd): MSD construction, archetype and shape-comparison figures"
```

---

### Task 7: Group summaries

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/figures_group.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py`

**Interfaces:**
- Consumes: the per-movie CSVs written by `run_movie` (`<movie>/msd/msd_fits.csv`, `track_analysis.csv`), `cfg.calibration`, `cfg.outputs`.
- Produces:
  - `figures_group.read_movie_frames(movie_dirs, min_length, units) -> list` — one `(label, fits, props, directed)` tuple per movie that has the CSVs, ported from `bulk:_read_group_frames` (L2532) + `bulk:summary_rows_from_disk` (L2068); movies missing the files are skipped.
  - `plot_motility_summary`, `plot_d_vs_alpha_summary`, `plot_metric_summary`, `plot_length_vs_speed_summary`, `plot_joint_summary` — ports of `bulk:_plot_run_summary`-adjacent functions: `bulk:_plot_d_vs_alpha_summary` (L1040), `bulk:_plot_metric_summary` (L1191), `bulk:_plot_joint_summary` (L1379), `bulk:_plot_length_vs_speed_summary` (L1497), and the motility bars of `bulk:write_run_summary` (L2083).
  - `MsdAnalysis.summarize_group` writing into `<output_dir>/<group>/_summary/msd/`:

| Catalogue name | Kind | Source |
|---|---|---|
| `motility_summary` | figure | stacked class shares per movie (`bulk:write_run_summary`) |
| `motility_summary` | table | `motility_summary.csv` — per-movie class counts |
| `d_vs_alpha_summary` | figure | `bulk:_plot_d_vs_alpha_summary` |
| `length_summary` | figure | `bulk:_plot_metric_summary` on `length` |
| `speed_summary` | figure | `bulk:_plot_metric_summary` on `mean_speed` |
| `length_vs_speed_summary` | figure | `bulk:_plot_length_vs_speed_summary` |
| `directed_displacement_summary` | figure | `bulk:_plot_metric_summary` on `net_displacement` |
| `directed_velocity_summary` | figure | `bulk:_plot_metric_summary` on the velocity column |
| `directed_displacement_vs_velocity_summary` | figure | `bulk:_plot_joint_summary` |
| `group_fits` | table | all movies' `msd_fits` rows with a `movie` column |

A table and a figure may share a name (`motility_summary`) — `OutputSelector` keys on kind, so this is fine; the files differ by extension.

- [ ] **Step 1: Write the failing tests**

```python
"""Group summaries and cross-group comparison for the msd analysis."""
import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking.batch import runner
from napari_spotiflow_tracking.batch.config import load_config


def drifting_movie(path, n_frames=45, size=96, n_spots=6, drift=0.8, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    starts = rng.uniform(10, size / 2, size=(n_spots, 2))
    stack = np.full((n_frames, size, size), 100.0)
    for t in range(n_frames):
        for cy, cx in starts + t * drift + rng.normal(0, 0.3, size=starts.shape):
            stack[t] += 1200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack.astype(np.uint16))
    return path


def two_group_config(tmp_path, groups):
    return load_config(data={
        "inputs": [{"path": str(p), "label": label} for label, p in groups],
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocessing": {"steps": []},
        "detection": {"method": "log"},
        "tracking": {"enabled": True, "search_range": 3, "memory": 0,
                     "min_track_length": 5},
        "calibration": {"pixel_size_nm": 100, "frame_interval_s": 0.5},
        "outputs": {"formats": ["png"], "dpi": 40},
        "analyses": {"msd": {"max_lag": 10, "min_length": 5}},
    })


GROUP_FIGURES = ["motility_summary", "d_vs_alpha_summary", "length_summary",
                 "speed_summary", "length_vs_speed_summary",
                 "directed_displacement_summary", "directed_velocity_summary",
                 "directed_displacement_vs_velocity_summary"]


def test_group_summary_written_for_single_group(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=1)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=2)
    result = runner.run(two_group_config(tmp_path, [("Control", tmp_path / "ctrl")]))
    assert result.exit_code == 0
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    assert (summary / "motility_summary.csv").exists()
    counts = pd.read_csv(summary / "motility_summary.csv")
    assert len(counts) == 2                       # one row per movie
    assert {"movie", "constrained", "diffusive", "directed"} <= set(counts.columns)
    fits = pd.read_csv(summary / "group_fits.csv")
    assert set(fits["movie"]) == {"m1", "m2"}
    written = {p.stem for p in summary.glob("*.png")}
    assert {"motility_summary", "d_vs_alpha_summary", "length_summary",
            "speed_summary"} <= written


def test_group_summary_figure_selection(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=3)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg.analyses["msd"]["figures"] = ["motility_summary"]
    runner.run(cfg)
    summary = tmp_path / "out" / "Control" / "_summary" / "msd"
    assert (summary / "motility_summary.png").exists()
    assert not (summary / "d_vs_alpha_summary.png").exists()


def test_group_summary_skips_movies_without_msd_outputs(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=4)
    drifting_movie(tmp_path / "ctrl" / "m2.tif", seed=5)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    runner.run(cfg)
    # Remove one movie's tables, then re-summarise via a second run with skip_existing.
    (tmp_path / "out" / "Control" / "m2" / "msd" / "msd_fits.csv").unlink()
    cfg2 = two_group_config(tmp_path, [("Control", tmp_path / "ctrl")])
    cfg2.skip_existing = True
    assert runner.run(cfg2).exit_code == 0
    counts = pd.read_csv(tmp_path / "out" / "Control" / "_summary" / "msd" /
                         "motility_summary.csv")
    assert list(counts["movie"]) == ["m1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py -v`
Expected: FAIL — the summary folder is not created.

- [ ] **Step 3: Write `figures_group.py`**

Port the summary plots per the contract. The per-movie point colour in the summary scatter plots comes from `theme.MOTILITY_COLORS` where the source coloured by class, and from `theme.NEUTRAL` where it used a single hue.

- [ ] **Step 4: Implement `summarize_group`**

```python
    def summarize_group(self, group, movie_dirs, settings, select, out_dir, cfg):
        units = Units(cfg.calibration, settings["sigma_nm"])
        per_movie = figures_group.read_movie_frames(movie_dirs, settings["min_length"], units)
        if not per_movie:
            return
        ...
```

Write the tables through `save_table` and each figure behind `select.figure(...)`, exactly as `run_movie` does. Create `out_dir` only when something is written.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/figures_group.py src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py
git commit -m "feat(batch-msd): per-group summaries over a group's movies"
```

---

### Task 8: Cross-group comparison

**Files:**
- Create: `src/napari_spotiflow_tracking/batch/analyses/msd/figures_compare.py`
- Modify: `src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py`
- Test: `src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py`

**Interfaces:**
- Consumes: Task 7's `read_movie_frames`; `theme.group_colors`; `inputs.input_label`, `inputs.color_pins`.
- Produces:
  - `figures_compare.shared_hist_axes(group_values, n_bins=45)` — `bulk:_shared_hist_axes` (L2261)
  - `figures_compare.plot_group_overlay(...)` — `bulk:_plot_group_overlay` (L2302), drawn twice per metric: raw counts and `density=True`
  - `figures_compare.plot_group_joint(...)` — `bulk:_plot_group_joint` (L2383)
  - `figures_compare.plot_group_motility(...)` — `bulk:_plot_group_motility` (L2489)
  - `MsdAnalysis.compare_groups` writing into `<output_dir>/_comparison/msd/`, ported from `bulk:write_group_comparison` (L2562) and `bulk:_write_group_joints` (L2681):

| Catalogue name | Kind | Content |
|---|---|---|
| `comparison_summary` | table | per-group, per-movie class counts |
| `motility_comparison` | figure | class shares per group |
| `length_comparison`, `speed_comparison`, `directed_displacement_comparison`, `directed_velocity_comparison` | figure | overlaid group histograms, raw counts |
| `<same names>_density` | figure | the same, y as a fraction of each group's tracks |
| `d_vs_alpha_comparison`, `length_vs_speed_comparison`, `directed_displacement_vs_velocity_comparison` | figure | joint plots coloured by group |
| `harmonised` | figure | per-group single-condition figures on the shared axis range, written as `harmonised/<group>_<metric>.<ext>` |

Group colours: `theme.group_colors([input_label(s) for s in cfg.inputs], color_pins(cfg))`, so a group that produced nothing keeps its slot. The old `GROUP_COLORS_SAFE = 4` warning becomes a `log.warning` when more than four groups are drawn overlaid.

- [ ] **Step 1: Write the failing tests**

Append to `test_batch_msd_group.py`:

```python
def test_comparison_written_for_two_groups(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", drift=0.2, seed=6)
    drifting_movie(tmp_path / "drug" / "m1.tif", drift=1.5, seed=7)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    assert runner.run(cfg).exit_code == 0
    comparison = tmp_path / "out" / "_comparison" / "msd"
    table = pd.read_csv(comparison / "comparison_summary.csv")
    assert set(table["group"]) == {"Control", "Drug"}
    written = {p.stem for p in comparison.glob("*.png")}
    assert {"motility_comparison", "length_comparison", "speed_comparison"} <= written
    assert {"length_comparison_density", "speed_comparison_density"} <= written
    assert (comparison / "harmonised").is_dir()
    harmonised = {p.stem for p in (comparison / "harmonised").glob("*.png")}
    assert any(name.startswith("Control_") for name in harmonised)
    assert any(name.startswith("Drug_") for name in harmonised)


def test_no_comparison_for_a_single_group(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=8)
    runner.run(two_group_config(tmp_path, [("Control", tmp_path / "ctrl")]))
    assert not (tmp_path / "out" / "_comparison").exists()


def test_group_colours_follow_input_order(tmp_path):
    """A group whose movies all failed must not shift the others' colours."""
    from napari_spotiflow_tracking.batch import theme
    from napari_spotiflow_tracking.batch.inputs import color_pins, input_label
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=9)
    drifting_movie(tmp_path / "drug" / "m1.tif", seed=10)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    colors = theme.group_colors([input_label(s) for s in cfg.inputs], color_pins(cfg))
    assert colors["Control"] == theme.GROUP_COLORS[0]
    assert colors["Drug"] == theme.GROUP_COLORS[1]


def test_comparison_figure_selection(tmp_path):
    drifting_movie(tmp_path / "ctrl" / "m1.tif", seed=11)
    drifting_movie(tmp_path / "drug" / "m1.tif", seed=12)
    cfg = two_group_config(tmp_path, [("Control", tmp_path / "ctrl"),
                                      ("Drug", tmp_path / "drug")])
    cfg.analyses["msd"]["figures"] = ["motility_comparison"]
    runner.run(cfg)
    comparison = tmp_path / "out" / "_comparison" / "msd"
    assert (comparison / "motility_comparison.png").exists()
    assert not (comparison / "length_comparison.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py -k comparison -v`
Expected: FAIL — `_comparison` is not created.

- [ ] **Step 3: Write `figures_compare.py` and `compare_groups`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/batch/analyses/msd/figures_compare.py src/napari_spotiflow_tracking/batch/analyses/msd/analysis.py src/napari_spotiflow_tracking/_tests/test_batch_msd_group.py
git commit -m "feat(batch-msd): cross-group comparison figures and summary table"
```

---

### Task 9: Regression against the old script, docs and examples

**Files:**
- Create: `src/napari_spotiflow_tracking/_tests/test_batch_msd_regression.py`
- Modify: `pyproject.toml` (register the `slow` marker)
- Modify: `README.md`
- Modify: `examples/batch/two_conditions.yaml`
- Modify: `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`

**Interfaces:**
- Consumes: everything above; `scripts/bulk_track_analysis.py` loaded by path the way `_tests/test_bulk_cli.py` does.
- Produces: no new runtime code.

- [ ] **Step 1: Write the regression test**

```python
"""The ported msd analysis must reproduce the old script's numbers.

Opt-in: `pytest -m slow`. It runs the old script's analysis path and the new
analysis over the same tracks and compares the fit tables.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bulk_track_analysis.py"

pytestmark = pytest.mark.slow


def _load_script():
    if "bulk_track_analysis" in sys.modules:
        return sys.modules["bulk_track_analysis"]
    spec = importlib.util.spec_from_file_location("bulk_track_analysis", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bulk_track_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tracks(n_tracks=12, n_frames=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_tracks):
        drift = 0.8 if tid % 3 == 0 else 0.0
        pos = np.cumsum(rng.normal(drift, 1.0, size=(n_frames, 2)), axis=0)
        for frame, (y, x) in enumerate(pos):
            rows.append({"frame": frame, "y": y, "x": x, "particle": tid})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("pixel_size_nm, frame_interval_s", [(None, None), (100, 0.5)])
def test_fits_match_the_old_script(pixel_size_nm, frame_interval_s):
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    bulk = _load_script()
    tracked = _tracks()
    old_cal = bulk.Calibration(pixel_size_nm=pixel_size_nm,
                               frame_interval_s=frame_interval_s, sigma_nm=35.0)
    old_curves, old_fits = bulk.compute_msd_analysis(tracked, old_cal, max_lag=20,
                                                     min_length=5)
    units = Units(Calibration(pixel_size_nm=pixel_size_nm,
                              frame_interval_s=frame_interval_s), sigma_nm=35.0)
    new_curves, new_fits = compute.compute_msd_analysis(tracked, units, max_lag=20,
                                                        min_length=5)

    pd.testing.assert_frame_equal(old_curves.reset_index(drop=True),
                                  new_curves.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(old_fits.reset_index(drop=True),
                                  new_fits.reset_index(drop=True),
                                  check_exact=False, rtol=1e-6)
    assert (old_fits["motility_class"].values == new_fits["motility_class"].values).all()


def test_class_counts_match_the_old_script():
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    bulk = _load_script()
    tracked = _tracks(seed=3)
    _, old_fits = bulk.compute_msd_analysis(
        tracked, bulk.Calibration(sigma_nm=35.0), max_lag=20, min_length=5)
    _, new_fits = compute.compute_msd_analysis(
        tracked, Units(Calibration(), sigma_nm=35.0), max_lag=20, min_length=5)
    assert bulk.class_counts(old_fits) == compute.class_counts(new_fits)


def test_track_table_matches_the_old_columns():
    """Every column analyze_folder writes to track_analysis.csv must exist."""
    from napari_spotiflow_tracking.batch.analyses.msd import compute
    from napari_spotiflow_tracking.batch.analyses.msd.units import Units
    from napari_spotiflow_tracking.batch.config import Calibration

    tracked = _tracks(seed=4)
    units = Units(Calibration(pixel_size_nm=100, frame_interval_s=0.5), sigma_nm=35.0)
    _, fits = compute.compute_msd_analysis(tracked, units, max_lag=20, min_length=5)
    table = compute.track_table(tracked, None, fits, units)
    expected = {"track_id", "length", "total_displacement", "net_displacement",
                "mean_speed", "D", "alpha", "motility_class", "duration_s",
                "total_displacement_um", "net_displacement_um", "mean_speed_um_s"}
    assert expected <= set(table.columns)
```

- [ ] **Step 2: Register the marker and run the regression**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
markers = ["slow: opt-in tests that run the old bulk script for comparison"]
```

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_batch_msd_regression.py -m slow -v`
Expected: all 4 tests PASS. A mismatch means the port changed the numbers — fix the port, not the test, and report what differed.

- [ ] **Step 3: Update the README**

Replace the "### Analyses" section of `README.md` (the block starting `**This version ships no analyses yet**`) with:

````markdown
### Analyses

Detection and tracking run once per movie, and every enabled analysis reads
those results, writing tables and figures per movie, per group, and across
groups. `spotiflow-batch list-outputs <name>` lists what an analysis can write,
and `figures`/`tables` select from it (`all`, `none`, or a list of names).

**`msd`** — mean squared displacement and motility classification, enabled by
the `tracking` preset:

```yaml
analyses:
  msd:
    max_lag: 25           # longest lag fitted, in frames
    min_length: 5         # shortest track admitted to the fits
    sigma_nm: 35          # static localization error, used only with calibration
    max_tracks_plot: 200  # cap for the per-track figures
    figures: all
    tables: all
```

Every track's MSD is fitted with `MSD = 4·D·t^α + 4·σ²`, and α bins the track
into **constrained** (α < 0.4), **diffusive** (0.4 ≤ α ≤ 1.2) or **directed**
(α > 1.2), following the motility analysis in
[Commun Biol 4, 137 (2021)](https://www.nature.com/articles/s42003-021-01740-y#Sec9).
Per movie you get `msd_curves.csv`, `msd_fits.csv` and `track_analysis.csv`
beside 15 figures — the regime map, the combined MSD curves, class shares, the
D–α joint distribution, length/speed/displacement distributions, and the three
explanatory figures (construction, archetypes, shape comparison) that follow
Figure 1 of
[Schirripa Spagnolo & Luin, *Int. J. Mol. Sci.* **25**, 8660 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354962/).

Each group gets the same metrics pooled over its movies in
`<group>/_summary/msd/`, and a run with more than one group also writes
`_comparison/msd/`: overlaid group distributions, their density versions, the
harmonised per-group figures on a shared axis range, and
`comparison_summary.csv`. Group colours are assigned in `inputs` order and are
the same in every figure of the run.

With `calibration` set, D is reported in µm²/s^α and every axis reads µm and
seconds; without it, px and frames (and σ is dropped, since a nanometre figure
has no meaning in pixels).

PAPA cycle counts and photobleaching-step analysis are being ported next.
````

- [ ] **Step 4: Update the example config**

In `examples/batch/two_conditions.yaml`, append an `analyses` block with the `msd` settings shown in the README above, and a comment saying `spotiflow-batch list-outputs msd` lists the figure names for `figures:`.

- [ ] **Step 5: Update the spec's build order**

In `docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`, mark build-order item 2 **Done** the way item 1 is marked.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q` (≈6 min, timeout ≥ 900 s)
Expected: 0 failures, 0 errors. The `slow` tests are deselected by default; confirm the count grew by the new non-slow tests.

- [ ] **Step 7: Smoke-run on real data**

Write `<scratchpad>/msd_smoke.yaml` pointing `inputs` at two real dataset folders (ask the user which, or reuse `D:/Data/Sudipta/Arpan/GoodDataset/field2.tif` as a single group) with `modality: tracking`, `calibration` filled in, and `output_dir` inside the scratchpad. Run `.venv/Scripts/spotiflow-batch.exe run <scratchpad>/msd_smoke.yaml`, then record: exit code, `run_summary.csv` contents, which figures appeared per movie, per group and (if two groups) in `_comparison`. Open `msd_regimes.png` and `d_vs_alpha.png` and compare them against the old script's versions for the same data.

- [ ] **Step 8: Commit**

```bash
git add src/napari_spotiflow_tracking/_tests/test_batch_msd_regression.py pyproject.toml README.md examples/batch/two_conditions.yaml
git add -f docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md
git commit -m "test(batch-msd): regression against the old script; document the msd analysis"
```

---

## Self-review notes

- Spec coverage for phase 2: `msd` analysis with its catalogue and settings (Tasks 2, 4–6); calibration-aware fits and physical units (Task 1, 3); per-movie tables and all 15 figures of `analyze_folder` (Tasks 2, 4, 5, 6); group summaries (Task 7); cross-group comparison replacing `--compare`, with group colours following `inputs` order (Task 8); regression against the old script (Task 9); docs (Task 9). The MkDocs page stays in build-order item 5.
- Names used across tasks: `Units(calibration, sigma_nm)` (Task 1 → 2, 7, 8, 9); `compute.compute_msd_analysis(tracked, units, max_lag, min_length) -> (curves, fits)`, `compute.track_table(tracked, image, fits, units)`, `compute.class_counts`, `compute.directed_metrics` (Task 1 → 2, 5, 7); `figures_movie.plot_*(..., stem, formats, dpi)` (Tasks 4–6 → `analysis.run_movie`); `figures_group.read_movie_frames(movie_dirs, min_length, units)` (Task 7 → 8); catalogue names are the output filenames throughout.
- Deliberate deviations from the old script, each recorded in the porting contract: colours come from the theme (phase-1 ruling — `tab:orange`/`tab:green` fail under protanopia); figures are saved through `outputs.save_figure` (lock-safe, multi-format); `min_track_length` is the analysis's own `min_length` setting rather than a CLI flag shared with tracking.
