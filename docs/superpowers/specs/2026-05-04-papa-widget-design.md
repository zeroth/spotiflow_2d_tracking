# PAPA Widget — Design Spec

**Date:** 2026-05-04
**Scope:** Add a fourth widget, "PAPA" (Proximity-Assisted Photoactivation), to the `napari-spotiflow-tracking` plugin. The widget consumes a `napari.layers.Points` layer (output of the Detection widget) and renders two interactive matplotlib figures that summarise particles per frame across alternating Group 1 / Group 2 cycles, mirroring the analysis in [papa_analysis.py](../../../papa_analysis.py).

## 1. Architecture & Files

### New module — `src/napari_spotiflow_tracking/_papa_analysis.py`

Pure-Python (no Qt) helpers ported from `papa_analysis.py`. Importable for tests.

- `compute_particles_per_frame(points: np.ndarray) -> pd.DataFrame`
  - Input: `(N, 3)` array with columns `(frame, y, x)` from a Points layer.
  - Output: DataFrame with columns `frame` (1-indexed `int`) and `n_particles` (`int`), reindexed so every frame from `1..max_frame` is present (zero-detection frames included with `n_particles=0`).
- `build_cycle_info(max_time: int, group1_size: int, group2_size: int, skip_frames: int) -> list[dict]`
  - Logic copied verbatim from `papa_analysis.py:47-79`. Cycles 1-indexed.
- `annotate_with_cycle_group(per_frame: pd.DataFrame, cycle_info: list[dict]) -> pd.DataFrame`
  - Adds `cycle` (`Int64`, nullable) and `group` (`Int64`, nullable) columns. Frames falling in the skip gap between groups receive `NaN` in both columns.

### New widget — `src/napari_spotiflow_tracking/_papa_widget.py`

`PAPAWidget(QWidget)` containing the Qt UI, two embedded matplotlib canvases, and the rendering functions.

Rendering functions (private methods or module-level):
- `_render_combined_plot(fig, per_frame, cycle_info, max_time, g1, g2, skip, label1, label2)` — clears `fig`, plots particles-per-frame line + cycle/group separators.
- `_render_ratio_plot(fig, per_frame, cycle_info, label1, label2)` — clears `fig`, draws the 3-panel ratio figure.
- `_style_figure(fig)` — applies hardcoded napari dark theme (background `#262930`, foreground `#d6d6d6`).

### Manifest — `src/napari_spotiflow_tracking/napari.yaml`

Add a fourth command + widget entry, after the Tracking widget. The order Pre-Processing → Detection → Tracking → PAPA reflects the user workflow.

```yaml
contributions:
  commands:
    - id: napari-spotiflow-tracking.papa_widget
      python_name: napari_spotiflow_tracking._papa_widget:PAPAWidget
      title: PAPA
  widgets:
    - command: napari-spotiflow-tracking.papa_widget
      display_name: PAPA
```

### Dependency — `pyproject.toml`

Add `matplotlib` to `[project] dependencies`. It is currently only pulled in transitively (via napari / scikit-image) — making it explicit is correct for direct use.

### Tests

New file `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`. See Section 6.

## 2. Widget Layout & UI

The widget's top-level layout is a `QTabWidget` with two tabs.

```
┌─ PAPA ────────────────────────────────────┐
│ ┌──[ Input ]──[ Plots ]──┐                │
│ │  (active tab content)   │                │
│ └─────────────────────────┘                │
└────────────────────────────────────────────┘
```

### Input tab

`QVBoxLayout` containing:

1. **Input** (`QGroupBox`)
   - `QLabel("Points layer:")` + `QComboBox` + `↻` refresh button (`QPushButton`, fixed width 30).
2. **Cycle Pattern** (`QGroupBox`)
   - `Group 1 size:` `QLineEdit("5")` with `QIntValidator(1, 100000)`.
   - `Group 2 size:` `QLineEdit("5")` with `QIntValidator(1, 100000)`.
   - `Skip frames:` `QLineEdit("1")` with `QIntValidator(0, 100)`.
3. **Labels** (`QGroupBox`)
   - `Group 1 label:` `QLineEdit("DR")`.
   - `Group 2 label:` `QLineEdit("PAPA")`.
4. `QPushButton("Plot")`.
5. `QPushButton("Export CSV")` — disabled until the first successful plot.
6. `QLabel` (status), initial text `"Ready. Set parameters and click Plot."`.
7. `addStretch()`.

### Plots tab

`QVBoxLayout` inside a `QScrollArea` (so the two figures can scroll vertically without stretching the dock):

```
[ NavigationToolbar2QT for fig_combined ]
[ FigureCanvas        for fig_combined ]
[ NavigationToolbar2QT for fig_ratio    ]
[ FigureCanvas        for fig_ratio    ]
```

Initial state: both figures are blank (no axes drawn) until the first Plot click.

After a successful Plot run, the widget calls `self._tabs.setCurrentWidget(self._plots_tab)` so the user sees the result.

### Layer combo refresh

Same pattern as [_tracking_widget.py:94-99](../../../src/napari_spotiflow_tracking/_tracking_widget.py#L94-L99): connect to `viewer.layers.events.inserted` and `viewer.layers.events.removed`, both calling `self._refresh_points_combo()`. Manual refresh button as fallback.

## 3. Data Flow & Computation

### Inputs

- `napari.layers.Points` with `data` of shape `(N, 3)` and columns `(frame, y, x)`. Frames are 0-indexed (napari convention).
- Cycle parameters from the `QLineEdit` fields: `group1_size: int`, `group2_size: int`, `skip_frames: int`.
- Labels: `label_group1: str`, `label_group2: str`.

### Pipeline on Plot click

1. Validate the layer (Section 5 enumerates the guards).
2. `per_frame = compute_particles_per_frame(points)` — internally does `np.rint(...).astype(int) + 1` to convert napari's 0-indexed frames to 1-indexed (matches `papa_analysis.py`), then groupby + reindex. Returns DataFrame with columns `frame` (1-indexed) and `n_particles`, with every frame in `[1, max_frame]` present (zero-detection frames included).
3. `max_time = int(per_frame["frame"].max())`. **Note:** this is the max *detected* frame in the Points layer, not the total image stack length. Frames in the original image after the last detection are not considered. This is a deliberate choice — the widget has no access to the source image — and means a Points layer that is sparse at its tail will produce a slightly truncated cycle plot relative to running `papa_analysis.py` on the same image.
4. `cycle_info = build_cycle_info(max_time, group1_size, group2_size, skip_frames)`.
5. `per_frame = annotate_with_cycle_group(per_frame, cycle_info)` — adds nullable `cycle`/`group` columns; skip-gap frames get `NaN` in both.
6. `_render_combined_plot(self._fig_combined, per_frame, cycle_info, max_time, ...)` then `self._canvas_combined.draw_idle()`.
7. `_render_ratio_plot(self._fig_ratio, per_frame, cycle_info, ...)` then `self._canvas_ratio.draw_idle()`.
8. Cache: `self._last_per_frame_df = per_frame`.
9. Enable Export CSV button. Set status label to `"Plotted {n_cycles} cycles across {max_time} frames."` — and if guard #8 in Section 5 fired, append `" (cycle pattern exceeds frame range; only partial cycles drawn.)"`. Switch active tab to Plots.

### CSV export

On Export CSV button click:
- `path, _ = QFileDialog.getSaveFileName(self, "Save particles CSV", "", "CSV files (*.csv)")`.
- If `path`, write `self._last_per_frame_df` to CSV with columns `frame, n_particles, cycle, group` — frames 1-indexed; rows for skipped frames included with `cycle=NaN, group=NaN, n_particles=0`.
- Show `napari.utils.notifications.show_info` on success.

### Computation runs synchronously

Groupby + matplotlib draw on a few thousand frames is fast enough that no `QThread` worker is needed for v1. If real datasets prove slow, a worker following [_workers.py](../../../src/napari_spotiflow_tracking/_workers.py) can be added later.

## 4. Plot Rendering & Theming

### Embedding

Each figure uses Qt-Agg embedding directly (no `napari-matplotlib` dependency):

```python
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

self._fig_combined = Figure(figsize=(10, 4), tight_layout=True)
self._canvas_combined = FigureCanvas(self._fig_combined)
self._toolbar_combined = NavigationToolbar(self._canvas_combined, self)
```

Same pattern for `_fig_ratio` with `figsize=(10, 9)` (three stacked panels). `NavigationToolbar2QT` provides Home, Pan, Zoom-rect, Back/Forward, Configure subplots, Save — covers the user's interactivity requirements (zoom, etc.).

### Theming

Hardcoded napari dark-theme colors applied at the end of each render:

```python
NAPARI_BG = "#262930"
NAPARI_FG = "#d6d6d6"

def _style_figure(fig):
    fig.patch.set_facecolor(NAPARI_BG)
    for ax in fig.axes:
        ax.set_facecolor(NAPARI_BG)
        for spine in ax.spines.values():
            spine.set_color(NAPARI_FG)
        ax.tick_params(colors=NAPARI_FG)
        ax.xaxis.label.set_color(NAPARI_FG)
        ax.yaxis.label.set_color(NAPARI_FG)
        ax.title.set_color(NAPARI_FG)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color(NAPARI_FG)
```

Live theme tracking is out of scope.

### Combined plot

- One `Axes`. Plot `per_frame["frame"]` vs `per_frame["n_particles"]` as a line, color `#e0e0e0`, `linewidth=1.0`.
- Vertical separators at `info["end_frame"] + 0.5` for each `cycle_info` entry where `end_frame < max_time`. Group 1 separators use `tab:blue`, Group 2 use `tab:orange` (alternating via `(group - 1) % 2`). The first separator of each group gets a legend label (`label_group1` / `label_group2`); subsequent separators have `label=None` to avoid duplicate legend entries.
- Title: `f"All Cycles Combined — Particles per Frame  (G1={g1}f + skip{skip}f + G2={g2}f + skip{skip}f)"`.
- xlabel `"Time (frame)"`, ylabel `"Number of Particles"`. Legend `loc="upper right", fontsize=8`. Grid `alpha=0.3`.

### Ratio plot

- `fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})`.
- Panel 1 (`ax1`): grouped bar chart — `tab:blue` bars (`label_group1 + " (G1)"`) and `tab:orange` bars (`label_group2 + " (G2)"`). ylabel `"Total Particle Count"`, title `"Total Particle Count per Group per Cycle"`. Legend `fontsize=8`.
- Panel 2 (`ax2`): `tab:green` bars showing `g1_total / g2_total` per cycle, with `axhline(y=1.0, color="gray", linestyle="--")`. ylabel `f"Ratio ({label1} / {label2})"`, title `f"Particle Count Ratio ({label1} / {label2}) per Cycle"`.
- Panel 3 (`ax3`): `tab:red` bars showing `g2_total / g1_total` per cycle, with the same `axhline`. ylabel/title use `label2 / label1`.
- `xticks` are cycle indices (1-indexed). xlabel on the bottom panel: `"Cycle"`. Grid `axis="y", alpha=0.3` on each panel.

Both render functions clear the figure (`fig.clear()`) before drawing, so re-running Plot replaces rather than overlays.

## 5. Error Handling & Edge Cases

All errors update the status label *and* call `napari.utils.notifications.show_error(...)`, matching [_tracking_widget.py:115-117](../../../src/napari_spotiflow_tracking/_tracking_widget.py#L115-L117).

### Validation guards on Plot click (in order)

1. No Points layer selected → `"No Points layer selected."`
2. Selected layer no longer in viewer → `"Selected layer no longer exists."`
3. Layer is not a `napari.layers.Points` → `"Selected layer is not a Points layer."`
4. `data.ndim != 2 or data.shape[1] != 3` → `"Points layer must have 3 columns (frame, y, x)."`
5. `len(data) == 0` → `"Points layer is empty; nothing to plot."`
6. Empty / non-positive `QLineEdit` → `"Group 1 size must be a positive integer."`, `"Group 2 size must be a positive integer."`, `"Skip frames must be a non-negative integer."`
7. `max_time < 1` → `"Need at least 1 frame of data."`
8. `group1_size + skip_frames + group2_size + skip_frames > max_time` — **soft warning**, not a hard error. Plot still runs; the success-status string in Section 3 step 9 has the warning appended to it.

### Edge cases in computation

- **Frames with zero detections** — handled by reindex inside `compute_particles_per_frame`.
- **Negative or non-integer frame values** — coerce via `np.rint(...).astype(int)`. Negative values get clipped to `0` (becoming frame `1` after the +1 shift), with a status warning `"Negative frame indices found; clipped to 0."`.
- **Final partial cycle missing G2** — `build_cycle_info` already handles this; the ratio plot's G1/G2 ratio for that cycle is `NaN`, matplotlib renders no bar.
- **Division by zero in ratios** — guarded with `g1_total / g2_total if g2_total > 0 else np.nan` (and symmetric for G2/G1), copied from `papa_analysis.py:154-155`.

### Export CSV

Disabled until first successful plot. On `QFileDialog` cancel: no-op. On write failure: error in status label + notification.

## 6. Testing

New file `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`. Uses `pytest-qt`'s `qtbot` and the `make_napari_viewer` fixture.

### Pure-function tests (no Qt)

1. `test_compute_particles_per_frame_basic` — array with frames having {1, 0, 4} spots produces the expected DataFrame (1-indexed frames after the conversion).
2. `test_compute_particles_per_frame_fills_missing_frames` — input only has frames 0, 2, 5; output reindexes to include 1, 3, 4 with `n_particles=0`.
3. `test_build_cycle_info_exact_fit` — `max_time=12, g1=5, g2=5, skip=1` → 1 complete cycle with G1 frames 1-5, G2 frames 7-11.
4. `test_build_cycle_info_partial_g2` — `max_time=8, g1=5, g2=5, skip=1` → cycle 1 G1 frames 1-5, G2 frames 7-8 (truncated by max_time).
5. `test_build_cycle_info_no_room_for_g2` — `max_time=5, g1=5, g2=5, skip=1` → only G1, no G2.
6. `test_annotate_with_cycle_group` — verifies `cycle` and `group` columns are correctly assigned, including `NaN` for skip-gap frames.

### Widget tests (with `qtbot`)

7. `test_widget_construction` — instantiate `PAPAWidget(viewer)`; verify both tabs and both canvases exist; verify Export CSV starts disabled.
8. `test_plot_button_no_layer` — click Plot with no Points layer; status updated; Export CSV stays disabled.
9. `test_plot_button_happy_path` — add a synthetic Points layer with known spots across frames; set group sizes; click Plot; status says `"Plotted N cycles across M frames."`; Export CSV becomes enabled; active tab switches to Plots.
10. `test_export_csv` — after a successful plot, monkeypatch `QFileDialog.getSaveFileName` to return a `tmp_path` file; click Export CSV; assert file exists; load CSV and assert columns `frame, n_particles, cycle, group`, 1-indexed frames, and `NaN` for skip-gap rows.
11. `test_invalid_lineedit_input` — clear `group1_size`, click Plot; error in status label; no plot rendered (figures still blank).
12. `test_layer_combo_refresh` — add and remove a Points layer; verify the combo updates via `events.inserted` / `events.removed`.

### Out of scope

- Pixel-level matplotlib output. Tests verify draw didn't raise and figures have axes; visual review is the user's job.
- Asserting against napari's live theme. Hardcoded values.
