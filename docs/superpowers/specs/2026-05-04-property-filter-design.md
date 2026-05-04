# Property Filter Design Spec

**Date:** 2026-05-04
**Scope:** Add a reusable property-filter component to the Detection and Tracking widgets of the `napari-spotiflow-tracking` plugin. Inspired by the `FilterWidget` in the reference plugin at `D:\Code\Lab\napari-tracking-analysis`. Phase 1 covers the filter component, the per-track property computation it depends on, and the integrations into Detection and Tracking. Per-track selection + quick-analysis (Phase 2) is deliberately out of scope for this spec.

## 1. Architecture & Files

### New module — `src/napari_spotiflow_tracking/_filter_panel.py`

A single `FilterPanel(QWidget)` class plus three file-private helpers (no separate modules).

```python
FilterPanel(
    viewer: napari.Viewer,
    layer_kind: Literal["Points", "Tracks"],
    id_column: str,             # "spot_id" or "track_id"
    parent: QWidget | None = None,
)
```

Internals (file-private):

- `_PropertyTableModel(QAbstractTableModel)` — wraps the active layer's `metadata["properties"]` DataFrame.
- `_PropertyTableProxy(QSortFilterProxyModel)` — filters by per-column ranges, stored in a `dict[str, tuple[float, float]]`. Calls `invalidateFilter()` when ranges change.
- `_PropertyHistogram(QWidget)` — `FigureCanvasQTAgg` showing overlaid normalized histograms for the currently-checked properties (see §2).

**Range sliders:** use `superqt.QLabeledRangeSlider` (already a transitive dependency via napari). Dual-handle horizontal slider with min/max labels and proper int/float typing. No custom range slider widget is needed.

**Public signal** (Phase-2 hook, no-op in Phase 1):

```python
selectionChanged = Signal(int)  # emits row's id_column value when user clicks a table row
```

### New module — `src/napari_spotiflow_tracking/_track_properties.py`

```python
def compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame
```

Pure-Python helper. See §3 for the body.

### New shared module — `src/napari_spotiflow_tracking/_mpl_theme.py`

Lifted from [_papa_widget.py](../../../src/napari_spotiflow_tracking/_papa_widget.py). Contains:

```python
NAPARI_BG = "#262930"
NAPARI_FG = "#d6d6d6"
LINE_COLOR = "#e0e0e0"

def style_figure(fig: matplotlib.figure.Figure) -> None: ...
```

`PAPAWidget._style_figure` is replaced with a call to `style_figure`. Constants on the class (`NAPARI_BG`, `NAPARI_FG`, `LINE_COLOR`) are kept and re-export from the shared module so existing tests on `widget.NAPARI_BG` still pass. `FilterPanel`'s histogram uses the same `style_figure`.

### Modifications

- **[_detection_widget.py](../../../src/napari_spotiflow_tracking/_detection_widget.py)**
  - Wrap the existing top-level `QVBoxLayout` in a `QScrollArea` (`setWidgetResizable(True)`).
  - On detect-finish, build a per-spot DataFrame with columns `spot_id, frame, y, x, intensity` and store on `points_layer.metadata["properties"]`.
  - On mask-generation-finish, merge regionprops (area, mean_intensity, equivalent_diameter, solidity, etc.) into the same Points layer's `metadata["properties"]` on `(frame, y, x)`.
  - Remove the existing primitive Spot Filtering group (current lines 207-259, 608-661) — `_apply_filter`, `_filter_mask_combo`, `_min_area`, `_max_area`, `_min_intensity`, `_max_intensity` and the "Apply Filter" button.
  - Append a `FilterPanel(viewer, "Points", "spot_id")` instance below the existing Detect button section.
- **[_tracking_widget.py](../../../src/napari_spotiflow_tracking/_tracking_widget.py)**
  - Wrap the layout in a `QScrollArea`.
  - Add an Image-layer combo below the existing Points-layer combo. Defaults to empty (no image) → no intensity columns. Refresh on layer events.
  - On track-finish, call `compute_track_properties(tracked_df, image=...)` and store the result on `tracks_layer.metadata["properties"]`. Also store the raw `tracked_df` on `tracks_layer.metadata["tracked_df"]` (consumed by FilterPanel to rebuild the sibling Tracks layer when filters change).
  - Append a `FilterPanel(viewer, "Tracks", "track_id")` below the existing Track button section.
- **[_papa_widget.py](../../../src/napari_spotiflow_tracking/_papa_widget.py)** — minor: replace `_style_figure` body with a call to the new `style_figure` helper. Keep class constants `NAPARI_BG`, `NAPARI_FG`, `LINE_COLOR` re-exported from `_mpl_theme.py` so `test_papa_widget.py::TestPAPAWidgetTheming` continues to use `widget.NAPARI_BG`.

### Tests

- `_tests/test_track_properties.py` — pure-Python helper tests.
- `_tests/test_filter_panel.py` — Qt widget tests using `pytest-qt`.
- `_tests/test_filter_integration.py` — end-to-end through the host widgets.

See §6.

## 2. UI Layout

`FilterPanel` is rendered as a single `QGroupBox` titled "Filter":

```
┌─ Filter ───────────────────────────────────┐
│ Layer: [combo] [↻]                         │
│                                            │
│ ┌─ Range filters ──────────────────────┐  │
│ │ length         [QLabeledRangeSlider]  │  │
│ │ mean_intensity [QLabeledRangeSlider]  │  │
│ │ ... one row per filterable column     │  │
│ └───────────────────────────────────────┘  │
│                                            │
│ Show histograms:                           │
│   ☑ length            (color swatch)       │
│   ☑ mean_intensity    (color swatch)       │
│   ☑ net_displacement  (color swatch)       │
│ ┌──────────────────────────────────────┐  │
│ │ [overlaid normalized histograms]      │  │
│ │  legend: length [1 - 87]              │  │
│ │          mean_intensity [102 - 25341] │  │
│ └──────────────────────────────────────┘  │
│                                            │
│ ☑ Show filtered in viewer                  │
│ ┌─ Filtered Rows ──────────────────────┐  │
│ │ QTableView (single-row selection)     │  │
│ └───────────────────────────────────────┘  │
│ Status: 142 / 247 rows match               │
│ [ Reset Filters ]  [ Export Filtered CSV ] │
└────────────────────────────────────────────┘
```

**Layer combo** auto-populates with `viewer.layers` filtered by `layer_kind`. Refreshed via `events.inserted` / `events.removed`. On selection change, the panel reads the layer's `metadata["properties"]` and rebuilds sliders/checkboxes/histogram.

**Range filters** are auto-generated:

- One row per numeric column in the DataFrame, excluding `id_column` and spatial coords (`x, y, frame` for Points; `t, x, y, track_id` for Tracks).
- Integer columns use `QLabeledRangeSlider` configured for ints; float columns use floats.
- Slider range = `[col.min(), col.max()]`; initial value = the full range.
- Slider label shows the column name to the left of the slider.
- **Constant columns** (`col.min() == col.max()`) display a static label `"<col>: constant <value>"` instead of a slider — they take no part in filtering.

**Slider debounce:** slider value changes are routed through a single `QTimer.singleShot(150, self._refresh)` to avoid frame thrash on continuous drag. The 150 ms timer resets on each change.

**Histogram (3-inch fixed height, ~80 px wider than the panel content):**

- Single matplotlib `Axes` inside `FigureCanvasQTAgg` (3 inches tall, fixed via `setFixedHeight`).
- One overlaid histogram per checked property: 30 bins per property, edges from `(col.min(), col.max())`, x-mapped to `[0, 1]` so all properties share the same x-axis.
- `histtype="stepfilled", alpha=0.5`. Distinct colors via `matplotlib.cm.tab10` (cycled if more than 10 properties).
- Legend formatted as `f"{name}  [{cmin:.3g} – {cmax:.3g}]"`. Legend in upper-right.
- Theme via `_mpl_theme.style_figure(fig)`.
- Redraws on:
  - filter-debounce tick (post-filter rows used for the histogram data),
  - any property-checkbox toggle,
  - layer change.

**Property-show checkboxes** appear above the canvas in a vertical layout. Each `QCheckBox`'s text color is the property's plot color (set via `palette` so the QCheckBox label matches the histogram trace). Checked by default. Toggling redraws the histogram only — does not affect the filter / table.

**Filtered Rows table** uses `QTableView` + `_PropertyTableProxy`, single-row selection enabled (`SelectionMode.SingleSelection`, `SelectionBehavior.SelectRows`). Selecting a row emits `selectionChanged(int(row[id_column]))` (Phase 2 hook).

**Status label** updates on every `_refresh()` call: `f"{n_filtered} / {n_total} rows match"`.

**Reset Filters** sets every slider back to its column's full range and clears the proxy filter dict; triggers a `_refresh()`.

**Export Filtered CSV** opens a `QFileDialog`. On confirm, writes the post-filter rows (column order = original DataFrame order, including all id/spatial columns) via `to_csv(path, index=False)`. If 0 rows match, writes a header-only CSV and shows a status warning.

**Show filtered in viewer** checkbox controls the sibling layer (see §3).

## 3. Data Flow & Computation

### Points-layer properties (Detection)

At detect-finish in [_detection_widget.py:497-509](../../../src/napari_spotiflow_tracking/_detection_widget.py#L497-L509), after `find_spots_spotiflow` returns `(points, details)`, the worker's `finished` signal is extended to include the `details.intens` array. The widget builds:

```python
n = len(points)
spot_id = np.arange(n, dtype=int)
if points.shape[1] == 3:  # 3D: (frame, y, x)
    df = pd.DataFrame({
        "spot_id": spot_id,
        "frame": points[:, 0].astype(int),
        "y": points[:, 1],
        "x": points[:, 2],
        "intensity": np.asarray(intensities, dtype=float),
    })
else:  # 2D: (y, x)
    df = pd.DataFrame({
        "spot_id": spot_id,
        "y": points[:, 0],
        "x": points[:, 1],
        "intensity": np.asarray(intensities, dtype=float),
    })
points_layer.metadata["properties"] = df
```

At mask-generation finish in [_detection_widget.py:562-578](../../../src/napari_spotiflow_tracking/_detection_widget.py#L562-L578):

- `MaskGenerationWorker.finished` continues to deliver `(mask, props_df)`.
- The widget tracks which Points layer the mask gen was launched against (a new `self._mask_target_points: napari.layers.Points | None`).
- If the target Points layer still exists:
  - If it already has `metadata["properties"]` (it was created via Detection), merge regionprops into it on rounded `(frame, y, x)`:

    ```python
    existing = target_points.metadata.get("properties")
    if isinstance(existing, pd.DataFrame):
        key_cols = ["frame", "y", "x"] if "frame" in existing.columns else ["y", "x"]
        left = existing.copy()
        right = props_df.copy()
        for k in key_cols:
            left[f"_k_{k}"] = np.rint(left[k]).astype(int)
            right[f"_k_{k}"] = np.rint(right[k]).astype(int)
        merged = left.merge(
            right.drop(columns=key_cols),
            on=[f"_k_{k}" for k in key_cols], how="left",
        ).drop(columns=[f"_k_{k}" for k in key_cols])
        target_points.metadata["properties"] = merged
    ```

  - If `metadata["properties"]` is missing (the Points layer was imported by the user, not from Detection), build a fresh DataFrame from `props_df` directly: add a `spot_id = np.arange(len(props_df))` column at the front, keep all regionprops columns, and assign as the new `metadata["properties"]`. This ensures the FilterPanel still works on Points layers the user didn't generate via Detection.

The mask layer's `metadata["regionprops"]` continues to be set (back-compat for any external consumer).

### Tracks-layer properties (Tracking)

The Tracking widget gains an Image-layer combo (always visible, defaults to empty). On track-finish:

```python
image_name = self._image_combo.currentText()
image = (
    np.asarray(self.viewer.layers[image_name].data)
    if image_name and image_name in self.viewer.layers
    else None
)
track_props = compute_track_properties(tracked_df, image=image)
tracks_layer.metadata["properties"] = track_props
```

`compute_track_properties` body in `_track_properties.py`:

```python
import numpy as np
import pandas as pd


def compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute per-track geometric (and optionally intensity) properties.

    Input DataFrame must have columns: frame (int), y (float), x (float),
    particle (int).
    """
    if len(tracked_df) == 0:
        cols = ["track_id", "length", "total_displacement",
                "net_displacement", "mean_speed"]
        if image is not None:
            cols += ["mean_intensity", "max_intensity", "intensity_std"]
        return pd.DataFrame(columns=cols)

    if image is not None:
        if image.ndim == 3:
            t_max = int(tracked_df["frame"].max())
            if image.shape[0] <= t_max:
                raise ValueError("Image T-axis does not span tracked frames")
        elif image.ndim != 2:
            raise ValueError("Image must be 2D (Y, X) or 3D (T, Y, X)")

    rows = []
    for tid, grp in tracked_df.groupby("particle"):
        grp = grp.sort_values("frame")
        coords = grp[["y", "x"]].to_numpy()
        length = len(grp)
        diffs = np.diff(coords, axis=0)
        total = float(np.linalg.norm(diffs, axis=1).sum()) if length > 1 else 0.0
        net = float(np.linalg.norm(coords[-1] - coords[0])) if length > 1 else 0.0
        mean_speed = total / max(length - 1, 1)
        row = {
            "track_id": int(tid),
            "length": length,
            "total_displacement": total,
            "net_displacement": net,
            "mean_speed": mean_speed,
        }
        if image is not None:
            frames = grp["frame"].astype(int).to_numpy()
            ys = np.clip(np.rint(coords[:, 0]).astype(int), 0, image.shape[-2] - 1)
            xs = np.clip(np.rint(coords[:, 1]).astype(int), 0, image.shape[-1] - 1)
            vals = image[frames, ys, xs] if image.ndim == 3 else image[ys, xs]
            row["mean_intensity"] = float(vals.mean())
            row["max_intensity"] = float(vals.max())
            row["intensity_std"] = float(vals.std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
```

### Filter pipeline (FilterPanel)

State held by the panel:

- `_source_layer: napari.layers.Layer | None` — the layer the user picked.
- `_source_df: pd.DataFrame | None` — copy of `_source_layer.metadata["properties"]`.
- `_table_model: _PropertyTableModel`, `_proxy: _PropertyTableProxy`.
- `_filter_ranges: dict[str, tuple[float, float]]` — current per-property ranges.
- `_show_props: dict[str, bool]` — checkbox state per property.
- `_sibling_layer: napari.layers.Layer | None`.

On every `_refresh()` (debounced):

1. Build the filter mask from `_filter_ranges`:
   ```python
   mask = pd.Series(True, index=self._source_df.index)
   for col, (lo, hi) in self._filter_ranges.items():
       mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
   filtered_df = self._source_df[mask].reset_index(drop=True)
   ```
2. Update the proxy's filter map (the proxy delegates to the same `_filter_ranges` dict; calling `invalidateFilter()` triggers a table redraw).
3. Update the histogram (re-bin the post-filter values for each checked property).
4. Update the status label.
5. Update the sibling layer (if `Show filtered in viewer` is checked) — see below.

### Sibling layer (used for both Points and Tracks)

- Sibling name is `f"{source.name} (filtered)"`.
- **Points sibling**: `viewer.add_points(filtered_df[["frame", "y", "x"]].to_numpy(), name=sibling_name, size=source.size, face_color="cyan")`. If the source is 2D (no frame), use `["y", "x"]`.
- **Tracks sibling**: rebuild a Tracks-layer-shaped `(N, 4)` array `(track_id, t, y, x)` from the underlying `tracked_df` filtered to the kept `track_id`s. The TracksWidget hands the `tracked_df` to the FilterPanel via `metadata["tracked_df"]` on the Tracks layer (added by Tracking on save). `viewer.add_tracks(tracks_array, name=sibling_name)`.
- **Update strategy:** if the sibling already exists, update `.data` in place; otherwise create.
- **Removal:**
  - On `Show filtered in viewer` unchecked → remove sibling.
  - On source layer removed (events.removed) → remove sibling.
  - On user manually removing the sibling → uncheck the checkbox silently.

## 4. Error Handling & Edge Cases

All operational errors are surfaced in a status label inside the panel. No popups. Critical compute errors (e.g. malformed tracked_df) propagate up to the host widget which already calls `show_error`.

**FilterPanel:**

1. No matching layers in viewer → `"No <Points|Tracks> layers in the viewer."` Sliders/table/histogram hidden.
2. Selected layer has no `metadata["properties"]` → status `"Selected layer has no properties metadata. Run Detection / Tracking to populate it."` Sliders/table/histogram hidden.
3. `metadata["properties"]` is not a `pd.DataFrame` → `"Properties metadata is not a DataFrame; ignored."` Treated as #2.
4. DataFrame has 0 rows → sliders skipped, table empty, status `"No rows to filter."`
5. DataFrame has only id_column and spatial coords → `"No filterable properties on this layer."` Table still works.
6. Single-value column → static label, no slider, takes no part in filtering.
7. Reset Filters → restores all slider ranges to column min/max.
8. Export with 0 matching rows → write header-only CSV, status warning.
9. Source layer removed → clear panel state, run combo refresh.

**TrackingWidget — image source:**

- Image combo always visible. Empty selection is valid → no intensity columns.
- If the Image layer's T-axis doesn't span the tracked frames, `compute_track_properties` raises `ValueError`; the widget catches it, calls `show_error`, and re-runs the property compute with `image=None` so a usable Tracks layer + filter still appear.

**Track property compute:**

- Length-1 tracks → `total_displacement = net_displacement = mean_speed = 0.0`. With image, `intensity_std = 0.0`.
- OOB coords → clipped to image edges via `np.clip`.
- Empty input → empty DataFrame with the right columns (no rows).

**Detection — regionprops merge:**

- Multiple Points layers exist when mask-gen completes → merge into the layer that was the target of the mask-gen run (tracked via `self._mask_target_points`).
- Coord rounding mismatch → merge on `np.rint(...).astype(int)` of `(frame, y, x)`. Rows that don't match are simply dropped from the merge (left-join, missing values become NaN — handled by FilterPanel as "constant" or numeric-with-NaN sliders).
- Mask gen run before any Points layer existed → regionprops stays only on the mask layer; user can re-run Detection. No automatic recovery.

**Sibling layer lifecycle (already in §3, restated for completeness):**

- Created on first filter change while the checkbox is checked.
- Updated in place on subsequent filter changes.
- Removed when checkbox unchecked, source removed, or user manually deletes it.

## 5. Testing

### `_tests/test_track_properties.py` (pure-Python, no Qt)

1. `test_geometric_only_no_image` — three known tracks, verify length / total_displacement / net_displacement / mean_speed.
2. `test_single_frame_track` — length-1 tracks → all motion stats `0.0`.
3. `test_with_image_intensity` — synthetic 3D image with known intensities → mean/max/std verified.
4. `test_image_2d_fallback` — 2D image works (no time index).
5. `test_image_dim_mismatch_raises` — `image.shape[0] <= max(frame)` → `ValueError`.
6. `test_oob_coords_clipped` — coords outside image → no IndexError.
7. `test_returns_sorted_by_track_id`.
8. `test_empty_input` — empty DataFrame → empty result with correct columns.
9. `test_columns_no_image` — column set is `{track_id, length, total_displacement, net_displacement, mean_speed}`.
10. `test_columns_with_image` — adds `{mean_intensity, max_intensity, intensity_std}`.

### `_tests/test_filter_panel.py` (with `pytest-qt`)

*Construction & model:*

1. `test_construction_no_layers` — instantiate `FilterPanel(viewer, "Points", "spot_id")`; status reads "No Points layers".
2. `test_layer_with_properties_populates_sliders` — Points layer with `metadata["properties"]` having `length` and `intensity` → 2 sliders, ranges = column min/max.
3. `test_layer_without_properties_metadata_disables_sliders` — friendly status, sliders/table/histogram hidden.
4. `test_id_column_and_spatial_coords_excluded` — DataFrame has `spot_id, frame, y, x, intensity` → only `intensity` slider.
5. `test_constant_column_skipped` — single-value column → static label, no slider.

*Filtering:*

6. `test_slider_movement_filters_table` — set slider to a sub-range → proxy.rowCount() drops appropriately. Bypass debounce by calling `_refresh` directly.
7. `test_status_label_reports_filtered_count`.
8. `test_reset_button_restores_full_range`.

*Sibling layer:*

9. `test_sibling_layer_created_when_checkbox_checked`.
10. `test_sibling_layer_removed_when_checkbox_unchecked`.
11. `test_sibling_layer_data_shrinks_on_filter`.
12. `test_sibling_removed_when_source_removed`.

*Histogram & export:*

13. `test_histogram_redraws_on_property_toggle` — uncheck a property → that artist removed from axes; check → re-added.
14. `test_export_csv_writes_filtered_rows` — monkeypatch `QFileDialog`, write to `tmp_path`, assert column set + filtered row count.

### `_tests/test_filter_integration.py` (end-to-end, with `pytest-qt`)

1. `test_detection_widget_populates_points_properties` — synthetic image through `DetectionWorker` (or stubbed); Points layer's `metadata["properties"]` is a DataFrame with the expected columns.
2. `test_detection_widget_filter_panel_shows_intensity_slider` — after detection, the embedded filter has one slider (`intensity`).
3. `test_mask_generation_merges_regionprops_into_points` — after mask gen, the Points layer's `metadata["properties"]` has the regionprops columns.
4. `test_tracking_widget_populates_tracks_properties_no_image` — track without image → only geometric columns.
5. `test_tracking_widget_populates_tracks_properties_with_image` — track with image → includes intensity columns.
6. `test_tracking_widget_filter_panel_filters_tracks` — move a slider in the embedded filter → sibling Tracks layer appears with reduced count.

### Out of scope

- Pixel-level matplotlib output of histograms.
- Visual confirmation of sibling-layer rendering (manual smoke step).
- Performance under 10k+ rows.
