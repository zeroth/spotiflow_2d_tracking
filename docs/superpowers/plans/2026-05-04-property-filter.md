# Property Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `FilterPanel` component embedded in both Detection and Tracking widgets. Filters are auto-generated from the active layer's `metadata["properties"]` DataFrame; sliders, a normalized-overlay histogram, and a filtered table all stay in sync; a sibling `(filtered)` layer mirrors the post-filter rows in the napari viewer.

**Architecture:** A pure-Python helper (`compute_track_properties`) computes per-track properties at tracking time. A reusable `FilterPanel(QWidget)` reads the source layer's `metadata["properties"]` DataFrame, auto-generates a `superqt.QLabeledRangeSlider` per numeric column, drives a `QSortFilterProxyModel`-backed table view, redraws an overlaid histogram with per-property show/hide checkboxes, and maintains a `<source> (filtered)` sibling layer. The Detection and Tracking widgets each get a `QScrollArea` wrapper and embed one `FilterPanel`.

**Tech Stack:** Python 3.9+, qtpy, superqt (`QLabeledRangeSlider`, `QLabeledDoubleRangeSlider`), matplotlib (Qt-Agg backend), pandas, numpy, napari, pytest, pytest-qt.

**Spec:** [docs/superpowers/specs/2026-05-04-property-filter-design.md](../specs/2026-05-04-property-filter-design.md)

**Environment:** Project uses `.venv` at `d:/Code/Lab/spotiflow_tracking/.venv/`. Always invoke `".venv/Scripts/python.exe" -m pytest <args>` from the project root. Do **NOT** modify `conftest.py` (root) or create `_tests/conftest.py`. The pre-existing torch/spotiflow stubs in `conftest.py` are deliberate.

---

## Task 1: Scaffold modules and add superqt dependency

**Files:**
- Modify: `pyproject.toml`
- Create: `src/napari_spotiflow_tracking/_mpl_theme.py`
- Create: `src/napari_spotiflow_tracking/_track_properties.py`
- Create: `src/napari_spotiflow_tracking/_filter_panel.py`
- Create: `src/napari_spotiflow_tracking/_tests/test_track_properties.py`
- Create: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`
- Create: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Add superqt to pyproject.toml dependencies**

In `pyproject.toml`, find the `[project] dependencies` block and insert `"superqt",` between `"qtpy",` and `"pandas",`:

```toml
dependencies = [
    "numpy",
    "spotiflow>=0.6.0",
    "qtpy",
    "superqt",
    "pandas",
    "trackpy",
    "matplotlib",
    "scikit-image",
    "scipy",
]
```

- [ ] **Step 2: Reinstall package**

Run: `".venv/Scripts/python.exe" -m pip install -e .`
Expected: superqt already satisfied (transitive via napari), package reinstalled successfully.

- [ ] **Step 3: Create `_mpl_theme.py` (still empty constants only — body filled in Task 2)**

```python
"""Shared napari dark-theme styling for embedded matplotlib figures."""

from __future__ import annotations

from matplotlib.figure import Figure

NAPARI_BG = "#262930"
NAPARI_FG = "#d6d6d6"
LINE_COLOR = "#e0e0e0"


def style_figure(fig: Figure) -> None:
    """Apply napari dark-theme colors to a matplotlib Figure and its axes."""
    fig.patch.set_facecolor(NAPARI_BG)
    for ax in fig.axes:
        ax.set_facecolor(NAPARI_BG)
        for spine in ax.spines.values():
            spine.set_color(NAPARI_FG)
        ax.tick_params(colors=NAPARI_FG)
        ax.xaxis.label.set_color(NAPARI_FG)
        ax.yaxis.label.set_color(NAPARI_FG)
        ax.title.set_color(NAPARI_FG)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(NAPARI_BG)
            legend.get_frame().set_edgecolor(NAPARI_FG)
            for text in legend.get_texts():
                text.set_color(NAPARI_FG)
```

- [ ] **Step 4: Create `_track_properties.py` with stub**

```python
"""Per-track property computation for the Tracks layer's filter panel.

Pure-Python; no Qt imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute per-track geometric (and optionally intensity) properties.

    Input ``tracked_df`` must have columns: frame (int), y (float), x (float),
    particle (int) — the output of ``track_blobs``.

    When ``image`` is given (2D (Y,X) or 3D (T,Y,X)), intensity stats are
    sampled at each track vertex's nearest integer coordinates.

    Returns a DataFrame with one row per track, sorted by ``track_id``.
    """
    raise NotImplementedError
```

- [ ] **Step 5: Create `_filter_panel.py` with class skeleton**

```python
"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget


class FilterPanel(QWidget):
    """Reusable filter panel embedded in Detection and Tracking widgets."""

    selectionChanged = Signal(int)  # row's id_column value when user clicks

    def __init__(
        self,
        viewer: napari.Viewer,
        layer_kind: Literal["Points", "Tracks"],
        id_column: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._viewer = viewer
        self._layer_kind = layer_kind
        self._id_column = id_column
```

- [ ] **Step 6: Create three empty test files**

Create `src/napari_spotiflow_tracking/_tests/test_track_properties.py`:

```python
"""Tests for compute_track_properties pure-Python helper."""
```

Create `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
"""Tests for FilterPanel."""
```

Create `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
"""End-to-end tests for FilterPanel embedded in Detection and Tracking widgets."""
```

- [ ] **Step 7: Verify tests collect without error**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/ --collect-only -q`
Expected: no errors. Pre-existing test count remains the same; the 3 new files are empty.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/napari_spotiflow_tracking/_mpl_theme.py src/napari_spotiflow_tracking/_track_properties.py src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_track_properties.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py
git commit -m "feat(filter): scaffold filter panel, track properties, and shared mpl theme"
```

---

## Task 2: Refactor `_style_figure` from PAPA into shared `_mpl_theme.py`

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`

The new `_mpl_theme.style_figure` already has the body (Task 1 step 3). This task just wires PAPA to use it without changing test behavior. PAPA's class constants are kept and mirrored from the shared module so existing `widget.NAPARI_BG` test assertions still pass.

- [ ] **Step 1: Run pre-existing PAPA tests to confirm green baseline**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py -v`
Expected: 29 passed (existing baseline).

- [ ] **Step 2: Update `_papa_widget.py` to use the shared module**

In `src/napari_spotiflow_tracking/_papa_widget.py`, find the imports near the top and add:

```python
from napari_spotiflow_tracking._mpl_theme import (
    LINE_COLOR as _SHARED_LINE_COLOR,
    NAPARI_BG as _SHARED_BG,
    NAPARI_FG as _SHARED_FG,
    style_figure,
)
```

Find the constants and `_style_figure` classmethod:

```python
    NAPARI_BG = "#262930"
    NAPARI_FG = "#d6d6d6"
    LINE_COLOR = "#e0e0e0"

    @classmethod
    def _style_figure(cls, fig):
        fig.patch.set_facecolor(cls.NAPARI_BG)
        for ax in fig.axes:
            ax.set_facecolor(cls.NAPARI_BG)
            for spine in ax.spines.values():
                spine.set_color(cls.NAPARI_FG)
            ax.tick_params(colors=cls.NAPARI_FG)
            ax.xaxis.label.set_color(cls.NAPARI_FG)
            ax.yaxis.label.set_color(cls.NAPARI_FG)
            ax.title.set_color(cls.NAPARI_FG)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(cls.NAPARI_BG)
                legend.get_frame().set_edgecolor(cls.NAPARI_FG)
                for text in legend.get_texts():
                    text.set_color(cls.NAPARI_FG)
```

Replace with:

```python
    NAPARI_BG = _SHARED_BG
    NAPARI_FG = _SHARED_FG
    LINE_COLOR = _SHARED_LINE_COLOR

    @classmethod
    def _style_figure(cls, fig):
        style_figure(fig)
```

- [ ] **Step 3: Run PAPA tests to confirm green**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py -v`
Expected: 29 passed.

- [ ] **Step 4: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py
git commit -m "refactor(papa): use shared _mpl_theme.style_figure"
```

---

## Task 3: Implement `compute_track_properties` (TDD)

**Files:**
- Modify: `src/napari_spotiflow_tracking/_track_properties.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_track_properties.py`

- [ ] **Step 1: Write failing tests**

Replace contents of `src/napari_spotiflow_tracking/_tests/test_track_properties.py`:

```python
"""Tests for compute_track_properties pure-Python helper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._track_properties import compute_track_properties


def _track_df(rows):
    return pd.DataFrame(rows, columns=["frame", "y", "x", "particle"])


class TestGeometricOnly:
    def test_three_known_tracks(self):
        # Track 0: 5 frames straight line, 1 px right per frame
        # Track 1: 3 frames diagonal, sqrt(2) px per frame
        # Track 2: 1 frame (length-1)
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 1.0, 0],
            [2, 0.0, 2.0, 0],
            [3, 0.0, 3.0, 0],
            [4, 0.0, 4.0, 0],
            [0, 5.0, 5.0, 1],
            [1, 6.0, 6.0, 1],
            [2, 7.0, 7.0, 1],
            [10, 10.0, 10.0, 2],
        ])
        out = compute_track_properties(df)
        # Sorted by track_id
        assert out["track_id"].tolist() == [0, 1, 2]
        # Track 0: length=5, total=4, net=4, mean_speed=1
        assert out.loc[out["track_id"] == 0, "length"].iloc[0] == 5
        assert out.loc[out["track_id"] == 0, "total_displacement"].iloc[0] == pytest.approx(4.0)
        assert out.loc[out["track_id"] == 0, "net_displacement"].iloc[0] == pytest.approx(4.0)
        assert out.loc[out["track_id"] == 0, "mean_speed"].iloc[0] == pytest.approx(1.0)
        # Track 1: 3 frames, sqrt(2) per step, total = 2*sqrt(2)
        assert out.loc[out["track_id"] == 1, "length"].iloc[0] == 3
        assert out.loc[out["track_id"] == 1, "total_displacement"].iloc[0] == pytest.approx(2 * np.sqrt(2))
        assert out.loc[out["track_id"] == 1, "net_displacement"].iloc[0] == pytest.approx(2 * np.sqrt(2))
        assert out.loc[out["track_id"] == 1, "mean_speed"].iloc[0] == pytest.approx(np.sqrt(2))

    def test_single_frame_track_zero_motion(self):
        df = _track_df([[0, 5.0, 5.0, 0]])
        out = compute_track_properties(df)
        assert out["length"].iloc[0] == 1
        assert out["total_displacement"].iloc[0] == 0.0
        assert out["net_displacement"].iloc[0] == 0.0
        assert out["mean_speed"].iloc[0] == 0.0

    def test_columns_no_image(self):
        df = _track_df([[0, 0.0, 0.0, 0]])
        out = compute_track_properties(df)
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }

    def test_returns_sorted_by_track_id(self):
        df = _track_df([
            [0, 0.0, 0.0, 5],
            [0, 0.0, 0.0, 1],
            [0, 0.0, 0.0, 3],
        ])
        out = compute_track_properties(df)
        assert out["track_id"].tolist() == [1, 3, 5]

    def test_empty_input(self):
        df = _track_df([])
        out = compute_track_properties(df)
        assert len(out) == 0
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }


class TestWithImage:
    def test_image_intensity_3d(self):
        # 3D image: image[t, y, x] = t * 100 + y * 10 + x
        image = np.zeros((3, 10, 10), dtype=float)
        for t in range(3):
            for y in range(10):
                for x in range(10):
                    image[t, y, x] = t * 100 + y * 10 + x
        # Track 0: stays at (0, 0) for 3 frames -> values 0, 100, 200
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 0.0, 0],
            [2, 0.0, 0.0, 0],
        ])
        out = compute_track_properties(df, image=image)
        assert out["mean_intensity"].iloc[0] == pytest.approx(100.0)
        assert out["max_intensity"].iloc[0] == pytest.approx(200.0)
        assert out["intensity_std"].iloc[0] == pytest.approx(np.std([0, 100, 200]))

    def test_image_2d_fallback(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        # Track at (2, 3) and (4, 5)
        df = _track_df([
            [0, 2.0, 3.0, 0],
            [1, 4.0, 5.0, 0],
        ])
        out = compute_track_properties(df, image=image)
        # Sampling on 2D image: image[2, 3] = 23, image[4, 5] = 45
        assert out["mean_intensity"].iloc[0] == pytest.approx((23 + 45) / 2)

    def test_image_dim_mismatch_raises(self):
        image = np.zeros((2, 10, 10))  # only 2 frames
        df = _track_df([
            [0, 0.0, 0.0, 0],
            [1, 0.0, 0.0, 0],
            [5, 0.0, 0.0, 0],  # frame 5 out of range
        ])
        with pytest.raises(ValueError, match="T-axis"):
            compute_track_properties(df, image=image)

    def test_oob_coords_clipped(self):
        # Track has coords way outside image bounds
        image = np.ones((1, 5, 5), dtype=float)
        df = _track_df([[0, -10.0, 100.0, 0]])
        out = compute_track_properties(df, image=image)
        # Clipped coord still samples valid pixel
        assert out["mean_intensity"].iloc[0] == 1.0

    def test_columns_with_image(self):
        image = np.ones((1, 5, 5))
        df = _track_df([[0, 0.0, 0.0, 0]])
        out = compute_track_properties(df, image=image)
        assert set(out.columns) == {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
            "mean_intensity", "max_intensity", "intensity_std",
        }

    def test_invalid_image_ndim_raises(self):
        image = np.zeros((3, 3, 3, 3))  # 4D
        df = _track_df([[0, 0.0, 0.0, 0]])
        with pytest.raises(ValueError, match="2D.*3D"):
            compute_track_properties(df, image=image)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_track_properties.py -v`
Expected: all tests fail with `NotImplementedError`.

- [ ] **Step 3: Implement `compute_track_properties`**

Replace the `compute_track_properties` body in `src/napari_spotiflow_tracking/_track_properties.py`:

```python
def compute_track_properties(
    tracked_df: pd.DataFrame,
    image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute per-track geometric (and optionally intensity) properties.

    Input ``tracked_df`` must have columns: frame (int), y (float), x (float),
    particle (int) — the output of ``track_blobs``.

    When ``image`` is given (2D (Y,X) or 3D (T,Y,X)), intensity stats are
    sampled at each track vertex's nearest integer coordinates.

    Returns a DataFrame with one row per track, sorted by ``track_id``.
    """
    cols = ["track_id", "length", "total_displacement",
            "net_displacement", "mean_speed"]
    if image is not None:
        cols += ["mean_intensity", "max_intensity", "intensity_std"]

    if len(tracked_df) == 0:
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
        if length > 1:
            diffs = np.diff(coords, axis=0)
            total = float(np.linalg.norm(diffs, axis=1).sum())
            net = float(np.linalg.norm(coords[-1] - coords[0]))
        else:
            total = 0.0
            net = 0.0
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

    return pd.DataFrame(rows, columns=cols).sort_values("track_id").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_track_properties.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_track_properties.py src/napari_spotiflow_tracking/_tests/test_track_properties.py
git commit -m "feat(filter): add compute_track_properties helper"
```

---

## Task 4: FilterPanel — construction shell + layer combo

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing construction tests**

Replace contents of `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
"""Tests for FilterPanel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def filter_panel_points(make_napari_viewer):
    from napari_spotiflow_tracking._filter_panel import FilterPanel
    viewer = make_napari_viewer()
    panel = FilterPanel(viewer, "Points", "spot_id")
    return panel, viewer


@pytest.fixture
def filter_panel_tracks(make_napari_viewer):
    from napari_spotiflow_tracking._filter_panel import FilterPanel
    viewer = make_napari_viewer()
    panel = FilterPanel(viewer, "Tracks", "track_id")
    return panel, viewer


class TestFilterPanelConstruction:
    def test_constructs_without_error_no_layers(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert panel is not None

    def test_status_when_no_layers(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert "no points layer" in panel._status_label.text().lower()

    def test_layer_combo_starts_empty(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert panel._layer_combo.count() == 0

    def test_layer_combo_lists_points(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 1
        assert panel._layer_combo.itemText(0) == "spots"

    def test_layer_combo_ignores_tracks(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_tracks(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]), name="t")
        assert panel._layer_combo.count() == 0

    def test_tracks_panel_lists_tracks(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        viewer.add_tracks(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]), name="t")
        assert panel._layer_combo.count() == 1

    def test_tracks_panel_ignores_points(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 0

    def test_combo_updates_on_layer_remove(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 1
        viewer.layers.remove("spots")
        assert panel._layer_combo.count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestFilterPanelConstruction -v`
Expected: failures (`AttributeError` on `_status_label`, `_layer_combo`).

- [ ] **Step 3: Implement construction shell**

Replace contents of `src/napari_spotiflow_tracking/_filter_panel.py`:

```python
"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
import napari.layers
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FilterPanel(QWidget):
    """Reusable filter panel embedded in Detection and Tracking widgets."""

    selectionChanged = Signal(int)

    def __init__(
        self,
        viewer: napari.Viewer,
        layer_kind: Literal["Points", "Tracks"],
        id_column: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._viewer = viewer
        self._layer_kind = layer_kind
        self._id_column = id_column

        self._setup_ui()
        self._connect_events()
        self._refresh_layer_combo()
        self._update_status()

    def _layer_class(self):
        if self._layer_kind == "Points":
            return napari.layers.Points
        return napari.layers.Tracks

    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        group = QGroupBox("Filter")
        layout = QVBoxLayout()
        group.setLayout(layout)
        outer.addWidget(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        row.addWidget(self._layer_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layer_combo)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

    def _connect_events(self):
        self._viewer.layers.events.inserted.connect(self._on_layer_event)
        self._viewer.layers.events.removed.connect(self._on_layer_event)

    def _on_layer_event(self, event=None):
        self._refresh_layer_combo()

    def _refresh_layer_combo(self):
        prev = self._layer_combo.currentText()
        self._layer_combo.clear()
        cls = self._layer_class()
        for layer in self._viewer.layers:
            if isinstance(layer, cls):
                self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(prev)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._update_status()

    def _on_layer_changed(self, name: str):
        self._update_status()

    def _update_status(self):
        if self._layer_combo.count() == 0:
            self._status_label.setText(f"No {self._layer_kind} layers in the viewer.")
        else:
            self._status_label.setText("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestFilterPanelConstruction -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): FilterPanel construction shell with layer combo"
```

---

## Task 5: FilterPanel — auto-generated range sliders

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests for slider generation**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
def _add_points_with_props(viewer, name, df):
    layer = viewer.add_points(df[["frame", "y", "x"]].to_numpy(), name=name)
    layer.metadata["properties"] = df
    return layer


class TestSliderGeneration:
    def test_no_metadata_disables_sliders(self, filter_panel_points):
        panel, viewer = filter_panel_points
        # Layer with no metadata
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert "no properties metadata" in panel._status_label.text().lower()
        assert len(panel._sliders) == 0

    def test_metadata_not_dataframe(self, filter_panel_points):
        panel, viewer = filter_panel_points
        layer = viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        layer.metadata["properties"] = {"not": "a df"}
        # Force re-read by reselecting the layer
        panel._on_layer_changed("spots")
        assert "not a dataframe" in panel._status_label.text().lower()

    def test_creates_slider_per_filterable_column(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "intensity": [10.0, 20.0, 30.0],
            "area": [100, 200, 300],
        })
        _add_points_with_props(viewer, "spots", df)
        # spot_id, frame, y, x are excluded -> intensity + area = 2 sliders
        assert set(panel._sliders.keys()) == {"intensity", "area"}

    def test_excludes_id_and_spatial_columns(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        df = pd.DataFrame({
            "track_id": [0, 1],
            "length": [10, 20],
        })
        # Use a real Tracks layer
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        assert set(panel._sliders.keys()) == {"length"}

    def test_constant_column_gets_static_label_not_slider(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "category": [1, 1, 1],  # constant
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert "category" not in panel._sliders
        assert "intensity" in panel._sliders
        assert "category" in panel._constant_labels

    def test_slider_range_matches_column_min_max(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        slider = panel._sliders["intensity"]
        lo, hi = slider.value()
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(30.0)

    def test_integer_column_uses_int_slider(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        from superqt import QLabeledRangeSlider
        df = pd.DataFrame({"track_id": [0, 1], "length": [5, 25]})
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        assert isinstance(panel._sliders["length"], QLabeledRangeSlider)

    def test_float_column_uses_float_slider(self, filter_panel_points):
        panel, viewer = filter_panel_points
        from superqt import QLabeledDoubleRangeSlider
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [1.0, 2.0],
            "x": [3.0, 4.0],
            "intensity": [1.5, 2.5],
        })
        _add_points_with_props(viewer, "spots", df)
        assert isinstance(panel._sliders["intensity"], QLabeledDoubleRangeSlider)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestSliderGeneration -v`
Expected: failures (no `_sliders`, no `_constant_labels`).

- [ ] **Step 3: Implement slider generation**

In `src/napari_spotiflow_tracking/_filter_panel.py`:

Add to imports at the top:

```python
import pandas as pd
from qtpy.QtCore import Qt
from superqt import QLabeledDoubleRangeSlider, QLabeledRangeSlider
```

Replace the `_setup_ui` method to add a sliders container, and replace `_on_layer_changed`. The full updated class body should look like the patch below — apply by replacing only the relevant sections.

Find:

```python
    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        group = QGroupBox("Filter")
        layout = QVBoxLayout()
        group.setLayout(layout)
        outer.addWidget(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        row.addWidget(self._layer_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layer_combo)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
```

Replace with:

```python
    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        group = QGroupBox("Filter")
        layout = QVBoxLayout()
        group.setLayout(layout)
        outer.addWidget(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        row.addWidget(self._layer_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layer_combo)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._sliders: dict = {}
        self._constant_labels: dict = {}
        self._source_df: pd.DataFrame | None = None
        self._source_layer = None
        self._filter_ranges: dict = {}
```

Replace `_on_layer_changed`:

```python
    def _on_layer_changed(self, name: str):
        self._clear_sliders()
        self._source_df = None
        self._source_layer = None
        self._filter_ranges = {}

        if not name or name not in self._viewer.layers:
            self._update_status()
            return

        layer = self._viewer.layers[name]
        if not isinstance(layer, self._layer_class()):
            self._update_status()
            return

        props = layer.metadata.get("properties")
        if props is None:
            self._status_label.setText(
                "Selected layer has no properties metadata. "
                "Run Detection / Tracking to populate it."
            )
            self._sliders_group.setVisible(False)
            return
        if not isinstance(props, pd.DataFrame):
            self._status_label.setText(
                "Properties metadata is not a DataFrame; ignored."
            )
            self._sliders_group.setVisible(False)
            return

        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._update_status()

    def _clear_sliders(self):
        for slider in self._sliders.values():
            slider.setParent(None)
        for label in self._constant_labels.values():
            label.setParent(None)
        self._sliders = {}
        self._constant_labels = {}
        # Remove any leftover children from layout
        while self._sliders_layout.count():
            item = self._sliders_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _is_filterable_column(self, col: str) -> bool:
        if col == self._id_column:
            return False
        if self._layer_kind == "Points" and col in {"frame", "y", "x"}:
            return False
        if self._layer_kind == "Tracks" and col in {"t", "y", "x", "track_id"}:
            return False
        return True

    def _build_sliders(self, df: pd.DataFrame):
        for col in df.columns:
            if not self._is_filterable_column(col):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{col}: constant {col_min}"))
                container = QWidget()
                container.setLayout(row)
                self._sliders_layout.addWidget(container)
                self._constant_labels[col] = container
                continue

            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            if pd.api.types.is_integer_dtype(df[col]):
                slider = QLabeledRangeSlider(Qt.Horizontal)
                slider.setRange(int(col_min), int(col_max))
                slider.setValue((int(col_min), int(col_max)))
            else:
                slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
                slider.setRange(float(col_min), float(col_max))
                slider.setValue((float(col_min), float(col_max)))
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._sliders_layout.addWidget(container)
            self._sliders[col] = slider
            self._filter_ranges[col] = (col_min, col_max)

    def _update_status(self):
        if self._layer_combo.count() == 0:
            self._status_label.setText(f"No {self._layer_kind} layers in the viewer.")
        elif self._source_df is None:
            # Layer chosen but no valid metadata; status was already set by
            # _on_layer_changed.
            pass
        elif len(self._sliders) == 0 and len(self._constant_labels) == 0:
            self._status_label.setText("No filterable properties on this layer.")
        elif len(self._sliders) == 0:
            self._status_label.setText("All filterable columns are constant.")
        else:
            n_total = len(self._source_df)
            self._status_label.setText(f"{n_total} / {n_total} rows match")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 16 passed (8 from Task 4 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): auto-generate range sliders per numeric property"
```

---

## Task 6: FilterPanel — table model + proxy + filter pipeline

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests for filter pipeline + table**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
class TestFilterPipeline:
    def test_initial_table_shows_all_rows(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert panel._proxy.rowCount() == 5

    def test_slider_range_filters_rows(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Set intensity range to [25, 45] -> rows with 30 and 40 match -> 2 rows
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()  # bypass debounce
        assert panel._proxy.rowCount() == 2

    def test_filter_status_label(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(10)),
            "frame": list(range(10)),
            "y": [0.0] * 10,
            "x": [0.0] * 10,
            "intensity": list(np.linspace(0, 100, 10)),
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((50.0, 100.0))
        panel._refresh()
        # 5 of 10 rows have intensity >= 50
        assert "5 / 10" in panel._status_label.text()

    def test_table_columns_match_source_df(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        model = panel._table_model
        assert model.columnCount() == len(df.columns)

    def test_selection_changed_signal_emits_id(self, filter_panel_points, qtbot):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [10, 20, 30],
            "frame": [0, 1, 2],
            "y": [0.0] * 3,
            "x": [0.0] * 3,
            "intensity": [1.0, 2.0, 3.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Select the second row in the proxy
        with qtbot.waitSignal(panel.selectionChanged, timeout=500) as blocker:
            proxy_index = panel._proxy.index(1, 0)
            panel._table_view.selectionModel().select(
                proxy_index,
                panel._table_view.selectionModel().Select
                | panel._table_view.selectionModel().Rows,
            )
            panel._on_table_selection_changed(proxy_index, proxy_index)
        assert blocker.args == [20]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestFilterPipeline -v`
Expected: failures (`_proxy`, `_refresh`, `_table_model`, `_table_view`, etc. don't exist yet).

- [ ] **Step 3: Implement model, proxy, table view, and filter pipeline**

In `src/napari_spotiflow_tracking/_filter_panel.py`:

Update imports near the top to add:

```python
from qtpy.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    QTimer,
    QVariant,
)
from qtpy.QtWidgets import QHeaderView, QTableView
```

Add two private classes immediately after the imports and before `class FilterPanel`:

```python
class _PropertyTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame = pd.DataFrame()

    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True).copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        if role == Qt.UserRole + 1:
            return float(self._df.iat[index.row(), index.column()])
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            if 0 <= section < self._df.shape[1]:
                return str(self._df.columns[section])
        return str(section)


class _PropertyTableProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ranges: dict = {}

    def setRanges(self, ranges: dict):
        self._ranges = dict(ranges)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None or not self._ranges:
            return True
        for col, (lo, hi) in self._ranges.items():
            col_idx = list(model._df.columns).index(col)
            val = model._df.iat[source_row, col_idx]
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v < lo or v > hi:
                return False
        return True
```

In `_setup_ui`, after the sliders group setup and before the status label, add a table view section:

Find:

```python
        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
```

Replace with:

```python
        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        self._table_model = _PropertyTableModel(self)
        self._proxy = _PropertyTableProxy(self)
        self._proxy.setSourceModel(self._table_model)
        self._table_view = QTableView()
        self._table_view.setModel(self._proxy)
        self._table_view.setSelectionMode(QTableView.SingleSelection)
        self._table_view.setSelectionBehavior(QTableView.SelectRows)
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        layout.addWidget(self._table_view)

        sel_model = self._table_view.selectionModel()
        sel_model.currentRowChanged.connect(self._on_table_selection_changed)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh)
```

In `_on_layer_changed`, after the `self._build_sliders(props)` call and before `self._update_status()`, add:

Find:

```python
        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._update_status()
```

Replace with:

```python
        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._table_model.setDataFrame(props)
        self._proxy.setRanges({})
        self._update_status()
```

In `_build_sliders`, wire each slider's `valueChanged` to schedule the refresh debounce. After the `self._sliders[col] = slider` line, add:

Find:

```python
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._sliders_layout.addWidget(container)
            self._sliders[col] = slider
            self._filter_ranges[col] = (col_min, col_max)
```

Replace with:

```python
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._sliders_layout.addWidget(container)
            self._sliders[col] = slider
            self._filter_ranges[col] = (col_min, col_max)
            slider.valueChanged.connect(
                lambda v, c=col: self._on_slider_changed(c, v)
            )
```

Add new methods at the end of the class:

```python
    def _on_slider_changed(self, col: str, value):
        lo, hi = value
        self._filter_ranges[col] = (lo, hi)
        self._refresh_timer.start()

    def _refresh(self):
        if self._source_df is None or len(self._sliders) == 0:
            self._update_status()
            return

        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)

        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

    def _on_table_selection_changed(self, current, previous):
        if not current.isValid():
            return
        source_index = self._proxy.mapToSource(current)
        if not source_index.isValid():
            return
        col_idx = list(self._table_model._df.columns).index(self._id_column)
        row_id = int(self._table_model._df.iat[source_index.row(), col_idx])
        self.selectionChanged.emit(row_id)
```

Note: existing tests already use `panel._refresh()` and `panel._proxy` — both are now defined.

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 21 passed (16 from before + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): table model + proxy + debounced filter pipeline"
```

---

## Task 7: FilterPanel — Reset and Export buttons

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
class TestResetAndExport:
    def test_reset_restores_full_range(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((25.0, 35.0))
        panel._refresh()
        assert panel._proxy.rowCount() == 1

        panel._reset_btn.click()
        # After reset, slider returns to full range and table shows all
        assert panel._proxy.rowCount() == 5
        lo, hi = panel._sliders["intensity"].value()
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(50.0)

    def test_export_writes_filtered_csv(self, filter_panel_points, tmp_path, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()

        out_path = tmp_path / "filtered.csv"
        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV files (*.csv)"),
        )
        panel._export_btn.click()

        assert out_path.exists()
        out_df = pd.read_csv(out_path)
        assert list(out_df.columns) == list(df.columns)
        assert len(out_df) == 2
        assert out_df["intensity"].tolist() == [30.0, 40.0]

    def test_export_zero_rows_writes_header_only(self, filter_panel_points, tmp_path, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((100.0, 200.0))  # no rows match
        panel._refresh()

        out_path = tmp_path / "empty.csv"
        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV files (*.csv)"),
        )
        panel._export_btn.click()

        out_df = pd.read_csv(out_path)
        assert len(out_df) == 0
        assert list(out_df.columns) == list(df.columns)

    def test_export_cancel_no_op(self, filter_panel_points, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0],
            "frame": [0],
            "y": [0.0],
            "x": [0.0],
            "intensity": [10.0],
        })
        _add_points_with_props(viewer, "spots", df)

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        # Should not raise
        panel._export_btn.click()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestResetAndExport -v`
Expected: failures (`_reset_btn`, `_export_btn` don't exist).

- [ ] **Step 3: Implement Reset and Export buttons**

In `src/napari_spotiflow_tracking/_filter_panel.py`:

Add `QFileDialog` to the qtpy imports:

```python
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)
```

In `_setup_ui`, after the status label and before the refresh timer setup, add a buttons row:

Find:

```python
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh)
```

Replace with:

```python
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset Filters")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        btn_row.addWidget(self._reset_btn)
        self._export_btn = QPushButton("Export Filtered CSV")
        self._export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh)
```

Add the two methods at the end of the class:

```python
    def _on_reset_clicked(self):
        if self._source_df is None:
            return
        for col, slider in self._sliders.items():
            col_min = self._source_df[col].min()
            col_max = self._source_df[col].max()
            if pd.api.types.is_integer_dtype(self._source_df[col]):
                slider.setValue((int(col_min), int(col_max)))
                self._filter_ranges[col] = (int(col_min), int(col_max))
            else:
                slider.setValue((float(col_min), float(col_max)))
                self._filter_ranges[col] = (float(col_min), float(col_max))
        self._refresh()

    def _on_export_clicked(self):
        if self._source_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save filtered CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)
        filtered.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 25 passed (21 from before + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): Reset Filters and Export CSV buttons"
```

---

## Task 8: FilterPanel — overlay histogram with property checkboxes

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
class TestHistogram:
    def test_checkbox_per_property(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
            "area": [100, 200, 300, 400, 500],
        })
        _add_points_with_props(viewer, "spots", df)
        assert set(panel._hist_checkboxes.keys()) == {"intensity", "area"}
        assert all(cb.isChecked() for cb in panel._hist_checkboxes.values())

    def test_histogram_has_one_axes(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert len(panel._hist_fig.axes) == 1

    def test_unchecking_property_removes_artist(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [0.0] * 3,
            "x": [0.0] * 3,
            "intensity": [10.0, 20.0, 30.0],
            "area": [100, 200, 300],
        })
        _add_points_with_props(viewer, "spots", df)
        # Force redraw
        panel._refresh()
        ax = panel._hist_fig.axes[0]
        n_patches_before = len(ax.patches)
        assert n_patches_before > 0

        # Uncheck intensity
        panel._hist_checkboxes["intensity"].setChecked(False)
        ax_after = panel._hist_fig.axes[0]
        n_patches_after = len(ax_after.patches)
        # At least the intensity patches are gone, area still there
        assert n_patches_after < n_patches_before
        assert n_patches_after > 0  # area still drawn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestHistogram -v`
Expected: failures (no `_hist_checkboxes`, `_hist_fig`).

- [ ] **Step 3: Implement histogram**

In `src/napari_spotiflow_tracking/_filter_panel.py`, add to imports:

```python
import matplotlib.cm as mpl_cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QCheckBox, QScrollArea

from napari_spotiflow_tracking._mpl_theme import style_figure
```

In `_setup_ui`, after the `self._sliders_group = sliders_group` line and before the table model setup, add the histogram section:

Find:

```python
        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        self._table_model = _PropertyTableModel(self)
```

Replace with:

```python
        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        hist_group = QGroupBox("Histogram")
        hist_outer = QVBoxLayout()
        hist_group.setLayout(hist_outer)

        cb_scroll = QScrollArea()
        cb_scroll.setWidgetResizable(True)
        cb_scroll.setMaximumHeight(80)
        cb_inner = QWidget()
        self._hist_checkbox_layout = QVBoxLayout()
        cb_inner.setLayout(self._hist_checkbox_layout)
        cb_scroll.setWidget(cb_inner)
        hist_outer.addWidget(cb_scroll)

        self._hist_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._hist_canvas = FigureCanvas(self._hist_fig)
        self._hist_canvas.setFixedHeight(220)
        hist_outer.addWidget(self._hist_canvas)

        layout.addWidget(hist_group)
        self._hist_group = hist_group
        self._hist_checkboxes: dict = {}
        self._hist_colors: dict = {}

        self._table_model = _PropertyTableModel(self)
```

In `_clear_sliders` (which clears the sliders layout on layer change), also clear histogram checkboxes. Find:

```python
    def _clear_sliders(self):
        for slider in self._sliders.values():
            slider.setParent(None)
        for label in self._constant_labels.values():
            label.setParent(None)
        self._sliders = {}
        self._constant_labels = {}
        # Remove any leftover children from layout
        while self._sliders_layout.count():
            item = self._sliders_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
```

Replace with:

```python
    def _clear_sliders(self):
        for slider in self._sliders.values():
            slider.setParent(None)
        for label in self._constant_labels.values():
            label.setParent(None)
        self._sliders = {}
        self._constant_labels = {}
        while self._sliders_layout.count():
            item = self._sliders_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        for cb in self._hist_checkboxes.values():
            cb.setParent(None)
        self._hist_checkboxes = {}
        self._hist_colors = {}
        while self._hist_checkbox_layout.count():
            item = self._hist_checkbox_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
```

In `_build_sliders`, after the slider for each column is added, also add a histogram checkbox. Find the existing slider loop and replace with:

```python
    def _build_sliders(self, df: pd.DataFrame):
        cmap = mpl_cm.get_cmap("tab10")
        color_idx = 0
        for col in df.columns:
            if not self._is_filterable_column(col):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{col}: constant {col_min}"))
                container = QWidget()
                container.setLayout(row)
                self._sliders_layout.addWidget(container)
                self._constant_labels[col] = container
                continue

            color = cmap(color_idx % 10)
            color_idx += 1
            self._hist_colors[col] = color

            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            if pd.api.types.is_integer_dtype(df[col]):
                slider = QLabeledRangeSlider(Qt.Horizontal)
                slider.setRange(int(col_min), int(col_max))
                slider.setValue((int(col_min), int(col_max)))
            else:
                slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
                slider.setRange(float(col_min), float(col_max))
                slider.setValue((float(col_min), float(col_max)))
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._sliders_layout.addWidget(container)
            self._sliders[col] = slider
            self._filter_ranges[col] = (col_min, col_max)
            slider.valueChanged.connect(
                lambda v, c=col: self._on_slider_changed(c, v)
            )

            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.toggled.connect(self._redraw_histogram)
            self._hist_checkbox_layout.addWidget(cb)
            self._hist_checkboxes[col] = cb
```

In `_refresh`, call `_redraw_histogram()` at the end. Find:

```python
    def _refresh(self):
        if self._source_df is None or len(self._sliders) == 0:
            self._update_status()
            return

        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)

        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")
```

Replace with:

```python
    def _refresh(self):
        if self._source_df is None or len(self._sliders) == 0:
            self._update_status()
            return

        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)

        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

        self._redraw_histogram(filtered=filtered)
```

Add the new method:

```python
    def _redraw_histogram(self, filtered: pd.DataFrame | None = None, *_):
        if self._source_df is None or self._hist_fig is None:
            return
        if filtered is None:
            mask = pd.Series(True, index=self._source_df.index)
            for col, (lo, hi) in self._filter_ranges.items():
                mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
            filtered = self._source_df[mask].reset_index(drop=True)

        self._hist_fig.clear()
        ax = self._hist_fig.add_subplot(111)
        ax.set_xlabel("normalized [0, 1]")
        ax.set_ylabel("count")

        any_drawn = False
        for col, slider in self._sliders.items():
            cb = self._hist_checkboxes.get(col)
            if cb is None or not cb.isChecked():
                continue
            col_min = self._source_df[col].min()
            col_max = self._source_df[col].max()
            if col_max == col_min:
                continue
            values = filtered[col].to_numpy() if col in filtered.columns else np.array([])
            if values.size == 0:
                continue
            normed = (values - col_min) / (col_max - col_min)
            color = self._hist_colors.get(col, "tab:blue")
            ax.hist(
                normed, bins=30, range=(0, 1),
                histtype="stepfilled", alpha=0.5, color=color,
                label=f"{col}  [{col_min:.3g} - {col_max:.3g}]",
            )
            any_drawn = True

        if any_drawn:
            ax.legend(loc="upper right", fontsize=8)
        style_figure(self._hist_fig)
        self._hist_canvas.draw_idle()
```

Also call `_redraw_histogram` from `_on_layer_changed` after the `_table_model.setDataFrame(props)` line:

Find:

```python
        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._table_model.setDataFrame(props)
        self._proxy.setRanges({})
        self._update_status()
```

Replace with:

```python
        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._table_model.setDataFrame(props)
        self._proxy.setRanges({})
        self._redraw_histogram()
        self._update_status()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 28 passed (25 from before + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): overlay histogram with per-property show/hide checkboxes"
```

---

## Task 9: FilterPanel — sibling layer for Points

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
class TestSiblingLayerPoints:
    def test_checkbox_unchecked_no_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [10.0, 20.0],
            "x": [10.0, 20.0],
            "intensity": [1.0, 2.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Default unchecked
        assert panel._show_filtered_cb.isChecked() is False
        assert all("(filtered)" not in l.name for l in viewer.layers)

    def test_check_creates_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [10.0, 20.0, 30.0],
            "x": [10.0, 20.0, 30.0],
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        names = [l.name for l in viewer.layers]
        assert "spots (filtered)" in names

    def test_filter_shrinks_sibling_data(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2, 3, 4],
            "frame": [0, 1, 2, 3, 4],
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()
        sibling = viewer.layers["spots (filtered)"]
        assert len(sibling.data) == 2

    def test_uncheck_removes_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        assert "spots (filtered)" in [l.name for l in viewer.layers]
        panel._show_filtered_cb.setChecked(False)
        assert "spots (filtered)" not in [l.name for l in viewer.layers]

    def test_source_removed_removes_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0],
            "frame": [0],
            "y": [0.0],
            "x": [0.0],
            "intensity": [10.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        assert "spots (filtered)" in [l.name for l in viewer.layers]
        viewer.layers.remove("spots")
        assert "spots (filtered)" not in [l.name for l in viewer.layers]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestSiblingLayerPoints -v`
Expected: failures (`_show_filtered_cb` not defined).

- [ ] **Step 3: Implement Points sibling layer**

In `src/napari_spotiflow_tracking/_filter_panel.py`:

In `_setup_ui`, before the buttons row, add a checkbox:

Find:

```python
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset Filters")
```

Replace with:

```python
        self._show_filtered_cb = QCheckBox("Show filtered in viewer")
        self._show_filtered_cb.setChecked(False)
        self._show_filtered_cb.toggled.connect(self._on_show_filtered_toggled)
        layout.addWidget(self._show_filtered_cb)

        self._sibling_layer = None

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset Filters")
```

In `_refresh`, after computing `filtered`, also update sibling:

Find:

```python
        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

        self._redraw_histogram(filtered=filtered)
```

Replace with:

```python
        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

        self._redraw_histogram(filtered=filtered)
        if self._show_filtered_cb.isChecked():
            self._update_sibling_layer(filtered)
```

Add new methods to the class:

```python
    def _on_show_filtered_toggled(self, checked: bool):
        if not checked:
            self._remove_sibling_layer()
            return
        if self._source_df is None:
            return
        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)
        self._update_sibling_layer(filtered)

    def _sibling_name(self) -> str:
        if self._source_layer is None:
            return ""
        return f"{self._source_layer.name} (filtered)"

    def _remove_sibling_layer(self):
        name = self._sibling_name()
        if not name:
            return
        if name in self._viewer.layers:
            try:
                self._viewer.layers.remove(name)
            except (KeyError, ValueError):
                pass
        self._sibling_layer = None

    def _update_sibling_layer(self, filtered: pd.DataFrame):
        if self._source_layer is None:
            return
        name = self._sibling_name()
        if self._layer_kind == "Points":
            if "frame" in filtered.columns:
                data = filtered[["frame", "y", "x"]].to_numpy()
            else:
                data = filtered[["y", "x"]].to_numpy()
            if name in self._viewer.layers:
                self._viewer.layers[name].data = data
                self._sibling_layer = self._viewer.layers[name]
            else:
                self._sibling_layer = self._viewer.add_points(
                    data, name=name, size=3, face_color="cyan",
                )
        elif self._layer_kind == "Tracks":
            self._update_tracks_sibling(filtered, name)

    def _update_tracks_sibling(self, filtered: pd.DataFrame, name: str):
        # Implemented in Task 10
        pass
```

In `__init__`, after `self._connect_events()`, add a per-layer-removed listener so when source is removed, we drop the sibling. Update `_on_layer_event` to also call `_check_source_removed`:

Find:

```python
    def _on_layer_event(self, event=None):
        self._refresh_layer_combo()
```

Replace with:

```python
    def _on_layer_event(self, event=None):
        self._check_source_removed()
        self._refresh_layer_combo()

    def _check_source_removed(self):
        if self._source_layer is None:
            return
        try:
            still_in_viewer = self._source_layer in self._viewer.layers
        except Exception:
            still_in_viewer = False
        if not still_in_viewer:
            self._remove_sibling_layer()
            self._source_layer = None
            self._source_df = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 33 passed (28 + 5).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): sibling layer for filtered Points"
```

---

## Task 10: FilterPanel — sibling layer for Tracks

**Files:**
- Modify: `src/napari_spotiflow_tracking/_filter_panel.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_panel.py`:

```python
class TestSiblingLayerTracks:
    def _add_tracks_with_props(self, viewer, name, tracked_df, meta_df):
        # tracked_df: columns frame, y, x, particle (output of track_blobs)
        # Build (N, 4) napari Tracks data: (track_id, t, y, x)
        rows = tracked_df.sort_values(["particle", "frame"])[
            ["particle", "frame", "y", "x"]
        ].to_numpy()
        layer = viewer.add_tracks(rows, name=name)
        layer.metadata["properties"] = meta_df
        layer.metadata["tracked_df"] = tracked_df
        return layer

    def test_check_creates_tracks_sibling(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1, 2],
            "y": [0.0, 0.0, 5.0, 5.0, 5.0],
            "x": [0.0, 0.0, 5.0, 5.0, 5.0],
            "particle": [1, 1, 2, 2, 2],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2], "length": [2, 3]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._show_filtered_cb.setChecked(True)
        assert "t (filtered)" in [l.name for l in viewer.layers]

    def test_filter_keeps_only_long_tracks(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1, 2],
            "y": [0.0, 0.0, 5.0, 5.0, 5.0],
            "x": [0.0, 0.0, 5.0, 5.0, 5.0],
            "particle": [1, 1, 2, 2, 2],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2], "length": [2, 3]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._show_filtered_cb.setChecked(True)
        # Filter to length >= 3 -> only track 2 (3 vertices)
        panel._sliders["length"].setValue((3, 3))
        panel._refresh()
        sibling = viewer.layers["t (filtered)"]
        assert len(sibling.data) == 3
        assert all(row[0] == 2 for row in sibling.data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py::TestSiblingLayerTracks -v`
Expected: failures (sibling Tracks layer not implemented).

- [ ] **Step 3: Implement Tracks sibling**

In `src/napari_spotiflow_tracking/_filter_panel.py`, replace the stub `_update_tracks_sibling`:

Find:

```python
    def _update_tracks_sibling(self, filtered: pd.DataFrame, name: str):
        # Implemented in Task 10
        pass
```

Replace with:

```python
    def _update_tracks_sibling(self, filtered: pd.DataFrame, name: str):
        tracked_df = self._source_layer.metadata.get("tracked_df")
        if not isinstance(tracked_df, pd.DataFrame):
            return
        kept_ids = filtered[self._id_column].astype(int).tolist()
        if not kept_ids:
            data = np.empty((0, 4))
        else:
            sub = tracked_df[tracked_df["particle"].isin(kept_ids)]
            sub = sub.sort_values(["particle", "frame"])
            data = sub[["particle", "frame", "y", "x"]].to_numpy()

        if name in self._viewer.layers:
            try:
                self._viewer.layers.remove(name)
            except (KeyError, ValueError):
                pass

        if len(data) == 0:
            self._sibling_layer = None
            return
        self._sibling_layer = self._viewer.add_tracks(data, name=name)
```

Note: napari's Tracks layer doesn't easily support in-place data replacement of varying shapes; rebuilding the layer is the simplest reliable approach.

Also add `import numpy as np` to the imports at the top of the file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_panel.py -v`
Expected: 35 passed (33 + 2).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_filter_panel.py src/napari_spotiflow_tracking/_tests/test_filter_panel.py
git commit -m "feat(filter): sibling layer for filtered Tracks"
```

---

## Task 11: Detection widget — populate Points metadata at detect time

**Files:**
- Modify: `src/napari_spotiflow_tracking/_workers.py`
- Modify: `src/napari_spotiflow_tracking/_segmentation.py`
- Modify: `src/napari_spotiflow_tracking/_detection_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Write failing integration test**

Replace contents of `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
"""End-to-end tests for FilterPanel embedded in Detection and Tracking widgets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking._detection_widget import DetectionWidget
    viewer = make_napari_viewer()
    widget = DetectionWidget(viewer)
    return widget, viewer


class TestDetectionPointsMetadata:
    def test_detect_finished_populates_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Synthetic 2-frame stack: 1 spot per frame
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([100.0, 200.0])
        # Call _on_detection_finished directly with our synthetic data
        widget._on_detection_finished(points, None, intensities)

        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        assert len(layers) == 1
        props = layers[0].metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        assert list(props.columns) == ["spot_id", "frame", "y", "x", "intensity"]
        assert props["intensity"].tolist() == [100.0, 200.0]
        assert props["spot_id"].tolist() == [0, 1]

    def test_detect_finished_2d_points_no_frame(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[5.0, 5.0], [6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)
        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        props = layers[0].metadata.get("properties")
        assert list(props.columns) == ["spot_id", "y", "x", "intensity"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestDetectionPointsMetadata -v`
Expected: failures (`_on_detection_finished` only takes 2 positional args, not 3).

- [ ] **Step 3: Extend `detect_spots` to also return intensities**

In `src/napari_spotiflow_tracking/_segmentation.py`, find:

```python
def detect_spots(
    image_2d: np.ndarray,
    model,
    prob_thresh: float | None = 0.5,
    min_distance: int = 2,
) -> tuple[np.ndarray, object]:
    """Detect spots in a single 2D image using a loaded Spotiflow model.

    Returns (points, details) where points is (N, 2) array of [y, x].
    """
    points, details = model.predict(
        image_2d,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        verbose=False,
    )
    return points, details
```

(Leave it unchanged; we'll extract intensities at the worker level — `details.intens` is already returned.)

In `src/napari_spotiflow_tracking/_segmentation.py`, find `detect_spots_log` and add a helper to sample intensities for LoG:

After `detect_spots_log`, add:

```python
def sample_point_intensities(image_2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample image intensity at each point's nearest integer coords."""
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    ys = np.clip(np.rint(points[:, 0]).astype(int), 0, image_2d.shape[0] - 1)
    xs = np.clip(np.rint(points[:, 1]).astype(int), 0, image_2d.shape[1] - 1)
    return image_2d[ys, xs].astype(float)
```

- [ ] **Step 4: Update DetectionWorker to deliver intensities**

In `src/napari_spotiflow_tracking/_workers.py`, find:

```python
class DetectionWorker(QThread):
    """Background worker for spot detection + Gaussian fitting.

    Handles both single 2D images and T,Y,X stacks.
    Supports Spotiflow and LoG detection methods.
    """

    progress = Signal(str, int, int)  # (stage, current, total)
    finished = Signal(object, object)  # (all_points, all_masks)
    errored = Signal(str)
```

Replace with:

```python
class DetectionWorker(QThread):
    """Background worker for spot detection + Gaussian fitting.

    Handles both single 2D images and T,Y,X stacks.
    Supports Spotiflow and LoG detection methods.
    """

    progress = Signal(str, int, int)  # (stage, current, total)
    finished = Signal(object, object, object)  # (all_points, all_masks, all_intensities)
    errored = Signal(str)
```

Find `_process_frame`:

```python
    def _process_frame(self, frame_image: np.ndarray):
        """Detect spots and optionally fit Gaussians for a single 2D frame."""
        if self._remove_bg:
            frame_image = remove_background_high_pass(frame_image, sigma=self._bg_sigma)

        if self._method == "spotiflow":
            points, _ = detect_spots(
                frame_image, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )
        else:
            points = detect_spots_log(
                frame_image,
                min_sigma=self._log_min_sigma,
                max_sigma=self._log_max_sigma,
                num_sigma=self._log_num_sigma,
                threshold=self._log_threshold,
            )

        if self._generate_mask:
            result = fit_and_mask_2d(
                frame_image, points,
                patch_radius=self._patch_radius,
                fallback_radius=self._fallback_radius,
                backend=self._fit_backend,
            )
            return points, result.mask
        return points, None
```

Replace with:

```python
    def _process_frame(self, frame_image: np.ndarray):
        """Detect spots and optionally fit Gaussians for a single 2D frame."""
        if self._remove_bg:
            frame_image = remove_background_high_pass(frame_image, sigma=self._bg_sigma)

        if self._method == "spotiflow":
            points, details = detect_spots(
                frame_image, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )
            try:
                intens = np.asarray(details.intens, dtype=float).ravel()
            except AttributeError:
                intens = sample_point_intensities(frame_image, points)
        else:
            points = detect_spots_log(
                frame_image,
                min_sigma=self._log_min_sigma,
                max_sigma=self._log_max_sigma,
                num_sigma=self._log_num_sigma,
                threshold=self._log_threshold,
            )
            intens = sample_point_intensities(frame_image, points)

        if self._generate_mask:
            result = fit_and_mask_2d(
                frame_image, points,
                patch_radius=self._patch_radius,
                fallback_radius=self._fallback_radius,
                backend=self._fit_backend,
            )
            return points, result.mask, intens
        return points, None, intens
```

Add to imports at the top of `_workers.py`:

```python
from napari_spotiflow_tracking._segmentation import detect_spots, detect_spots_log, sample_point_intensities
```

(Replace the existing `from napari_spotiflow_tracking._segmentation import detect_spots, detect_spots_log` line.)

In `run`, find:

```python
    def run(self):
        try:
            if self._image.ndim == 2:
                self.progress.emit("Detecting spots", 0, 1)
                points, mask = self._process_frame(self._image)
                self.progress.emit("Detecting spots", 1, 1)
                self.finished.emit(points, mask)

            elif self._image.ndim == 3:
                n_frames = self._image.shape[0]
                all_points = []
                all_masks = []

                for t in range(n_frames):
                    self.progress.emit("Detecting spots", t + 1, n_frames)
                    points, mask = self._process_frame(self._image[t])

                    if len(points) > 0:
                        frame_col = np.full((len(points), 1), t)
                        points_with_frame = np.hstack([frame_col, points])
                        all_points.append(points_with_frame)

                    if mask is not None:
                        all_masks.append(mask)

                if all_points:
                    combined_points = np.vstack(all_points)
                else:
                    combined_points = np.empty((0, 3))
                combined_masks = np.stack(all_masks, axis=0) if all_masks else None

                self.finished.emit(combined_points, combined_masks)
            else:
                self.errored.emit(
                    f"Expected 2D or 3D (T,Y,X) image, got ndim={self._image.ndim}"
                )
        except Exception:
            self.errored.emit(traceback.format_exc())
```

Replace with:

```python
    def run(self):
        try:
            if self._image.ndim == 2:
                self.progress.emit("Detecting spots", 0, 1)
                points, mask, intens = self._process_frame(self._image)
                self.progress.emit("Detecting spots", 1, 1)
                self.finished.emit(points, mask, intens)

            elif self._image.ndim == 3:
                n_frames = self._image.shape[0]
                all_points = []
                all_masks = []
                all_intens = []

                for t in range(n_frames):
                    self.progress.emit("Detecting spots", t + 1, n_frames)
                    points, mask, intens = self._process_frame(self._image[t])

                    if len(points) > 0:
                        frame_col = np.full((len(points), 1), t)
                        points_with_frame = np.hstack([frame_col, points])
                        all_points.append(points_with_frame)
                        all_intens.append(intens)

                    if mask is not None:
                        all_masks.append(mask)

                if all_points:
                    combined_points = np.vstack(all_points)
                    combined_intens = np.concatenate(all_intens)
                else:
                    combined_points = np.empty((0, 3))
                    combined_intens = np.empty((0,))
                combined_masks = np.stack(all_masks, axis=0) if all_masks else None

                self.finished.emit(combined_points, combined_masks, combined_intens)
            else:
                self.errored.emit(
                    f"Expected 2D or 3D (T,Y,X) image, got ndim={self._image.ndim}"
                )
        except Exception:
            self.errored.emit(traceback.format_exc())
```

- [ ] **Step 5: Update Detection widget to consume intensities**

In `src/napari_spotiflow_tracking/_detection_widget.py`, add `pandas as pd` import (already present per current file).

Find:

```python
    def _on_detection_finished(self, points: np.ndarray, masks):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._detect_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._cancel_detect_btn.setEnabled(False)
        self._last_points = points
        self._export_btn.setEnabled(len(points) > 0)
        n_spots = len(points)
        self._status_label.setText(f"Detected {n_spots} spots.")
        show_info(f"Done — detected {n_spots} spots.")

        if n_spots > 0:
            self.viewer.add_points(
                points,
                name="Detected Spots",
                size=3,
                face_color="red",
            )
        if masks is not None:
            self.viewer.add_labels(masks, name="Spot Masks", opacity=0.4)
```

Replace with:

```python
    def _on_detection_finished(self, points: np.ndarray, masks, intensities=None):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._detect_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._cancel_detect_btn.setEnabled(False)
        self._last_points = points
        self._export_btn.setEnabled(len(points) > 0)
        n_spots = len(points)
        self._status_label.setText(f"Detected {n_spots} spots.")
        show_info(f"Done — detected {n_spots} spots.")

        if intensities is None:
            intensities = np.zeros(n_spots, dtype=float)

        if n_spots > 0:
            new_layer = self.viewer.add_points(
                points,
                name="Detected Spots",
                size=3,
                face_color="red",
            )
            spot_id = np.arange(n_spots, dtype=int)
            if points.shape[1] == 3:
                df = pd.DataFrame({
                    "spot_id": spot_id,
                    "frame": points[:, 0].astype(int),
                    "y": points[:, 1].astype(float),
                    "x": points[:, 2].astype(float),
                    "intensity": np.asarray(intensities, dtype=float),
                })
            else:
                df = pd.DataFrame({
                    "spot_id": spot_id,
                    "y": points[:, 0].astype(float),
                    "x": points[:, 1].astype(float),
                    "intensity": np.asarray(intensities, dtype=float),
                })
            new_layer.metadata["properties"] = df
        if masks is not None:
            self.viewer.add_labels(masks, name="Spot Masks", opacity=0.4)
```

- [ ] **Step 6: Run integration test to verify it passes**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestDetectionPointsMetadata -v`
Expected: 2 passed.

- [ ] **Step 7: Run pre-existing detection widget tests to confirm no regression**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_detection_widget.py -v`
Expected: all pre-existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/napari_spotiflow_tracking/_workers.py src/napari_spotiflow_tracking/_segmentation.py src/napari_spotiflow_tracking/_detection_widget.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py
git commit -m "feat(detection): populate Points metadata.properties at detect time"
```

---

## Task 12: Detection widget — regionprops merge at mask-gen time + remove primitive filter

**Files:**
- Modify: `src/napari_spotiflow_tracking/_detection_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Write failing test for regionprops merge**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
class TestMaskGenMerge:
    def test_merge_into_existing_points_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Pre-populate a Points layer with detection-style metadata
        points_data = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        points_layer = viewer.add_points(points_data, name="Detected Spots")
        points_layer.metadata["properties"] = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "intensity": [10.0, 20.0],
        })
        widget._mask_target_points = points_layer

        # Simulate mask-gen finishing with regionprops including new columns
        mask = np.zeros((2, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "area": [10, 20],
            "mean_intensity": [50.0, 60.0],
        })
        widget._on_mask_finished(mask, props_df)

        merged = points_layer.metadata["properties"]
        # Old columns retained
        assert "spot_id" in merged.columns
        assert "intensity" in merged.columns
        # New columns added
        assert "area" in merged.columns
        assert "mean_intensity" in merged.columns
        # Values merged correctly
        assert merged.loc[merged["spot_id"] == 0, "area"].iloc[0] == 10
        assert merged.loc[merged["spot_id"] == 1, "mean_intensity"].iloc[0] == 60.0

    def test_merge_when_target_has_no_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Points layer without properties metadata (user-imported)
        points_data = np.array([[0, 5.0, 5.0]])
        points_layer = viewer.add_points(points_data, name="My Points")
        widget._mask_target_points = points_layer

        mask = np.zeros((1, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0],
            "y": [5.0],
            "x": [5.0],
            "area": [42],
        })
        widget._on_mask_finished(mask, props_df)

        out = points_layer.metadata["properties"]
        assert isinstance(out, pd.DataFrame)
        assert "spot_id" in out.columns
        assert "area" in out.columns
        assert out["area"].iloc[0] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestMaskGenMerge -v`
Expected: failures (no `_mask_target_points`, mask-finish doesn't merge into Points).

- [ ] **Step 3: Track which Points layer mask-gen targets**

In `src/napari_spotiflow_tracking/_detection_widget.py`, in `__init__`, add:

```python
        self._mask_target_points: napari.layers.Points | None = None
```

In `_generate_mask_from_points`, find:

```python
        points_layer = self.viewer.layers[points_name]
        image_layer = self.viewer.layers[image_name]
        points_data = np.asarray(points_layer.data)
        image_data = np.asarray(image_layer.data)
```

Replace with:

```python
        points_layer = self.viewer.layers[points_name]
        image_layer = self.viewer.layers[image_name]
        points_data = np.asarray(points_layer.data)
        image_data = np.asarray(image_layer.data)
        self._mask_target_points = points_layer
```

- [ ] **Step 4: Update `_on_mask_finished` to merge regionprops into the target Points layer**

Find:

```python
    def _on_mask_finished(self, mask, props_df):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_mask_btn.setEnabled(False)

        mask_layer = self.viewer.add_labels(
            np.asarray(mask), name="Spot Masks", opacity=0.4,
        )
        # Store regionprops in layer metadata for downstream filtering
        mask_layer.metadata["regionprops"] = props_df

        n_regions = len(props_df)
        show_info(f"Done — mask generated ({n_regions} regions with properties).")
        self._status_label.setText(f"Mask generated ({n_regions} regions).")
```

Replace with:

```python
    def _on_mask_finished(self, mask, props_df):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_mask_btn.setEnabled(False)

        mask_layer = self.viewer.add_labels(
            np.asarray(mask), name="Spot Masks", opacity=0.4,
        )
        mask_layer.metadata["regionprops"] = props_df

        target = self._mask_target_points
        if target is not None and target in self.viewer.layers:
            self._merge_regionprops_into_points(target, props_df)

        n_regions = len(props_df)
        show_info(f"Done — mask generated ({n_regions} regions with properties).")
        self._status_label.setText(f"Mask generated ({n_regions} regions).")

    def _merge_regionprops_into_points(self, points_layer, props_df: pd.DataFrame):
        existing = points_layer.metadata.get("properties")
        if isinstance(existing, pd.DataFrame):
            key_cols = ["frame", "y", "x"] if "frame" in existing.columns else ["y", "x"]
            left = existing.copy()
            right = props_df.copy()
            for k in key_cols:
                if k in left.columns:
                    left[f"_k_{k}"] = np.rint(left[k]).astype(int)
                if k in right.columns:
                    right[f"_k_{k}"] = np.rint(right[k]).astype(int)
            join_keys = [f"_k_{k}" for k in key_cols if k in existing.columns and k in props_df.columns]
            drop_right_cols = [c for c in key_cols if c in right.columns]
            merged = left.merge(
                right.drop(columns=drop_right_cols),
                on=join_keys, how="left",
            ).drop(columns=join_keys)
            points_layer.metadata["properties"] = merged
        else:
            new_df = props_df.copy()
            new_df.insert(0, "spot_id", np.arange(len(new_df), dtype=int))
            points_layer.metadata["properties"] = new_df
```

- [ ] **Step 5: Remove the primitive Spot Filtering UI section**

In `src/napari_spotiflow_tracking/_detection_widget.py`, find the entire Spot Filtering section in `_setup_ui` (lines roughly 207-259):

```python
        # ── Spot Filtering ────────────────────────────────────────────
        filter_group = QGroupBox("Spot Filtering")
        filter_layout = QVBoxLayout()

        filter_layout.addWidget(QLabel(
            "Filter spots by region properties (from mask metadata)."
        ))

        # Labels layer selector
        filter_mask_row = QHBoxLayout()
        filter_mask_row.addWidget(QLabel("Mask layer:"))
        self._filter_mask_combo = QComboBox()
        filter_mask_row.addWidget(self._filter_mask_combo)
        filter_layout.addLayout(filter_mask_row)

        # Min/max area
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Area:"))
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 100000)
        self._min_area.setValue(0)
        self._min_area.setPrefix("min: ")
        area_row.addWidget(self._min_area)
        self._max_area = QSpinBox()
        self._max_area.setRange(0, 100000)
        self._max_area.setValue(100000)
        self._max_area.setPrefix("max: ")
        area_row.addWidget(self._max_area)
        filter_layout.addLayout(area_row)

        # Min/max mean intensity
        intens_row = QHBoxLayout()
        intens_row.addWidget(QLabel("Mean intensity:"))
        self._min_intensity = QDoubleSpinBox()
        self._min_intensity.setRange(0, 1e6)
        self._min_intensity.setValue(0)
        self._min_intensity.setPrefix("min: ")
        self._min_intensity.setDecimals(1)
        intens_row.addWidget(self._min_intensity)
        self._max_intensity = QDoubleSpinBox()
        self._max_intensity.setRange(0, 1e6)
        self._max_intensity.setValue(1e6)
        self._max_intensity.setPrefix("max: ")
        self._max_intensity.setDecimals(1)
        intens_row.addWidget(self._max_intensity)
        filter_layout.addLayout(intens_row)

        self._filter_btn = QPushButton("Apply Filter")
        self._filter_btn.clicked.connect(self._apply_filter)
        filter_layout.addWidget(self._filter_btn)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
```

Delete this entire block.

Also remove the `_apply_filter` method body and the `_refresh_filter_mask_combo` method. Find and delete:

```python
    def _refresh_filter_mask_combo(self):
        prev = self._filter_mask_combo.currentText()
        self._filter_mask_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._filter_mask_combo.addItem(layer.name)
        idx = self._filter_mask_combo.findText(prev)
        if idx >= 0:
            self._filter_mask_combo.setCurrentIndex(idx)
```

And:

```python
    # ── Spot Filtering ────────────────────────────────────────────────

    def _apply_filter(self):
        """Filter spots by area and intensity using regionprops from mask metadata."""
        mask_name = self._filter_mask_combo.currentText()
        if not mask_name:
            show_error("No mask layer selected.")
            return

        mask_layer = self.viewer.layers[mask_name]
        props_df = mask_layer.metadata.get("regionprops")
        if props_df is None or len(props_df) == 0:
            show_error("No regionprops in mask metadata. Generate mask first.")
            return

        min_area = self._min_area.value()
        max_area = self._max_area.value()
        min_intens = self._min_intensity.value()
        max_intens = self._max_intensity.value()

        # Filter
        filtered = props_df[
            (props_df["area"] >= min_area)
            & (props_df["area"] <= max_area)
            & (props_df["mean_intensity"] >= min_intens)
            & (props_df["mean_intensity"] <= max_intens)
        ]

        n_before = len(props_df)
        n_after = len(filtered)
        n_removed = n_before - n_after

        # Build filtered Points layer from centroids
        if "frame" in filtered.columns:
            points_data = filtered[["frame", "y", "x"]].to_numpy()
        else:
            points_data = filtered[["y", "x"]].to_numpy()

        self.viewer.add_points(
            points_data,
            name="Filtered Spots",
            size=3,
            face_color="green",
        )

        # Store filtered props in the new points layer metadata
        pts_layers = [l for l in self.viewer.layers if l.name == "Filtered Spots"]
        if pts_layers:
            pts_layers[-1].metadata["regionprops"] = filtered.reset_index(drop=True)

        show_info(f"Filter: kept {n_after}/{n_before} spots (removed {n_removed}).")
        self._status_label.setText(
            f"Filtered: {n_after}/{n_before} spots kept."
        )
```

Also remove the call `self._refresh_filter_mask_combo()` from `_setup_ui` (near the end) and from `_on_layer_change`.

- [ ] **Step 6: Run integration tests**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py -v`
Expected: all integration tests so far pass.

- [ ] **Step 7: Run pre-existing detection widget tests**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_detection_widget.py -v`
Expected: pre-existing tests pass except any that referenced `_apply_filter`, `_filter_mask_combo`, `_min_area`, etc. Those tests should be updated or deleted in this step. Read `test_detection_widget.py` and remove tests that reference the deleted attributes — those tests are now obsolete.

After deletion, re-run:
Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_detection_widget.py -v`
Expected: all remaining pre-existing detection tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/napari_spotiflow_tracking/_detection_widget.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py src/napari_spotiflow_tracking/_tests/test_detection_widget.py
git commit -m "feat(detection): merge regionprops into Points metadata, remove primitive filter"
```

---

## Task 13: Detection widget — embed FilterPanel + QScrollArea wrapper

**Files:**
- Modify: `src/napari_spotiflow_tracking/_detection_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Write failing test**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
class TestDetectionFilterPanelEmbedded:
    def test_widget_has_filter_panel(self, detection_widget):
        widget, viewer = detection_widget
        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panels = widget.findChildren(FilterPanel)
        assert len(panels) == 1
        assert panels[0]._layer_kind == "Points"
        assert panels[0]._id_column == "spot_id"

    def test_filter_panel_picks_up_detected_layer(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)

        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panel = widget.findChildren(FilterPanel)[0]
        # Manually force layer combo refresh (in tests events may be eager)
        panel._refresh_layer_combo()
        panel._on_layer_changed("Detected Spots")
        assert "intensity" in panel._sliders
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestDetectionFilterPanelEmbedded -v`
Expected: failure (no FilterPanel inside DetectionWidget).

- [ ] **Step 3: Wrap layout in QScrollArea and append FilterPanel**

In `src/napari_spotiflow_tracking/_detection_widget.py`, add to imports:

```python
from qtpy.QtWidgets import QScrollArea
from napari_spotiflow_tracking._filter_panel import FilterPanel
```

Find the start of `_setup_ui`:

```python
    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
```

Replace with:

```python
    def _setup_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout()
        inner.setLayout(layout)
        scroll.setWidget(inner)
```

At the end of `_setup_ui`, after the existing `layout.addStretch()` line and the layer-combo refresh calls, append:

Find:

```python
        layout.addStretch()
        self._refresh_image_combo()
        self._refresh_points_combo()
```

Replace with:

```python
        # Embedded property filter
        self._filter_panel = FilterPanel(self.viewer, "Points", "spot_id", parent=self)
        layout.addWidget(self._filter_panel)

        layout.addStretch()
        self._refresh_image_combo()
        self._refresh_points_combo()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestDetectionFilterPanelEmbedded -v`
Expected: 2 passed.

- [ ] **Step 5: Run full detection + integration test sets**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_detection_widget.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py -v`
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/_detection_widget.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py
git commit -m "feat(detection): embed FilterPanel + QScrollArea wrapper"
```

---

## Task 14: Tracking widget — image combo + properties + FilterPanel

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tracking_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
@pytest.fixture
def tracking_widget(make_napari_viewer):
    from napari_spotiflow_tracking._tracking_widget import TrackingWidget
    viewer = make_napari_viewer()
    widget = TrackingWidget(viewer)
    return widget, viewer


class TestTrackingPropertiesAndFilter:
    def _add_points_layer(self, viewer):
        # 6 detections, 2 tracks (3 frames each)
        data = np.array([
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [2, 2.0, 0.0],
            [0, 10.0, 10.0],
            [1, 11.0, 10.0],
            [2, 12.0, 10.0],
        ])
        viewer.add_points(data, name="spots")

    def test_widget_has_image_combo(self, tracking_widget):
        widget, viewer = tracking_widget
        assert widget._image_combo is not None
        assert widget._image_combo.count() == 0  # no image layers initially

    def test_widget_has_filter_panel(self, tracking_widget):
        widget, viewer = tracking_widget
        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panels = widget.findChildren(FilterPanel)
        assert len(panels) == 1
        assert panels[0]._layer_kind == "Tracks"
        assert panels[0]._id_column == "track_id"

    def test_track_no_image_populates_geometric_properties(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        tracks_layers = [l for l in viewer.layers if l.name == "Tracked Spots"]
        assert len(tracks_layers) == 1
        props = tracks_layers[0].metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        assert set(props.columns) >= {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }
        # Intensity columns absent
        assert "mean_intensity" not in props.columns

    def test_track_with_image_includes_intensity(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        viewer.add_image(np.ones((3, 32, 32)), name="img")
        widget._points_combo.setCurrentText("spots")
        widget._image_combo.setCurrentText("img")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        tracks_layers = [l for l in viewer.layers if l.name == "Tracked Spots"]
        props = tracks_layers[0].metadata.get("properties")
        assert "mean_intensity" in props.columns

    def test_tracked_df_stored_on_layer(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()
        tracks_layer = viewer.layers["Tracked Spots"]
        assert isinstance(tracks_layer.metadata.get("tracked_df"), pd.DataFrame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestTrackingPropertiesAndFilter -v`
Expected: failures (no `_image_combo`, no FilterPanel inside TrackingWidget, no metadata properties).

- [ ] **Step 3: Update `_tracking_widget.py`**

Replace the entire `_tracking_widget.py` file with the updated version. Read the current file first, then apply these changes:

(a) Add imports at the top:

```python
from qtpy.QtWidgets import QScrollArea
from napari_spotiflow_tracking._filter_panel import FilterPanel
from napari_spotiflow_tracking._track_properties import compute_track_properties
```

(b) In `_setup_ui`, replace:

```python
    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
```

With:

```python
    def _setup_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout()
        inner.setLayout(layout)
        scroll.setWidget(inner)
```

(c) In the Input QGroupBox in `_setup_ui`, after the existing Points-layer combo row, add an Image combo row. Find:

```python
        # Input points layer
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        row.addWidget(self._points_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_points_combo)
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
```

Replace with:

```python
        # Input layers
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        row.addWidget(self._points_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_points_combo)
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image layer:"))
        self._image_combo = QComboBox()
        img_row.addWidget(self._image_combo)
        img_refresh_btn = QPushButton("↻")
        img_refresh_btn.setFixedWidth(30)
        img_refresh_btn.setToolTip("Refresh layer list (optional, for intensity stats)")
        img_refresh_btn.clicked.connect(self._refresh_image_combo)
        img_row.addWidget(img_refresh_btn)
        input_layout.addLayout(img_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
```

(d) Add `_refresh_image_combo` method and update `_on_layer_change`. Find:

```python
    def _on_layer_change(self, event=None):
        self._refresh_points_combo()

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)
```

Replace with:

```python
    def _on_layer_change(self, event=None):
        self._refresh_points_combo()
        self._refresh_image_combo()

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)

    def _refresh_image_combo(self):
        prev = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        idx = self._image_combo.findText(prev)
        if idx >= 0:
            self._image_combo.setCurrentIndex(idx)
```

(e) In `_run_tracking`, after adding the Tracks layer, compute properties and store on metadata. Find the tail of `_run_tracking`:

```python
        tracks_array = tracks_to_napari(tracked_df)
        n_tracks = tracked_df["particle"].nunique()
        n_frames = int(df["frame"].nunique())

        self.viewer.add_tracks(tracks_array, name="Tracked Spots")
        self._last_tracked_df = tracked_df
        self._export_btn.setEnabled(True)
        msg = f"Found {n_tracks} tracks across {n_frames} frames."
        self._status_label.setText(msg)
        show_info(msg)
```

Replace with:

```python
        tracks_array = tracks_to_napari(tracked_df)
        n_tracks = tracked_df["particle"].nunique()
        n_frames = int(df["frame"].nunique())

        image_name = self._image_combo.currentText()
        image = None
        if image_name and image_name in self.viewer.layers:
            image = np.asarray(self.viewer.layers[image_name].data)

        try:
            track_props = compute_track_properties(tracked_df, image=image)
        except ValueError as exc:
            show_error(f"Track property compute failed: {exc}; falling back to no image.")
            track_props = compute_track_properties(tracked_df, image=None)

        tracks_layer = self.viewer.add_tracks(tracks_array, name="Tracked Spots")
        tracks_layer.metadata["properties"] = track_props
        tracks_layer.metadata["tracked_df"] = tracked_df

        self._last_tracked_df = tracked_df
        self._export_btn.setEnabled(True)
        msg = f"Found {n_tracks} tracks across {n_frames} frames."
        self._status_label.setText(msg)
        show_info(msg)
```

(f) At the end of `_setup_ui`, after the existing `layout.addStretch()` and `self._refresh_points_combo()`, append the FilterPanel and the new image-combo refresh:

Find:

```python
        layout.addStretch()
        self._refresh_points_combo()
```

Replace with:

```python
        # Embedded property filter
        self._filter_panel = FilterPanel(self.viewer, "Tracks", "track_id", parent=self)
        layout.addWidget(self._filter_panel)

        layout.addStretch()
        self._refresh_points_combo()
        self._refresh_image_combo()
```

- [ ] **Step 4: Run failing tests to verify they now pass**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestTrackingPropertiesAndFilter -v`
Expected: 5 passed.

- [ ] **Step 5: Run pre-existing tracking widget tests**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_tracking_widget.py -v`
Expected: pre-existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/_tracking_widget.py src/napari_spotiflow_tracking/_tests/test_filter_integration.py
git commit -m "feat(tracking): image combo + per-track properties + embedded FilterPanel"
```

---

## Task 15: End-to-end filter test through Tracking widget

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`

- [ ] **Step 1: Write the end-to-end test**

Append to `src/napari_spotiflow_tracking/_tests/test_filter_integration.py`:

```python
class TestFilterEndToEndTracking:
    def test_slider_filters_sibling_tracks_layer(self, tracking_widget):
        widget, viewer = tracking_widget
        # 2 tracks: short (length 2) and long (length 4)
        data = np.array([
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [0, 10.0, 10.0],
            [1, 11.0, 10.0],
            [2, 12.0, 10.0],
            [3, 13.0, 10.0],
        ])
        viewer.add_points(data, name="spots")
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panel = widget.findChildren(FilterPanel)[0]
        panel._refresh_layer_combo()
        panel._on_layer_changed("Tracked Spots")
        assert "length" in panel._sliders

        # Check sibling toggle and filter
        panel._show_filtered_cb.setChecked(True)
        # Filter to length >= 4 -> only the long track
        slider = panel._sliders["length"]
        slider.setValue((4, 4))
        panel._refresh()

        sibling = viewer.layers["Tracked Spots (filtered)"]
        # 4 vertices for the single long track
        assert len(sibling.data) == 4
```

- [ ] **Step 2: Run test to verify it passes**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py::TestFilterEndToEndTracking -v`
Expected: 1 passed.

- [ ] **Step 3: Run full integration test set**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/test_filter_integration.py -v`
Expected: all integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/napari_spotiflow_tracking/_tests/test_filter_integration.py
git commit -m "test(filter): end-to-end filter through Tracking widget"
```

---

## Task 16: Final test run + manual smoke test

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

Run: `".venv/Scripts/python.exe" -m pytest src/napari_spotiflow_tracking/_tests/ -v`
Expected: all tests pass — including pre-existing PAPA, detection, tracking, segmentation, and tracking-properties tests, plus the new filter panel and integration tests.

- [ ] **Step 2: Manual napari smoke test**

Run from project root:

```powershell
.venv\Scripts\python.exe -c "import napari; v = napari.Viewer(); v.window.add_plugin_dock_widget('napari-spotiflow-tracking', 'Spot Detection'); v.window.add_plugin_dock_widget('napari-spotiflow-tracking', 'Spot Tracking'); napari.run()"
```

In the napari window:

1. Open a small test image (e.g. `walkAv_SA_n2v.tif` or any `.tif` from the project root).
2. In the Detection widget, run **Detect Spots** (Spotiflow, default model). Confirm the new "Detected Spots" layer is created and the FilterPanel at the bottom of the Detection widget shows an `intensity` slider after picking the Detected Spots layer.
3. Move the intensity slider to a sub-range. Confirm the status label updates (`"N / M rows match"`) and the table view shrinks. Toggle "Show filtered in viewer" — confirm a `"Detected Spots (filtered)"` Points layer appears.
4. Click **Generate Mask from Points**. After completion, re-pick the Detected Spots layer in the FilterPanel combo and confirm new sliders appear (area, mean_intensity, etc. — the merged regionprops).
5. In the Tracking widget, pick the Detected Spots layer, optionally pick the Image layer (for intensity stats), and click **Track**. Confirm the Tracks layer is created, the FilterPanel inside the Tracking widget exposes `length`, `total_displacement`, etc. sliders.
6. Move the `length` slider, toggle "Show filtered in viewer", confirm the `"Tracked Spots (filtered)"` Tracks layer appears with a reduced count.
7. Click **Export Filtered CSV**, save to a temp path, and confirm the CSV has the expected columns and row count.

If anything fails, file a follow-up rather than amending.

---

## Self-Review Notes

- **Spec coverage:**
  - §1 architecture (`_filter_panel.py`, `_track_properties.py`, `_mpl_theme.py`) → Tasks 1, 2, 3, 4-10.
  - §1 detection modifications (regionprops merge, scrollarea, FilterPanel embed) → Tasks 11, 12, 13.
  - §1 tracking modifications (image combo, track properties, FilterPanel embed) → Task 14.
  - §1 PAPA refactor → Task 2.
  - §2 UI layout (sliders, histogram, table, sibling layer, reset/export) → Tasks 4-10.
  - §3 data flow (detect-time props, mask-time merge, track-time props, sibling layer logic) → Tasks 11, 12, 14, plus FilterPanel sibling tasks 9-10.
  - §4 error handling (no layers, missing metadata, constant column, OOB image, image dim mismatch, regionprops merge fallback, sibling lifecycle) → covered across Tasks 3-14 (each task asserts the relevant edge case).
  - §5 testing (3 test files with the specified test names) → Tasks 3 (track_properties), 4-10 (filter_panel), 11-15 (integration).

- **Type consistency:**
  - `compute_track_properties(tracked_df, image=None) -> pd.DataFrame` — same signature in Tasks 1, 3, 14.
  - `FilterPanel(viewer, layer_kind, id_column, parent=None)` — consistent across tasks.
  - `DetectionWorker.finished` signal — extended to 3-arg `(points, masks, intensities)` in Task 11; `_on_detection_finished(points, masks, intensities=None)` accepts the new arg in the same task.
  - `tracks_layer.metadata["properties"]` is a per-track DataFrame; `tracks_layer.metadata["tracked_df"]` is the raw vertex-level DataFrame — used in Tasks 14 (set) and 10 (read by sibling code).
  - `points_layer.metadata["properties"]` columns: `spot_id, [frame,] y, x, intensity, [regionprops...]` — consistent in Tasks 11, 12.

- **No placeholders:** every code-changing step shows the exact code; expected test output is stated; commands have exact form.
