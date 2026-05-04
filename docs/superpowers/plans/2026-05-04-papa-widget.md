# PAPA Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth dock widget, "PAPA", to the `napari-spotiflow-tracking` plugin that consumes a Points layer and renders two interactive matplotlib figures (combined particles-per-frame timeseries + 3-panel ratio analysis) for proximity-assisted photoactivation cycle analysis.

**Architecture:** Pure-Python helpers in `_papa_analysis.py` (no Qt, fully unit-testable) drive a `QTabWidget`-based `PAPAWidget(QWidget)` in `_papa_widget.py`. Two `FigureCanvasQTAgg` instances embed matplotlib figures with `NavigationToolbar2QT` for zoom/pan/save. Hardcoded napari dark-theme styling. Computation is synchronous (groupby + matplotlib draw is fast).

**Tech Stack:** Python 3.9+, qtpy, matplotlib (Qt-Agg backend), pandas, numpy, napari, pytest, pytest-qt.

**Spec:** [docs/superpowers/specs/2026-05-04-papa-widget-design.md](../specs/2026-05-04-papa-widget-design.md)

---

## Task 1: Add matplotlib dependency and create empty module skeletons

**Files:**
- Modify: `pyproject.toml`
- Create: `src/napari_spotiflow_tracking/_papa_analysis.py`
- Create: `src/napari_spotiflow_tracking/_papa_widget.py`
- Create: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`
- Create: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Add matplotlib to pyproject.toml dependencies**

Open `pyproject.toml` and add `"matplotlib",` to the `[project] dependencies` list (between `"trackpy",` and `"scikit-image",`):

```toml
dependencies = [
    "numpy",
    "spotiflow>=0.6.0",
    "qtpy",
    "pandas",
    "trackpy",
    "matplotlib",
    "scikit-image",
    "scipy",
]
```

- [ ] **Step 2: Reinstall package in editable mode to pick up new dep**

Run: `pip install -e .`
Expected: matplotlib already satisfied (transitive via napari/scikit-image), package reinstalled successfully.

- [ ] **Step 3: Create empty `_papa_analysis.py` with stub signatures**

Create `src/napari_spotiflow_tracking/_papa_analysis.py`:

```python
"""Pure-Python helpers for the PAPA widget.

No Qt imports; functions are unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_particles_per_frame(points: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Convert a Points layer (N, 3) array into a per-frame count DataFrame.

    Input frames are 0-indexed (napari convention); output frames are 1-indexed
    to match the canonical analysis in papa_analysis.py.

    Returns
    -------
    per_frame : DataFrame with columns ``frame`` (1-indexed int) and
        ``n_particles`` (int). Reindexed so every frame in [1, max_frame] is
        present (zero-detection frames included with ``n_particles=0``).
    n_negative_clipped : number of input rows that had negative frame indices
        (clipped to 0 before the +1 shift). Caller may surface this as a
        warning.
    """
    raise NotImplementedError


def build_cycle_info(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Generate cycle/group boundary information.

    Logic mirrors papa_analysis.py:47-79. Cycles and frames are 1-indexed.
    """
    raise NotImplementedError


def annotate_with_cycle_group(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
) -> pd.DataFrame:
    """Add ``cycle`` and ``group`` columns (nullable Int64) to ``per_frame``.

    Frames falling in skip-gap regions receive ``NaN`` in both columns.
    Returns a new DataFrame; does not mutate the input.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Create empty `_papa_widget.py` with class stub**

Create `src/napari_spotiflow_tracking/_papa_widget.py`:

```python
from __future__ import annotations

import napari
from qtpy.QtWidgets import QWidget


class PAPAWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
```

- [ ] **Step 5: Create empty test files**

Create `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`:

```python
"""Tests for _papa_analysis pure-Python helpers."""
```

Create `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
"""Tests for PAPAWidget."""
```

- [ ] **Step 6: Verify tests collect without error**

Run: `pytest src/napari_spotiflow_tracking/_tests/ --collect-only -q`
Expected: tests collect successfully. No new tests yet (the two new files are empty).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat: scaffold PAPA analysis module and widget"
```

---

## Task 2: Implement `compute_particles_per_frame` (TDD)

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_analysis.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`

- [ ] **Step 1: Write failing tests for `compute_particles_per_frame`**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`:

```python
import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking._papa_analysis import compute_particles_per_frame


class TestComputeParticlesPerFrame:
    def test_basic_counts(self):
        # Frame 0 has 1 spot, frame 1 has 0 spots, frame 2 has 4 spots.
        # 0-indexed input -> 1-indexed output.
        points = np.array([
            [0, 10.0, 10.0],
            [2, 11.0, 11.0],
            [2, 12.0, 12.0],
            [2, 13.0, 13.0],
            [2, 14.0, 14.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert list(per_frame.columns) == ["frame", "n_particles"]
        assert per_frame["frame"].tolist() == [1, 2, 3]
        assert per_frame["n_particles"].tolist() == [1, 0, 4]

    def test_fills_missing_frames(self):
        # Only frames 0, 2, 5 have detections -> output covers 1..6 with zeros.
        points = np.array([
            [0, 10.0, 10.0],
            [2, 10.0, 10.0],
            [5, 10.0, 10.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert per_frame["frame"].tolist() == [1, 2, 3, 4, 5, 6]
        assert per_frame["n_particles"].tolist() == [1, 0, 1, 0, 0, 1]

    def test_non_integer_frames_rounded(self):
        points = np.array([
            [0.4, 10.0, 10.0],
            [1.6, 10.0, 10.0],  # rounds to 2 -> 1-indexed 3
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 0
        assert per_frame["frame"].tolist() == [1, 2, 3]
        assert per_frame["n_particles"].tolist() == [1, 0, 1]

    def test_negative_frames_clipped(self):
        points = np.array([
            [-1, 10.0, 10.0],
            [-2, 10.0, 10.0],
            [0, 10.0, 10.0],
        ])
        per_frame, n_neg = compute_particles_per_frame(points)
        assert n_neg == 2
        # Negatives clipped to 0 -> 1-indexed frame 1, joined with the original
        # frame-0 spot -> 3 spots on frame 1.
        assert per_frame["frame"].tolist() == [1]
        assert per_frame["n_particles"].tolist() == [3]

    def test_returns_int_dtypes(self):
        points = np.array([[0, 1.0, 2.0], [1, 3.0, 4.0]])
        per_frame, _ = compute_particles_per_frame(points)
        assert pd.api.types.is_integer_dtype(per_frame["frame"])
        assert pd.api.types.is_integer_dtype(per_frame["n_particles"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestComputeParticlesPerFrame -v`
Expected: 5 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `compute_particles_per_frame`**

Replace the `compute_particles_per_frame` body in `src/napari_spotiflow_tracking/_papa_analysis.py`:

```python
def compute_particles_per_frame(points: np.ndarray) -> tuple[pd.DataFrame, int]:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with columns (frame, y, x)")

    raw_frames = np.rint(points[:, 0]).astype(int)
    n_negative_clipped = int((raw_frames < 0).sum())
    frames_0idx = np.clip(raw_frames, 0, None)
    frames_1idx = frames_0idx + 1

    if frames_1idx.size == 0:
        per_frame = pd.DataFrame({"frame": [], "n_particles": []}).astype(int)
        return per_frame, n_negative_clipped

    counts = (
        pd.Series(frames_1idx)
        .value_counts()
        .sort_index()
    )
    max_frame = int(frames_1idx.max())
    full_index = pd.Index(range(1, max_frame + 1), name="frame")
    counts = counts.reindex(full_index, fill_value=0).astype(int)

    per_frame = counts.reset_index()
    per_frame.columns = ["frame", "n_particles"]
    per_frame["frame"] = per_frame["frame"].astype(int)
    return per_frame, n_negative_clipped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestComputeParticlesPerFrame -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py
git commit -m "feat(papa): compute_particles_per_frame helper"
```

---

## Task 3: Implement `build_cycle_info` (TDD)

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_analysis.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`:

```python
from napari_spotiflow_tracking._papa_analysis import build_cycle_info


class TestBuildCycleInfo:
    def test_exact_fit_one_cycle(self):
        # max_time=12, g1=5, g2=5, skip=1: G1 frames 1-5, skip frame 6,
        # G2 frames 7-11, skip frame 12. One complete cycle.
        info = build_cycle_info(max_time=12, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
        ]

    def test_partial_g2(self):
        # max_time=8, g1=5, g2=5, skip=1: G1 1-5, skip 6, G2 7-8 (truncated).
        info = build_cycle_info(max_time=8, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 8},
        ]

    def test_no_room_for_g2(self):
        # max_time=5: only G1.
        info = build_cycle_info(max_time=5, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
        ]

    def test_two_full_cycles(self):
        # max_time=24, g1=5, g2=5, skip=1: cycle 1 G1 1-5, G2 7-11; cycle 2
        # G1 13-17, G2 19-23.
        info = build_cycle_info(max_time=24, group1_size=5, group2_size=5, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
            {"cycle": 2, "group": 1, "start_frame": 13, "end_frame": 17},
            {"cycle": 2, "group": 2, "start_frame": 19, "end_frame": 23},
        ]

    def test_zero_max_time(self):
        info = build_cycle_info(max_time=0, group1_size=5, group2_size=5, skip_frames=1)
        assert info == []

    def test_skip_zero(self):
        # No skip: G1 1-3 then G2 4-6.
        info = build_cycle_info(max_time=6, group1_size=3, group2_size=3, skip_frames=0)
        assert info == [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 3},
            {"cycle": 1, "group": 2, "start_frame": 4, "end_frame": 6},
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestBuildCycleInfo -v`
Expected: 6 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `build_cycle_info` (port from papa_analysis.py)**

Replace the `build_cycle_info` body in `_papa_analysis.py`:

```python
def build_cycle_info(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    cycle_info: list[dict] = []
    cycle_num = 0
    current_frame = 1

    while current_frame <= max_time:
        cycle_num += 1

        g1_start = current_frame
        g1_end = min(current_frame + group1_size - 1, max_time)
        cycle_info.append({
            "cycle": cycle_num,
            "group": 1,
            "start_frame": g1_start,
            "end_frame": g1_end,
        })
        current_frame = g1_end + 1 + skip_frames

        if current_frame <= max_time:
            g2_start = current_frame
            g2_end = min(current_frame + group2_size - 1, max_time)
            cycle_info.append({
                "cycle": cycle_num,
                "group": 2,
                "start_frame": g2_start,
                "end_frame": g2_end,
            })
            current_frame = g2_end + 1 + skip_frames

    return cycle_info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestBuildCycleInfo -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py
git commit -m "feat(papa): build_cycle_info helper"
```

---

## Task 4: Implement `annotate_with_cycle_group` (TDD)

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_analysis.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`:

```python
from napari_spotiflow_tracking._papa_analysis import annotate_with_cycle_group


class TestAnnotateWithCycleGroup:
    def test_assigns_cycle_and_group(self):
        per_frame = pd.DataFrame({
            "frame": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "n_particles": [0] * 12,
        })
        cycle_info = [
            {"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 5},
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
        ]
        out = annotate_with_cycle_group(per_frame, cycle_info)

        # Frames 1-5 -> cycle 1 group 1; frame 6 -> NaN/NaN (skip gap);
        # frames 7-11 -> cycle 1 group 2; frame 12 -> NaN/NaN.
        expected_cycle = [1, 1, 1, 1, 1, pd.NA, 1, 1, 1, 1, 1, pd.NA]
        expected_group = [1, 1, 1, 1, 1, pd.NA, 2, 2, 2, 2, 2, pd.NA]
        assert out["cycle"].tolist() == expected_cycle
        assert out["group"].tolist() == expected_group
        # Original columns preserved.
        assert out["frame"].tolist() == per_frame["frame"].tolist()
        assert out["n_particles"].tolist() == per_frame["n_particles"].tolist()

    def test_does_not_mutate_input(self):
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [0, 0]})
        cycle_info = [{"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 2}]
        _ = annotate_with_cycle_group(per_frame, cycle_info)
        assert "cycle" not in per_frame.columns
        assert "group" not in per_frame.columns

    def test_empty_cycle_info(self):
        per_frame = pd.DataFrame({"frame": [1, 2, 3], "n_particles": [0, 0, 0]})
        out = annotate_with_cycle_group(per_frame, [])
        assert out["cycle"].isna().all()
        assert out["group"].isna().all()

    def test_uses_nullable_int64_dtype(self):
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [0, 0]})
        cycle_info = [{"cycle": 1, "group": 1, "start_frame": 1, "end_frame": 2}]
        out = annotate_with_cycle_group(per_frame, cycle_info)
        assert str(out["cycle"].dtype) == "Int64"
        assert str(out["group"].dtype) == "Int64"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestAnnotateWithCycleGroup -v`
Expected: 4 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `annotate_with_cycle_group`**

Replace the `annotate_with_cycle_group` body in `_papa_analysis.py`:

```python
def annotate_with_cycle_group(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
) -> pd.DataFrame:
    out = per_frame.copy()
    cycle_col = pd.array([pd.NA] * len(out), dtype="Int64")
    group_col = pd.array([pd.NA] * len(out), dtype="Int64")

    frame_to_idx = {int(f): i for i, f in enumerate(out["frame"].tolist())}
    for info in cycle_info:
        for f in range(info["start_frame"], info["end_frame"] + 1):
            i = frame_to_idx.get(f)
            if i is not None:
                cycle_col[i] = info["cycle"]
                group_col[i] = info["group"]

    out["cycle"] = cycle_col
    out["group"] = group_col
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestAnnotateWithCycleGroup -v`
Expected: 4 passed.

- [ ] **Step 5: Run full _papa_analysis test module**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py -v`
Expected: 15 passed (5 + 6 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py
git commit -m "feat(papa): annotate_with_cycle_group helper"
```

---

## Task 5: Build the PAPAWidget Input tab UI

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing widget construction test**

Replace contents of `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
"""Tests for PAPAWidget."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def papa_widget(make_napari_viewer):
    from napari_spotiflow_tracking._papa_widget import PAPAWidget

    viewer = make_napari_viewer()
    widget = PAPAWidget(viewer)
    return widget, viewer


class TestPAPAWidgetConstruction:
    def test_creates_without_error(self, papa_widget):
        widget, _ = papa_widget
        assert widget is not None

    def test_has_input_and_plots_tabs(self, papa_widget):
        widget, _ = papa_widget
        assert widget._tabs.count() == 2
        assert widget._tabs.tabText(0) == "Input"
        assert widget._tabs.tabText(1) == "Plots"

    def test_default_field_values(self, papa_widget):
        widget, _ = papa_widget
        assert widget._group1_size.text() == "5"
        assert widget._group2_size.text() == "5"
        assert widget._skip_frames.text() == "1"
        assert widget._label_group1.text() == "DR"
        assert widget._label_group2.text() == "PAPA"

    def test_export_csv_starts_disabled(self, papa_widget):
        widget, _ = papa_widget
        assert widget._export_btn.isEnabled() is False

    def test_initial_status_message(self, papa_widget):
        widget, _ = papa_widget
        assert "Ready" in widget._status_label.text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py -v`
Expected: 5 failures (`AttributeError: ... has no attribute '_tabs'` etc.).

- [ ] **Step 3: Implement Input tab UI**

Replace contents of `src/napari_spotiflow_tracking/_papa_widget.py`:

```python
from __future__ import annotations

import napari
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PAPAWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_per_frame_df = None
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        self._input_tab = self._build_input_tab()
        self._plots_tab = self._build_plots_tab()
        self._tabs.addTab(self._input_tab, "Input")
        self._tabs.addTab(self._plots_tab, "Plots")

    def _build_input_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        row.addWidget(self._points_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Cycle Pattern group
        cycle_group = QGroupBox("Cycle Pattern")
        cycle_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Group 1 size:"))
        self._group1_size = QLineEdit("5")
        self._group1_size.setValidator(QIntValidator(1, 100000, self))
        row1.addWidget(self._group1_size)
        cycle_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Group 2 size:"))
        self._group2_size = QLineEdit("5")
        self._group2_size.setValidator(QIntValidator(1, 100000, self))
        row2.addWidget(self._group2_size)
        cycle_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Skip frames:"))
        self._skip_frames = QLineEdit("1")
        self._skip_frames.setValidator(QIntValidator(0, 100, self))
        row3.addWidget(self._skip_frames)
        cycle_layout.addLayout(row3)

        cycle_group.setLayout(cycle_layout)
        layout.addWidget(cycle_group)

        # Labels group
        labels_group = QGroupBox("Labels")
        labels_layout = QVBoxLayout()
        row_l1 = QHBoxLayout()
        row_l1.addWidget(QLabel("Group 1 label:"))
        self._label_group1 = QLineEdit("DR")
        row_l1.addWidget(self._label_group1)
        labels_layout.addLayout(row_l1)
        row_l2 = QHBoxLayout()
        row_l2.addWidget(QLabel("Group 2 label:"))
        self._label_group2 = QLineEdit("PAPA")
        row_l2.addWidget(self._label_group2)
        labels_layout.addLayout(row_l2)
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)

        # Buttons + status
        self._plot_btn = QPushButton("Plot")
        layout.addWidget(self._plot_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

        self._status_label = QLabel("Ready. Set parameters and click Plot.")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return page

    def _build_plots_tab(self) -> QWidget:
        # Plots tab built in Task 7. For now, return an empty scroll area
        # so the tab exists.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setLayout(QVBoxLayout())
        scroll.setWidget(inner)
        self._plots_inner = inner
        return scroll
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetConstruction -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): widget shell with Input tab UI"
```

---

## Task 6: Wire layer combo refresh

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing tests for layer combo behavior**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
import napari
import napari.layers


class TestPAPAWidgetLayerCombo:
    def test_combo_starts_empty(self, papa_widget):
        widget, _ = papa_widget
        assert widget._points_combo.count() == 0

    def test_combo_lists_points_layers(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20], [1, 11, 21]]), name="spots")
        assert widget._points_combo.count() == 1
        assert widget._points_combo.itemText(0) == "spots"

    def test_combo_ignores_image_layers(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_image(np.random.rand(64, 64), name="img")
        assert widget._points_combo.count() == 0

    def test_combo_updates_on_layer_remove(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20]]), name="spots")
        assert widget._points_combo.count() == 1
        viewer.layers.remove("spots")
        assert widget._points_combo.count() == 0

    def test_manual_refresh_button_works(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20]]), name="spots")
        widget._points_combo.clear()
        # Walk children to find the refresh button: it's the only push
        # button with text "↻".
        from qtpy.QtWidgets import QPushButton
        refresh_btns = [
            b for b in widget.findChildren(QPushButton) if b.text() == "↻"
        ]
        assert len(refresh_btns) == 1
        refresh_btns[0].click()
        assert widget._points_combo.count() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetLayerCombo -v`
Expected: tests `test_combo_lists_points_layers` and onwards fail (combo not refreshed).

- [ ] **Step 3: Implement combo refresh and event hookup**

In `_papa_widget.py`, modify `__init__` and add helpers. Replace `__init__`:

```python
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_per_frame_df = None
        self._setup_ui()
        self._connect_events()
        self._refresh_points_combo()
```

In the same file, modify the refresh button creation in `_build_input_tab` to wire it up. Find the line:

```python
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        row.addWidget(refresh_btn)
```

Replace with:

```python
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_points_combo)
        row.addWidget(refresh_btn)
```

Add at the bottom of the `PAPAWidget` class (after `_build_plots_tab`):

```python
    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetLayerCombo -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): layer combo refresh on insert/remove"
```

---

## Task 7: Embed matplotlib canvases in the Plots tab

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing test for canvases**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetCanvases:
    def test_combined_figure_and_canvas_exist(self, papa_widget):
        widget, _ = papa_widget
        assert widget._fig_combined is not None
        assert widget._canvas_combined is not None
        assert widget._toolbar_combined is not None

    def test_ratio_figure_and_canvas_exist(self, papa_widget):
        widget, _ = papa_widget
        assert widget._fig_ratio is not None
        assert widget._canvas_ratio is not None
        assert widget._toolbar_ratio is not None

    def test_figures_start_blank(self, papa_widget):
        widget, _ = papa_widget
        assert len(widget._fig_combined.axes) == 0
        assert len(widget._fig_ratio.axes) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetCanvases -v`
Expected: 3 failures (`AttributeError: ... has no attribute '_fig_combined'`).

- [ ] **Step 3: Add matplotlib imports and update `_build_plots_tab`**

In `_papa_widget.py`, add at the top of the imports block (after the qtpy import):

```python
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
```

Replace `_build_plots_tab`:

```python
    def _build_plots_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner.setLayout(inner_layout)

        self._fig_combined = Figure(figsize=(10, 4), tight_layout=True)
        self._canvas_combined = FigureCanvas(self._fig_combined)
        self._toolbar_combined = NavigationToolbar(self._canvas_combined, self)
        inner_layout.addWidget(self._toolbar_combined)
        inner_layout.addWidget(self._canvas_combined)

        self._fig_ratio = Figure(figsize=(10, 9), tight_layout=True)
        self._canvas_ratio = FigureCanvas(self._fig_ratio)
        self._toolbar_ratio = NavigationToolbar(self._canvas_ratio, self)
        inner_layout.addWidget(self._toolbar_ratio)
        inner_layout.addWidget(self._canvas_ratio)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        return scroll
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetCanvases -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): embed matplotlib canvases with toolbars in Plots tab"
```

---

## Task 8: Implement Plot button validation

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing tests for validation guards**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetValidation:
    def test_no_layer_selected(self, papa_widget):
        widget, _ = papa_widget
        widget._plot_btn.click()
        assert "no points layer" in widget._status_label.text().lower()
        assert widget._export_btn.isEnabled() is False

    def test_wrong_shape_layer(self, papa_widget):
        widget, viewer = papa_widget
        # Create a Points layer with 2 columns instead of 3.
        viewer.add_points(np.array([[10.0, 20.0]]), name="bad")
        widget._plot_btn.click()
        assert "3 columns" in widget._status_label.text()

    def test_empty_layer(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.empty((0, 3)), name="empty", ndim=3)
        widget._plot_btn.click()
        assert "empty" in widget._status_label.text().lower()

    def test_empty_group_size_field(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10.0, 20.0]]), name="spots")
        widget._group1_size.setText("")
        widget._plot_btn.click()
        assert "group 1 size" in widget._status_label.text().lower()

    def test_zero_group_size_rejected(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10.0, 20.0]]), name="spots")
        # Bypass the validator (which prevents typing 0) by setText.
        widget._group2_size.setText("0")
        widget._plot_btn.click()
        assert "group 2 size" in widget._status_label.text().lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetValidation -v`
Expected: 5 failures (button does nothing yet).

- [ ] **Step 3: Implement plot button + validation**

In `_papa_widget.py`, add the following imports near the top of the file (alongside the existing `import napari`):

```python
import numpy as np
from napari.utils.notifications import show_error, show_info

from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
)
```

Wire the Plot button to the click handler. In `_build_input_tab`, find:

```python
        self._plot_btn = QPushButton("Plot")
        layout.addWidget(self._plot_btn)
```

Replace with:

```python
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.clicked.connect(self._on_plot_clicked)
        layout.addWidget(self._plot_btn)
```

Add the following methods to the `PAPAWidget` class (after `_refresh_points_combo`):

```python
    def _set_status_error(self, msg: str):
        self._status_label.setText(msg)
        show_error(msg)

    def _parse_int_field(self, field, name: str, min_value: int) -> int | None:
        text = field.text().strip()
        if not text:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        try:
            value = int(text)
        except ValueError:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        if value < min_value:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        return value

    def _validate_inputs(self):
        """Returns a tuple (points, g1, g2, skip, label1, label2) on success,
        or None if any guard failed (status already updated)."""
        layer_name = self._points_combo.currentText()
        if not layer_name:
            self._set_status_error("No Points layer selected.")
            return None

        if layer_name not in self.viewer.layers:
            self._set_status_error("Selected layer no longer exists.")
            return None

        layer = self.viewer.layers[layer_name]
        if not isinstance(layer, napari.layers.Points):
            self._set_status_error("Selected layer is not a Points layer.")
            return None

        data = np.asarray(layer.data)
        if data.ndim != 2 or data.shape[1] != 3:
            self._set_status_error(
                "Points layer must have 3 columns (frame, y, x)."
            )
            return None

        if len(data) == 0:
            self._set_status_error("Points layer is empty; nothing to plot.")
            return None

        g1 = self._parse_int_field(self._group1_size, "Group 1 size", 1)
        if g1 is None:
            return None
        g2 = self._parse_int_field(self._group2_size, "Group 2 size", 1)
        if g2 is None:
            return None
        skip = self._parse_int_field(self._skip_frames, "Skip frames", 0)
        if skip is None:
            return None

        label1 = self._label_group1.text() or "G1"
        label2 = self._label_group2.text() or "G2"
        return data, g1, g2, skip, label1, label2

    def _on_plot_clicked(self):
        validated = self._validate_inputs()
        if validated is None:
            return
        # Plot pipeline added in Task 9.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetValidation -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): plot button validation guards"
```

---

## Task 9: Implement combined-plot rendering and pipeline

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing tests for the happy-path plot pipeline**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetPlotPipeline:
    def _make_synthetic_layer(self, viewer, n_frames=12, spots_per_frame=3):
        """Add a Points layer with `spots_per_frame` spots on each of `n_frames` frames."""
        rows = []
        rng = np.random.default_rng(0)
        for f in range(n_frames):
            for _ in range(spots_per_frame):
                rows.append([f, rng.uniform(0, 100), rng.uniform(0, 100)])
        viewer.add_points(np.array(rows), name="spots")

    def test_plot_renders_combined_axes(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=12)
        widget._group1_size.setText("5")
        widget._group2_size.setText("5")
        widget._skip_frames.setText("1")
        widget._plot_btn.click()
        assert len(widget._fig_combined.axes) == 1

    def test_plot_caches_per_frame_df(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=12, spots_per_frame=3)
        widget._plot_btn.click()
        df = widget._last_per_frame_df
        assert df is not None
        assert list(df.columns) == ["frame", "n_particles", "cycle", "group"]
        assert df["n_particles"].tolist() == [3] * 12

    def test_plot_status_message(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=12)
        widget._plot_btn.click()
        assert "plotted" in widget._status_label.text().lower()
        assert "1 cycles" in widget._status_label.text() or \
               "1 cycle" in widget._status_label.text()

    def test_export_button_enabled_after_plot(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=12)
        widget._plot_btn.click()
        assert widget._export_btn.isEnabled() is True

    def test_active_tab_switches_to_plots(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=12)
        assert widget._tabs.currentIndex() == 0
        widget._plot_btn.click()
        assert widget._tabs.currentIndex() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetPlotPipeline -v`
Expected: 5 failures.

- [ ] **Step 3: Implement combined plot rendering and complete the pipeline**

In `_papa_widget.py`, replace `_on_plot_clicked` with:

```python
    def _on_plot_clicked(self):
        validated = self._validate_inputs()
        if validated is None:
            return
        data, g1, g2, skip, label1, label2 = validated

        per_frame, n_neg = compute_particles_per_frame(data)
        if len(per_frame) == 0:
            self._set_status_error("Need at least 1 frame of data.")
            return

        max_time = int(per_frame["frame"].max())
        cycle_info = build_cycle_info(max_time, g1, g2, skip)
        per_frame = annotate_with_cycle_group(per_frame, cycle_info)

        self._render_combined_plot(
            self._fig_combined, per_frame, cycle_info,
            max_time, g1, g2, skip, label1, label2,
        )
        self._canvas_combined.draw_idle()

        self._render_ratio_plot(
            self._fig_ratio, per_frame, cycle_info, label1, label2,
        )
        self._canvas_ratio.draw_idle()

        self._last_per_frame_df = per_frame
        self._export_btn.setEnabled(True)

        n_cycles = max((c["cycle"] for c in cycle_info), default=0)
        msg = f"Plotted {n_cycles} cycles across {max_time} frames."
        if g1 + skip + g2 + skip > max_time and n_cycles > 0:
            msg += " (cycle pattern exceeds frame range; only partial cycles drawn.)"
        if n_neg > 0:
            msg += f" Negative frame indices found; clipped to 0 ({n_neg} rows)."
        self._status_label.setText(msg)
        show_info(msg)

        self._tabs.setCurrentWidget(self._plots_tab)
```

Add the combined-plot rendering method (and a placeholder for the ratio one — Task 10 fills it in):

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
                for text in legend.get_texts():
                    text.set_color(cls.NAPARI_FG)

    def _render_combined_plot(
        self, fig, per_frame, cycle_info, max_time,
        g1, g2, skip, label1, label2,
    ):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(
            per_frame["frame"], per_frame["n_particles"],
            color=self.LINE_COLOR, linewidth=1.0,
        )

        sep_colors = ["tab:blue", "tab:orange"]
        sep1_labeled = False
        sep2_labeled = False
        for info in cycle_info:
            if info["end_frame"] < max_time:
                sep_color = sep_colors[(info["group"] - 1) % 2]
                if info["group"] == 1 and not sep1_labeled:
                    sep_label = label1
                    sep1_labeled = True
                elif info["group"] == 2 and not sep2_labeled:
                    sep_label = label2
                    sep2_labeled = True
                else:
                    sep_label = None
                ax.axvline(
                    x=info["end_frame"] + 0.5, color=sep_color,
                    linestyle="-", alpha=0.6, linewidth=1.5,
                    label=sep_label,
                )

        ax.set_title(
            f"All Cycles Combined — Particles per Frame  "
            f"(G1={g1}f + skip{skip}f + G2={g2}f + skip{skip}f)"
        )
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Number of Particles")
        if sep1_labeled or sep2_labeled:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self._style_figure(fig)

    def _render_ratio_plot(self, fig, per_frame, cycle_info, label1, label2):
        # Stub - filled in Task 10.
        fig.clear()
        self._style_figure(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetPlotPipeline -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): plot pipeline + combined timeseries rendering"
```

---

## Task 10: Implement ratio plot rendering

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing test for ratio plot**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetRatioPlot:
    def _make_synthetic_layer(self, viewer, n_frames=24):
        rows = []
        for f in range(n_frames):
            for _ in range(3):
                rows.append([f, 10.0, 10.0])
        viewer.add_points(np.array(rows), name="spots")

    def test_ratio_figure_has_three_axes(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        assert len(widget._fig_ratio.axes) == 3

    def test_ratio_figure_titles_contain_labels(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._label_group1.setText("DR")
        widget._label_group2.setText("PAPA")
        widget._plot_btn.click()
        titles = [ax.get_title() for ax in widget._fig_ratio.axes]
        assert "DR / PAPA" in titles[1]
        assert "PAPA / DR" in titles[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetRatioPlot -v`
Expected: 2 failures (ratio fig has 0 axes; titles missing).

- [ ] **Step 3: Implement `_render_ratio_plot`**

In `_papa_widget.py`, replace the stub `_render_ratio_plot` with:

```python
    def _render_ratio_plot(self, fig, per_frame, cycle_info, label1, label2):
        fig.clear()
        cycles = sorted({info["cycle"] for info in cycle_info})

        g1_counts = []
        g2_counts = []
        ratios_g1_g2 = []
        ratios_g2_g1 = []

        for cyc in cycles:
            g1_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 1]
            g2_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 2]

            g1_total = 0
            if g1_info:
                mask = (
                    (per_frame["frame"] >= g1_info[0]["start_frame"])
                    & (per_frame["frame"] <= g1_info[0]["end_frame"])
                )
                g1_total = int(per_frame.loc[mask, "n_particles"].sum())

            g2_total = 0
            if g2_info:
                mask = (
                    (per_frame["frame"] >= g2_info[0]["start_frame"])
                    & (per_frame["frame"] <= g2_info[0]["end_frame"])
                )
                g2_total = int(per_frame.loc[mask, "n_particles"].sum())

            g1_counts.append(g1_total)
            g2_counts.append(g2_total)
            ratios_g1_g2.append(g1_total / g2_total if g2_total > 0 else np.nan)
            ratios_g2_g1.append(g2_total / g1_total if g1_total > 0 else np.nan)

        x = np.arange(len(cycles))
        bar_width = 0.35

        ax1, ax2, ax3 = fig.subplots(
            3, 1, sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1]},
        )

        ax1.bar(x - bar_width / 2, g1_counts, bar_width, color="tab:blue",
                label=f"{label1} (G1)")
        ax1.bar(x + bar_width / 2, g2_counts, bar_width, color="tab:orange",
                label=f"{label2} (G2)")
        ax1.set_ylabel("Total Particle Count")
        ax1.set_title("Total Particle Count per Group per Cycle")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x, ratios_g1_g2, 0.5, color="tab:green")
        ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_ylabel(f"Ratio ({label1} / {label2})")
        ax2.set_title(f"Particle Count Ratio ({label1} / {label2}) per Cycle")
        ax2.grid(axis="y", alpha=0.3)

        ax3.bar(x, ratios_g2_g1, 0.5, color="tab:red")
        ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax3.set_ylabel(f"Ratio ({label2} / {label1})")
        ax3.set_xlabel("Cycle")
        ax3.set_title(f"Particle Count Ratio ({label2} / {label1}) per Cycle")
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(c) for c in cycles])
        ax3.grid(axis="y", alpha=0.3)

        self._style_figure(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetRatioPlot -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): 3-panel ratio plot rendering"
```

---

## Task 11: Verify napari dark theme styling

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

(No code change to `_papa_widget.py` — `_style_figure` is already wired in by Tasks 9 and 10. This task pins the styling behavior with an explicit test.)

- [ ] **Step 1: Write the styling test**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetTheming:
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(12) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")

    def test_combined_figure_uses_dark_facecolor(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        # Hex #262930 -> rgb (0.149..., 0.160..., 0.188...). Allow
        # matplotlib's float-rgba representation.
        from matplotlib.colors import to_rgba
        expected = to_rgba(widget.NAPARI_BG)
        assert widget._fig_combined.get_facecolor() == expected
        for ax in widget._fig_combined.axes:
            assert ax.get_facecolor() == expected

    def test_ratio_figure_uses_dark_facecolor(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        from matplotlib.colors import to_rgba
        expected = to_rgba(widget.NAPARI_BG)
        assert widget._fig_ratio.get_facecolor() == expected
        for ax in widget._fig_ratio.axes:
            assert ax.get_facecolor() == expected
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetTheming -v`
Expected: 2 passed (styling already implemented in Task 9).

- [ ] **Step 3: Commit**

```bash
git add src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "test(papa): pin dark theme facecolor on figures and axes"
```

---

## Task 12: Implement CSV export

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`

- [ ] **Step 1: Write failing tests for export**

Append to `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

```python
class TestPAPAWidgetExportCSV:
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(12) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")

    def test_export_writes_csv(self, papa_widget, tmp_path, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        out_path = tmp_path / "papa.csv"
        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV files (*.csv)"),
        )
        widget._export_btn.click()

        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert list(df.columns) == ["frame", "n_particles", "cycle", "group"]
        # Frames are 1-indexed.
        assert df["frame"].min() == 1
        # Skip-gap frame (6) should be present with NaN cycle/group and
        # n_particles == 3 (the layer has 3 spots on every frame).
        skip_row = df[df["frame"] == 6].iloc[0]
        assert pd.isna(skip_row["cycle"])
        assert pd.isna(skip_row["group"])
        assert int(skip_row["n_particles"]) == 3

    def test_export_cancel_does_nothing(self, papa_widget, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        # Should not raise.
        widget._export_btn.click()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetExportCSV -v`
Expected: 1 failure (`test_export_writes_csv` — file is not written; `test_export_cancel_does_nothing` may pass trivially since the button is unwired).

- [ ] **Step 3: Implement export and wire the button**

In `_papa_widget.py`, add `QFileDialog` to the qtpy imports:

```python
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
```

Modify the export button creation in `_build_input_tab`. Find:

```python
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)
```

Replace with:

```python
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        layout.addWidget(self._export_btn)
```

Add the export method to the class (after `_render_ratio_plot`):

```python
    def _on_export_clicked(self):
        if self._last_per_frame_df is None or len(self._last_per_frame_df) == 0:
            self._set_status_error("No data to export. Click Plot first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save particles CSV", "", "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self._last_per_frame_df.to_csv(path, index=False)
        except OSError as exc:
            self._set_status_error(f"Failed to write CSV: {exc}")
            return
        msg = f"Exported {len(self._last_per_frame_df)} rows to {path}"
        self._status_label.setText(msg)
        show_info(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py::TestPAPAWidgetExportCSV -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa): CSV export with QFileDialog"
```

---

## Task 13: Register PAPA widget in the napari manifest

**Files:**
- Modify: `src/napari_spotiflow_tracking/napari.yaml`

- [ ] **Step 1: Add the PAPA widget command and widget entry**

Replace the contents of `src/napari_spotiflow_tracking/napari.yaml` with:

```yaml
name: napari-spotiflow-tracking
display_name: Spotiflow 2D Tracking
contributions:
  commands:
    - id: napari-spotiflow-tracking.preprocessing_widget
      python_name: napari_spotiflow_tracking._cleanup_widget:PreProcessingWidget
      title: Pre-Processing
    - id: napari-spotiflow-tracking.detection_widget
      python_name: napari_spotiflow_tracking._detection_widget:DetectionWidget
      title: Spot Detection
    - id: napari-spotiflow-tracking.tracking_widget
      python_name: napari_spotiflow_tracking._tracking_widget:TrackingWidget
      title: Spot Tracking
    - id: napari-spotiflow-tracking.papa_widget
      python_name: napari_spotiflow_tracking._papa_widget:PAPAWidget
      title: PAPA
  widgets:
    - command: napari-spotiflow-tracking.preprocessing_widget
      display_name: Pre-Processing
    - command: napari-spotiflow-tracking.detection_widget
      display_name: Spot Detection
    - command: napari-spotiflow-tracking.tracking_widget
      display_name: Spot Tracking
    - command: napari-spotiflow-tracking.papa_widget
      display_name: PAPA
```

- [ ] **Step 2: Verify manifest loads**

Run: `python -c "import yaml; yaml.safe_load(open('src/napari_spotiflow_tracking/napari.yaml'))"`
Expected: no output, no error.

- [ ] **Step 3: Verify napari plugin discovery**

Run: `python -c "import napari; from napari.plugins import _plugins; from napari_spotiflow_tracking._papa_widget import PAPAWidget; print('PAPAWidget importable')"`
Expected: `PAPAWidget importable`.

- [ ] **Step 4: Commit**

```bash
git add src/napari_spotiflow_tracking/napari.yaml
git commit -m "feat(papa): register PAPA widget in napari manifest"
```

---

## Task 14: Final test run + manual smoke test

**Files:** none modified (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest src/napari_spotiflow_tracking/_tests/ -v`
Expected: all tests pass — including the existing tests for tracking/detection/preprocessing widgets.

- [ ] **Step 2: Manual napari smoke test**

Run: `python -c "import napari; v = napari.Viewer(); v.window.add_plugin_dock_widget('napari-spotiflow-tracking', 'PAPA'); napari.run()"`

In the napari window:
1. Confirm a "PAPA" dock widget appears with two tabs (Input, Plots).
2. Add a Points layer manually via `viewer.add_points(...)` in the python console, e.g.
   ```python
   import numpy as np
   v.add_points(np.array([[f, 10, 10] for f in range(24) for _ in range(3)]), name='spots')
   ```
3. In the PAPA widget, leave defaults (G1=5, G2=5, skip=1, labels DR/PAPA), click **Plot**.
4. Verify the Plots tab opens, the combined timeseries shows particle counts with blue/orange separators, and the ratio panel shows 3 sub-axes with bar charts.
5. Test the navigation toolbar: zoom-rect, pan, home, save (save should produce a PNG file).
6. Click **Export CSV** and save to a temp path; open the resulting CSV and confirm 4 columns (frame, n_particles, cycle, group), 1-indexed frames, and NaN cycle/group rows for skip frames.

- [ ] **Step 3: Update memory and finalize**

If the smoke test passes, no further commits are required (per-task commits already in place). If issues are found, file a follow-up rather than amending.

---

## Self-Review Notes

- **Spec coverage:**
  - §1 architecture & files → Tasks 1, 5, 7, 13 (modules, manifest, dependency).
  - §2 widget layout → Tasks 5, 6, 7 (Input tab, combo refresh, Plots tab).
  - §3 data flow & computation → Tasks 8, 9 (validation + pipeline).
  - §4 plot rendering & theming → Tasks 9, 10, 11 (combined, ratio, theme).
  - §5 error handling → Task 8 (validation guards) + Task 9 (soft warning, negative-frame warning) + Task 12 (export cancel/failure).
  - §6 testing → Tasks 2, 3, 4 (pure-function), 5, 6, 7, 8, 9, 10, 11, 12 (widget).
- **Type consistency:** `compute_particles_per_frame` returns `tuple[DataFrame, int]` consistently in Tasks 1, 2, 9. `_render_combined_plot` and `_render_ratio_plot` signatures stable across Tasks 9, 10. Manifest IDs match the spec (`napari-spotiflow-tracking.papa_widget`).
- **No placeholders:** every code step contains the full code; tests have assertions; expected outputs are stated.
