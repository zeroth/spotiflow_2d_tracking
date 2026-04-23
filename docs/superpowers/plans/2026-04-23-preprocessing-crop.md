# Pre-Processing Crop Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Crop section to the Pre-Processing widget that crops the selected image using the axis-aligned bounding box of a rectangle drawn in a napari Shapes layer.

**Architecture:** Two pure helpers in `_preprocessing.py` (`first_rectangle_bbox` and `crop_image`) contain all non-Qt logic and are covered by unit tests. The widget code in `_cleanup_widget.py` only wires Qt widgets, reads napari layer state, and calls the helpers. A dropdown selector + "+ New" button creates/selects a Shapes layer in rectangle-draw mode.

**Tech Stack:** Python, numpy, napari, qtpy (Qt5/Qt6), pytest, pytest-qt via `make_napari_viewer`.

**Reference:** [`docs/superpowers/specs/2026-04-23-preprocessing-crop-design.md`](../specs/2026-04-23-preprocessing-crop-design.md)

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/napari_spotiflow_tracking/_preprocessing.py` | Pure image-processing helpers | **Modify**: add `first_rectangle_bbox`, `crop_image` |
| `src/napari_spotiflow_tracking/_tests/test_crop.py` | Unit tests for the helpers | **Create** |
| `src/napari_spotiflow_tracking/_cleanup_widget.py` | Pre-Processing Qt widget | **Modify**: add Crop `QGroupBox`, three methods, extend `_on_layer_change` |
| `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` | Widget smoke tests | **Create** |

---

## Task 1: Pure helpers (`first_rectangle_bbox` + `crop_image`) — TDD

**Files:**
- Create: `src/napari_spotiflow_tracking/_tests/test_crop.py`
- Modify: `src/napari_spotiflow_tracking/_preprocessing.py` (append at end)

### Step 1: Write the failing tests

Create `src/napari_spotiflow_tracking/_tests/test_crop.py` with the following exact content:

```python
import numpy as np
import pytest


class TestFirstRectangleBbox:
    def test_simple_2d_rectangle(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (10, 60, 20, 80)

    def test_clipping_to_image_bounds(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[-10, -10], [-10, 50], [200, 50], [200, -10]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (0, 100, 0, 50)

    def test_entirely_out_of_bounds_raises(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[200, 200], [200, 300], [400, 300], [400, 200]], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))

    def test_skips_non_rectangle_and_uses_next_rectangle(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        ellipse = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        rect = np.array([[20, 30], [20, 70], [60, 70], [60, 30]], dtype=float)
        bbox = first_rectangle_bbox(
            [ellipse, rect], ["ellipse", "rectangle"], image_shape=(100, 100),
        )
        assert bbox == (20, 60, 30, 70)

    def test_no_rectangle_raises(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        ellipse = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        with pytest.raises(ValueError, match="No rectangle"):
            first_rectangle_bbox([ellipse], ["ellipse"], image_shape=(100, 100))

    def test_rotated_rectangle_uses_aabb(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        # Square inscribed at 45 deg: corners at (50,30), (70,50), (50,70), (30,50).
        corners = np.array([[50, 30], [70, 50], [50, 70], [30, 50]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (30, 70, 30, 70)

    def test_3d_corner_coords(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        # Rectangle stored with (T, Y, X) coords; T is constant, use last two dims.
        corners = np.array(
            [[4, 10, 20], [4, 10, 80], [4, 60, 80], [4, 60, 20]], dtype=float,
        )
        bbox = first_rectangle_bbox(
            [corners], ["rectangle"], image_shape=(8, 100, 100),
        )
        assert bbox == (10, 60, 20, 80)

    def test_rounds_fractional_coords(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array(
            [[10.4, 20.6], [10.4, 79.4], [59.6, 79.4], [59.6, 20.6]], dtype=float,
        )
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        # 10.4 -> 10, 59.6 -> 60, 20.6 -> 21, 79.4 -> 79
        assert bbox == (10, 60, 21, 79)


class TestCropImage:
    def test_2d_crop(self):
        from napari_spotiflow_tracking._preprocessing import crop_image

        image = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        result = crop_image(image, (10, 60, 20, 80))
        assert result.shape == (50, 60)
        # Ensure it matches the manual slice
        assert np.array_equal(result, image[10:60, 20:80])

    def test_3d_stack_crop_preserves_time(self):
        from napari_spotiflow_tracking._preprocessing import crop_image

        stack = np.arange(5 * 100 * 100, dtype=np.float32).reshape(5, 100, 100)
        result = crop_image(stack, (10, 60, 20, 80))
        assert result.shape == (5, 50, 60)
        assert np.array_equal(result, stack[:, 10:60, 20:80])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_crop.py -v`

Expected: all tests fail with `ImportError` or `AttributeError` because `first_rectangle_bbox` and `crop_image` don't exist yet in `_preprocessing.py`.

- [ ] **Step 3: Implement the helpers**

Append this block to the **end** of `src/napari_spotiflow_tracking/_preprocessing.py`:

```python
# ── Crop via Shapes layer rectangle ──────────────────────────────────


def first_rectangle_bbox(shapes_data, shape_types, image_shape):
    """Compute axis-aligned bbox of the first rectangle in a Shapes layer.

    Iterates through the shapes list, picks the first one whose type is
    ``"rectangle"``, takes the last two coordinate dims as (Y, X), and
    returns the AABB clipped to the image's Y/X bounds.

    Args:
        shapes_data: list of (N, D) corner arrays, as stored on
            ``napari.layers.Shapes.data``.
        shape_types: parallel list of shape-type strings, as stored on
            ``napari.layers.Shapes.shape_type``.
        image_shape: shape tuple of the target image. The last two entries
            must be ``(H, W)``.

    Returns:
        Tuple ``(y_min, y_max, x_min, x_max)`` suitable for slicing.

    Raises:
        ValueError: no rectangle is present, or the clipped bbox is empty.
    """
    h, w = image_shape[-2], image_shape[-1]
    for data, stype in zip(shapes_data, shape_types):
        if stype != "rectangle":
            continue
        corners = np.asarray(data, dtype=float)
        ys = corners[:, -2]
        xs = corners[:, -1]
        y_min = max(0, int(round(float(ys.min()))))
        y_max = min(int(h), int(round(float(ys.max()))))
        x_min = max(0, int(round(float(xs.min()))))
        x_max = min(int(w), int(round(float(xs.max()))))
        if y_max - y_min <= 0 or x_max - x_min <= 0:
            raise ValueError(
                "Crop region is empty (rectangle outside image bounds)."
            )
        return (y_min, y_max, x_min, x_max)
    raise ValueError("No rectangle found in shapes layer.")


def crop_image(image, bbox):
    """Slice an image using a ``(y_min, y_max, x_min, x_max)`` bbox.

    Works for both 2D ``(Y, X)`` and 3D ``(T, Y, X)`` inputs because the
    slice indexes only the last two dimensions.

    Args:
        image: 2D or 3D numpy array.
        bbox: tuple returned by :func:`first_rectangle_bbox`.

    Returns:
        Cropped array (a view into ``image``).
    """
    y_min, y_max, x_min, x_max = bbox
    return image[..., y_min:y_max, x_min:x_max]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_crop.py -v`

Expected: all 10 tests pass (8 in `TestFirstRectangleBbox` + 2 in `TestCropImage`).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_preprocessing.py src/napari_spotiflow_tracking/_tests/test_crop.py
git commit -m "feat: add first_rectangle_bbox and crop_image helpers"
```

---

## Task 2: Wire the Crop UI into PreProcessingWidget — TDD

**Files:**
- Create: `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py`
- Modify: `src/napari_spotiflow_tracking/_cleanup_widget.py`

### Step 1: Write the failing widget tests

Create `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` with the following exact content:

```python
import numpy as np
import pytest


@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking._cleanup_widget import PreProcessingWidget

    viewer = make_napari_viewer()
    widget = PreProcessingWidget(viewer)
    return widget, viewer


class TestPreProcessingWidgetCreation:
    def test_creates_without_error(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget is not None

    def test_has_crop_controls(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget._shapes_combo is not None
        assert widget._crop_btn is not None
        assert widget._crop_btn.text() == "Apply Crop"


class TestCropWorkflow:
    def test_new_shapes_layer_button_creates_and_selects(self, preproc_widget):
        widget, viewer = preproc_widget

        widget._new_shapes_layer()

        # Exactly one Shapes layer named "Crop ROI" is present
        shapes_layers = [l for l in viewer.layers if l.__class__.__name__ == "Shapes"]
        assert len(shapes_layers) == 1
        assert shapes_layers[0].name == "Crop ROI"
        # Combo is populated and points at the new layer
        assert widget._shapes_combo.currentText() == "Crop ROI"

    def test_shapes_combo_refreshes_on_layer_add(self, preproc_widget):
        widget, viewer = preproc_widget

        viewer.add_shapes(name="my_rois")

        items = [widget._shapes_combo.itemText(i) for i in range(widget._shapes_combo.count())]
        assert "my_rois" in items

    def test_apply_crop_on_2d_image_adds_cropped_layer(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_rectangles(
            np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        cropped = viewer.layers["img (crop)"]
        assert cropped.data.shape == (50, 60)
        assert np.array_equal(cropped.data, image[10:60, 20:80])

    def test_apply_crop_on_3d_stack_preserves_time(self, preproc_widget):
        widget, viewer = preproc_widget

        stack = np.arange(5 * 100 * 100, dtype=np.float32).reshape(5, 100, 100)
        viewer.add_image(stack, name="stack")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("stack")

        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_rectangles(
            np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        cropped = viewer.layers["stack (crop)"]
        assert cropped.data.shape == (5, 50, 60)

    def test_apply_crop_without_rectangle_shows_no_output(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((100, 100), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        # Shapes layer with an ellipse only
        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_ellipses(
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        # No "img (crop)" layer was added
        names = [l.name for l in viewer.layers]
        assert "img (crop)" not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py -v`

Expected: `test_has_crop_controls`, `test_new_shapes_layer_button_creates_and_selects`, and the three crop-workflow tests fail with `AttributeError` on `_shapes_combo`, `_crop_btn`, `_new_shapes_layer`, or `_run_crop`. `test_creates_without_error` may still pass.

- [ ] **Step 3: Add the Crop group to `_setup_ui`**

In `src/napari_spotiflow_tracking/_cleanup_widget.py`, find the block that ends the **Input** group (right after `layout.addWidget(input_group)`) and **insert the following new block immediately after it**, before the `# Device` section:

```python
        # ── Crop ──────────────────────────────────────────────────────
        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout()

        shapes_row = QHBoxLayout()
        shapes_row.addWidget(QLabel("Shapes layer:"))
        self._shapes_combo = QComboBox()
        shapes_row.addWidget(self._shapes_combo)
        refresh_shapes_btn = QPushButton("↻")
        refresh_shapes_btn.setFixedWidth(30)
        refresh_shapes_btn.setToolTip("Refresh shapes layer list")
        refresh_shapes_btn.clicked.connect(self._refresh_shapes_combo)
        shapes_row.addWidget(refresh_shapes_btn)
        new_shapes_btn = QPushButton("+ New")
        new_shapes_btn.setToolTip(
            "Create a new empty Shapes layer and enter rectangle-draw mode"
        )
        new_shapes_btn.clicked.connect(self._new_shapes_layer)
        shapes_row.addWidget(new_shapes_btn)
        crop_layout.addLayout(shapes_row)

        self._crop_btn = QPushButton("Apply Crop")
        self._crop_btn.clicked.connect(self._run_crop)
        crop_layout.addWidget(self._crop_btn)

        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)
```

- [ ] **Step 4: Initialize the combo on setup**

Still in `_setup_ui`, find the line near the bottom that reads `self._refresh_image_combo()` and **change** it to call both refreshers:

```python
        layout.addStretch()
        self._refresh_image_combo()
        self._refresh_shapes_combo()
```

- [ ] **Step 5: Extend `_on_layer_change` to also refresh the shapes combo**

Locate the `_on_layer_change` method (defined just below `_connect_events`) and replace its body so both combos refresh:

```python
    def _on_layer_change(self, event=None):
        self._refresh_image_combo()
        self._refresh_shapes_combo()
```

- [ ] **Step 6: Import the helpers**

Locate the existing import block at the top of `_cleanup_widget.py`:

```python
from napari_spotiflow_tracking._preprocessing import (
    remove_background_high_pass,
    remove_background_high_pass_stack,
    remove_background_erosion,
    walking_average,
)
```

Replace it with:

```python
from napari_spotiflow_tracking._preprocessing import (
    crop_image,
    first_rectangle_bbox,
    remove_background_high_pass,
    remove_background_high_pass_stack,
    remove_background_erosion,
    walking_average,
)
```

- [ ] **Step 7: Add the three new methods**

Add these three methods to `PreProcessingWidget`. Place them immediately **after** `_refresh_image_combo` (so all combo helpers sit together):

```python
    def _refresh_shapes_combo(self):
        prev = self._shapes_combo.currentText()
        self._shapes_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes):
                self._shapes_combo.addItem(layer.name)
        idx = self._shapes_combo.findText(prev)
        if idx >= 0:
            self._shapes_combo.setCurrentIndex(idx)

    def _new_shapes_layer(self):
        layer = self.viewer.add_shapes(name="Crop ROI")
        layer.mode = "add_rectangle"
        self.viewer.layers.selection.active = layer
        self._refresh_shapes_combo()
        idx = self._shapes_combo.findText(layer.name)
        if idx >= 0:
            self._shapes_combo.setCurrentIndex(idx)

    def _run_crop(self):
        layer_name, image = self._get_image()
        if image is None:
            return
        if image.ndim not in (2, 3):
            show_error(
                f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D."
            )
            return

        shapes_name = self._shapes_combo.currentText()
        if not shapes_name:
            show_error("No shapes layer selected.")
            return

        shapes_layer = self.viewer.layers[shapes_name]
        try:
            bbox = first_rectangle_bbox(
                shapes_layer.data,
                shapes_layer.shape_type,
                image.shape,
            )
        except ValueError as e:
            show_error(str(e))
            return

        result = crop_image(image, bbox)
        y_min, y_max, x_min, x_max = bbox
        self.viewer.add_image(result, name=f"{layer_name} (crop)")
        show_info(
            f"Cropped to Y=[{y_min}:{y_max}] X=[{x_min}:{x_max}]"
        )
        self._status_label.setText(
            f"Cropped to Y=[{y_min}:{y_max}] X=[{x_min}:{x_max}]."
        )
```

- [ ] **Step 8: Run the widget tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py -v`

Expected: all 7 tests pass.

- [ ] **Step 9: Run the full test suite to catch regressions**

Run: `pytest src/napari_spotiflow_tracking/_tests/ -v`

Expected: all tests pass, including the existing `test_detection_widget`, `test_segmentation`, `test_fitting`, `test_tracking`, `test_tracking_widget` suites.

- [ ] **Step 10: Manual smoke test in napari**

Since this is a GUI feature, launch napari and exercise the new Crop group by hand. Use the project's `.venv`:

```bash
.venv/Scripts/python -m napari
```

Then:

1. From napari menu: **Plugins → Spotiflow 2D Tracking → Pre-Processing**.
2. Load any 2D TIFF or a `(T, Y, X)` stack (drag-drop into napari).
3. In the Pre-Processing widget, Crop section:
   a. Click **+ New** — a "Crop ROI" Shapes layer should appear and rectangle-draw mode should activate.
   b. Draw a rectangle in the canvas.
   c. Click **Apply Crop** — a new `<image_name> (crop)` image layer should appear containing only the cropped region.
4. Try it on a 3D stack — scrub the time slider in the cropped layer to confirm T is preserved.
5. Try clicking **Apply Crop** with no rectangle drawn (empty Shapes layer) — you should see a "No rectangle found in shapes layer" error toast, no new layer.
6. Draw a rectangle that extends past the image edge — it should still crop correctly (clipped to bounds).

Report back any discrepancies between observed behavior and the above.

- [ ] **Step 11: Commit**

```bash
git add src/napari_spotiflow_tracking/_cleanup_widget.py src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py
git commit -m "feat: add Crop section to Pre-Processing widget"
```

---

## Self-Review Checklist

1. **Spec coverage:** ✓
   - UI placement (Crop group between Input and Background Removal) — Task 2, Step 3.
   - Shapes combo + ⟳ + "+ New" button — Task 2, Step 3.
   - Apply Crop button wiring — Task 2, Steps 3 + 7.
   - First-rectangle extraction — Task 1, `first_rectangle_bbox`.
   - AABB computation including rotated rectangles — Task 1, tested in `test_rotated_rectangle_uses_aabb`.
   - Round + clip to image bounds — Task 1, tested in `test_clipping_to_image_bounds` and `test_rounds_fractional_coords`.
   - Empty bbox error — Task 1, tested in `test_entirely_out_of_bounds_raises`.
   - 2D + 3D (T,Y,X) slicing via `[..., y:y, x:x]` — Task 1, `crop_image`.
   - New image layer named `{name} (crop)` — Task 2, Step 7.
   - No worker / no preview / no multi-rect handling — honored by omission.
   - `_on_layer_change` extended — Task 2, Step 5.
   - All error-handling rows from the spec's Error Handling Summary are covered by the widget `_run_crop` method.

2. **Placeholder scan:** no TBDs, no "add appropriate error handling", no "similar to Task N". Every code block is complete.

3. **Type consistency:**
   - `first_rectangle_bbox(shapes_data, shape_types, image_shape)` — consistent across helper definition, test calls, and widget call site.
   - `crop_image(image, bbox)` — consistent.
   - Return tuple ordering `(y_min, y_max, x_min, x_max)` — same in helper, tests, and widget unpacking.
   - Import list in `_cleanup_widget.py` is alphabetized and includes both new symbols.
