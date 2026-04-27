# Pre-Processing Calculation Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Calculation section to the Pre-Processing widget that adds or subtracts a scalar from the selected image, with a "Mean" button that fills the value field with the image's overall mean.

**Architecture:** A pure helper `apply_calculation(image, op, value)` in `_preprocessing.py` does the math (always returns float64). Widget glue in `_cleanup_widget.py` adds a `QGroupBox` with an Operation combo, value spinbox, "Mean" button, and "Apply" button.

**Tech Stack:** Python, numpy, napari, qtpy (Qt5/Qt6), pytest, pytest-qt via `make_napari_viewer`.

**Reference:** [`docs/superpowers/specs/2026-04-23-preprocessing-calculation-design.md`](../specs/2026-04-23-preprocessing-calculation-design.md)

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/napari_spotiflow_tracking/_preprocessing.py` | Pure image-processing helpers | **Modify**: append `apply_calculation` |
| `src/napari_spotiflow_tracking/_tests/test_calculation.py` | Unit tests for the pure helper | **Create** |
| `src/napari_spotiflow_tracking/_cleanup_widget.py` | Pre-Processing Qt widget | **Modify**: add Calculation `QGroupBox` to `_setup_ui`, add `_run_calc_mean` and `_run_calc_apply` methods, import the helper |
| `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` | Widget smoke tests | **Modify**: append a `TestCalculationWorkflow` class |

---

## Task 1: Pure helper `apply_calculation` — TDD

**Files:**
- Create: `src/napari_spotiflow_tracking/_tests/test_calculation.py`
- Modify: `src/napari_spotiflow_tracking/_preprocessing.py` (append at end)

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking/_tests/test_calculation.py` with the following exact content:

```python
import numpy as np
import pytest


class TestApplyCalculation:
    def test_add_returns_sum(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((10, 10), dtype=np.uint16)
        result = apply_calculation(arr, "add", 5.0)
        assert np.array_equal(result, np.full((10, 10), 6.0))

    def test_subtract_returns_difference(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((10, 10), 10, dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 3.0)
        assert np.array_equal(result, np.full((10, 10), 7.0))

    @pytest.mark.parametrize(
        "in_dtype",
        [np.uint8, np.uint16, np.float32, np.float64, np.int32],
    )
    def test_output_dtype_is_float64(self, in_dtype):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=in_dtype)
        result = apply_calculation(arr, "add", 1.0)
        assert result.dtype == np.float64

    def test_3d_stack(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        stack = np.full((5, 10, 10), 2.0, dtype=np.float32)
        result = apply_calculation(stack, "add", 3.0)
        assert result.shape == (5, 10, 10)
        assert np.allclose(result, 5.0)

    def test_subtract_below_zero_no_underflow(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 5.0)
        assert np.array_equal(result, np.full((4, 4), -5.0))

    def test_add_negative_value(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((4, 4), 10.0, dtype=np.float32)
        result = apply_calculation(arr, "add", -3.0)
        assert np.array_equal(result, np.full((4, 4), 7.0))

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown op"):
            apply_calculation(arr, "multiply", 2.0)

    def test_result_is_independent_copy(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=np.float64)
        result = apply_calculation(arr, "add", 0.0)
        result[0, 0] = 999.0
        assert arr[0, 0] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`

If `pytest` is not on PATH, use `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`.

Expected: all tests fail with `ImportError` because `apply_calculation` does not yet exist in `_preprocessing.py`. (`pytest.parametrize` over 5 dtypes counts as 5 collected tests, so total failures ≈ 12.)

- [ ] **Step 3: Implement the helper**

Append this block to the **end** of `src/napari_spotiflow_tracking/_preprocessing.py`:

```python
# ── Scalar calculation (add / subtract) ──────────────────────────────


def apply_calculation(image: np.ndarray, op: str, value: float) -> np.ndarray:
    """Add or subtract a scalar from an image.

    Always returns a new ``float64`` array — never a view, never the input
    dtype. Promotion to float avoids underflow when subtracting from
    unsigned-integer images.

    Args:
        image: 2D or 3D numpy array.
        op: ``"add"`` or ``"subtract"``.
        value: scalar to add or subtract. May be negative.

    Returns:
        New ``float64`` array of the same shape as ``image``.

    Raises:
        ValueError: if ``op`` is not one of ``"add"`` or ``"subtract"``.
    """
    arr = image.astype(np.float64)
    if op == "add":
        return arr + value
    if op == "subtract":
        return arr - value
    raise ValueError(f"Unknown op: {op!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`

Expected: all tests pass (12 tests including the 5 parametrized dtype variants).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_preprocessing.py src/napari_spotiflow_tracking/_tests/test_calculation.py
git commit -m "feat: add apply_calculation helper (add/subtract scalar)"
```

---

## Task 2: Wire the Calculation UI into PreProcessingWidget — TDD

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` (append a new test class)
- Modify: `src/napari_spotiflow_tracking/_cleanup_widget.py` (add UI + methods + import)

- [ ] **Step 1: Append the failing widget tests**

Append the following block to the **end** of `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` (after the existing `TestCropWorkflow` class, leaving one blank line between classes):

```python
class TestCalculationWorkflow:
    def test_section_exists_with_defaults(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget._calc_op_combo is not None
        assert widget._calc_value is not None
        assert widget._calc_apply_btn is not None
        items = [
            widget._calc_op_combo.itemText(i)
            for i in range(widget._calc_op_combo.count())
        ]
        assert items == ["Add", "Subtract"]
        assert widget._calc_op_combo.currentText() == "Add"
        assert widget._calc_apply_btn.text() == "Apply"

    def test_mean_button_populates_spinbox(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.full((10, 10), 4.0, dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._run_calc_mean()

        assert widget._calc_value.value() == pytest.approx(4.0)

    def test_apply_add_creates_layer_with_offset(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5.0)
        widget._run_calc_apply()

        assert "img (+5)" in [l.name for l in viewer.layers]
        assert np.array_equal(
            viewer.layers["img (+5)"].data, np.full((10, 10), 5.0),
        )

    def test_apply_subtract_creates_layer_with_negative(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Subtract")
        widget._calc_value.setValue(3.0)
        widget._run_calc_apply()

        assert "img (-3)" in [l.name for l in viewer.layers]
        assert np.array_equal(
            viewer.layers["img (-3)"].data, np.full((10, 10), -3.0),
        )

    def test_apply_with_no_image_does_nothing(self, preproc_widget):
        widget, viewer = preproc_widget

        before_names = [l.name for l in viewer.layers]
        # No image is in the combo (it's empty)
        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5.0)
        widget._run_calc_apply()

        after_names = [l.name for l in viewer.layers]
        assert after_names == before_names

    def test_mean_with_no_image_does_nothing(self, preproc_widget):
        widget, _ = preproc_widget

        before = widget._calc_value.value()
        widget._run_calc_mean()
        # Spinbox is unchanged (no image was selected so nothing happened)
        assert widget._calc_value.value() == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py::TestCalculationWorkflow -v`

Expected: every test in `TestCalculationWorkflow` fails with `AttributeError` referencing `_calc_op_combo`, `_calc_value`, `_calc_apply_btn`, `_run_calc_mean`, or `_run_calc_apply`. The earlier `TestPreProcessingWidgetCreation` and `TestCropWorkflow` tests are not run (we restricted to the new class) but should still pass when run together.

- [ ] **Step 3: Import the helper**

Open `src/napari_spotiflow_tracking/_cleanup_widget.py`. Locate the existing import block (lines 22–29):

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

Replace it with:

```python
from napari_spotiflow_tracking._preprocessing import (
    apply_calculation,
    crop_image,
    first_rectangle_bbox,
    remove_background_high_pass,
    remove_background_high_pass_stack,
    remove_background_erosion,
    walking_average,
)
```

- [ ] **Step 4: Add the Calculation `QGroupBox` to `_setup_ui`**

In `src/napari_spotiflow_tracking/_cleanup_widget.py`, find the line in `_setup_ui` that reads `layout.addWidget(crop_group)` (the closing line of the Crop section, near the existing line 89). **Insert the following block immediately after it**, before the `# Device` section:

```python
        # ── Calculation ──────────────────────────────────────────────
        calc_group = QGroupBox("Calculation")
        calc_layout = QVBoxLayout()

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self._calc_op_combo = QComboBox()
        self._calc_op_combo.addItems(["Add", "Subtract"])
        op_row.addWidget(self._calc_op_combo)
        calc_layout.addLayout(op_row)

        value_row = QHBoxLayout()
        value_row.addWidget(QLabel("Value:"))
        self._calc_value = QDoubleSpinBox()
        self._calc_value.setRange(-1e9, 1e9)
        self._calc_value.setDecimals(4)
        self._calc_value.setValue(0.0)
        value_row.addWidget(self._calc_value)
        self._calc_mean_btn = QPushButton("Mean")
        self._calc_mean_btn.setToolTip(
            "Fill the value field with the mean of the selected image"
        )
        self._calc_mean_btn.clicked.connect(self._run_calc_mean)
        value_row.addWidget(self._calc_mean_btn)
        calc_layout.addLayout(value_row)

        self._calc_apply_btn = QPushButton("Apply")
        self._calc_apply_btn.clicked.connect(self._run_calc_apply)
        calc_layout.addWidget(self._calc_apply_btn)

        calc_group.setLayout(calc_layout)
        layout.addWidget(calc_group)
```

- [ ] **Step 5: Add the two new methods**

Add these two methods to `PreProcessingWidget`. Place them immediately **before** the existing `# ── Background Removal ─` section comment (i.e., right after the existing crop methods `_refresh_shapes_combo`, `_new_shapes_layer`, `_run_crop`):

```python
    # ── Calculation ──────────────────────────────────────────────────

    def _run_calc_mean(self):
        layer_name, image = self._get_image()
        if image is None:
            return
        mean_val = float(np.mean(image))
        self._calc_value.setValue(mean_val)
        self._status_label.setText(f"Mean = {mean_val:g}")

    def _run_calc_apply(self):
        layer_name, image = self._get_image()
        if image is None:
            return

        op = self._calc_op_combo.currentText().lower()
        value = self._calc_value.value()

        try:
            result = apply_calculation(image, op, value)
        except ValueError as e:
            show_error(str(e))
            return

        sign = "+" if op == "add" else "-"
        suffix = f"({sign}{value:g})"
        self.viewer.add_image(result, name=f"{layer_name} {suffix}")
        show_info(f"Calculated: {op} {value:g}")
        self._status_label.setText(f"Applied {op} {value:g}.")
```

- [ ] **Step 6: Run the new test class to verify it passes**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py::TestCalculationWorkflow -v`

Expected: all 6 tests in `TestCalculationWorkflow` pass.

- [ ] **Step 7: Run the full test suite to catch regressions**

Run: `pytest src/napari_spotiflow_tracking/_tests/ -v`

Expected: all tests pass — the existing 50 plus the 12 from Task 1 plus the 6 new widget tests = **68 tests**, all green. No regressions in `test_detection_widget`, `test_segmentation`, `test_fitting`, `test_tracking`, `test_tracking_widget`, `test_crop`, or the previously existing parts of `test_cleanup_widget`.

- [ ] **Step 8: Manual smoke test in napari (deferred to human)**

DO NOT execute. Manual napari testing will be done by the human operator. In your report, simply note that this step is pending human verification.

For the human's reference, the smoke test is:

```bash
.venv/Scripts/python -m napari
```

Then: Plugins → Spotiflow 2D Tracking → Pre-Processing → Calculation section.

1. Load any 2D TIFF or `(T, Y, X)` stack.
2. In the Calculation section, click **Mean** — the Value field should populate with a non-zero number (the image's mean intensity).
3. Tweak the value (e.g., round it).
4. Select **Subtract** in the Operation dropdown.
5. Click **Apply** — a new image layer named `<orig> (-<value>)` should appear with the offset applied. Verify by sampling pixels in napari (the new layer's intensities should equal `original - value`).
6. Try **Add** with a positive value and confirm the new layer's intensities are higher.
7. Try clicking **Apply** when no image is selected — no error toast, no new layer (silent no-op via `_get_image`).

- [ ] **Step 9: Commit**

```bash
git add src/napari_spotiflow_tracking/_cleanup_widget.py src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py
git commit -m "feat: add Calculation section to Pre-Processing widget"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Pure helper with `add` / `subtract` / `ValueError` on unknown op → Task 1.
   - Output always `float64` (verified for uint8, uint16, float32, float64, int32) → Task 1, parametrized test.
   - 2D and 3D shape preservation → Task 1, `test_3d_stack` + 2D tests.
   - Underflow safety on unsigned subtract → Task 1, `test_subtract_below_zero_no_underflow`.
   - Independent-copy contract → Task 1, `test_result_is_independent_copy`.
   - `QGroupBox("Calculation")` between Crop and Background Removal → Task 2, Step 4.
   - Operation combo with `["Add", "Subtract"]`, default Add → Task 2, Step 4 + `test_section_exists_with_defaults`.
   - `QDoubleSpinBox` value field, range `-1e9..1e9`, 4 decimals, default 0 → Task 2, Step 4.
   - **Mean** button computes `np.mean` and writes via `setValue` → Task 2, Steps 4 + 5 + `test_mean_button_populates_spinbox`.
   - **Apply** button runs op via helper, adds new layer with `(+v)` / `(-v)` suffix → Task 2, Step 5 + `test_apply_add_*` and `test_apply_subtract_*`.
   - Layer naming format with `{value:g}` to drop trailing zeros (`5.0` → `5`) → Task 2, Step 5 (`f"({sign}{value:g})"`) + tests assert `"img (+5)"` / `"img (-3)"`.
   - `_get_image()` reuse for "No image selected" path → Task 2, Step 5 + `test_apply_with_no_image_does_nothing` and `test_mean_with_no_image_does_nothing`.
   - `show_info` + `_status_label` updates → Task 2, Step 5.
   - No worker / no preview / no clipping / no per-frame mean → honored by omission.

2. **Placeholder scan:** No TBDs, no "add appropriate error handling", no "similar to Task N". Every code block is complete and self-contained.

3. **Type consistency:**
   - `apply_calculation(image, op, value)` — same signature in helper, tests, and widget call site.
   - Return type `np.ndarray` (float64) — consistent across helper docstring, tests, widget consumer.
   - Widget attribute names `_calc_op_combo`, `_calc_value`, `_calc_mean_btn`, `_calc_apply_btn` — same in tests and widget.
   - Widget method names `_run_calc_mean`, `_run_calc_apply` — same in widget definition and test calls.
   - Combo items `["Add", "Subtract"]` and `op = .lower()` — consistent with helper accepting `"add"` / `"subtract"`.
