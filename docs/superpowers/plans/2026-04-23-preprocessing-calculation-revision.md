# Pre-Processing Calculation Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the just-shipped Calculation feature so the result keeps the input dtype, the spinbox is integer-only and non-negative, the Mean button rounds to int, and overflow / underflow is clipped to the input dtype's range.

**Architecture:** Replace `apply_calculation` in `_preprocessing.py` with a dtype-preserving + clipping version that does integer math in `int64` and clips to the input dtype's range before casting back. Swap `QDoubleSpinBox → QSpinBox` in the widget and update the Mean / Apply paths to integer values. Update the existing tests in place — same files, same test class structures, revised assertions.

**Tech Stack:** Python, numpy, napari, qtpy, pytest, pytest-qt via `make_napari_viewer`.

**Reference:** [`docs/superpowers/specs/2026-04-23-preprocessing-calculation-revision-design.md`](../specs/2026-04-23-preprocessing-calculation-revision-design.md). The original feature spec is at [`docs/superpowers/specs/2026-04-23-preprocessing-calculation-design.md`](../specs/2026-04-23-preprocessing-calculation-design.md) and the original plan at [`docs/superpowers/plans/2026-04-23-preprocessing-calculation.md`](2026-04-23-preprocessing-calculation.md).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/napari_spotiflow_tracking/_preprocessing.py` | Pure image-processing helpers | **Modify**: replace the `apply_calculation` body and docstring (lines 402-428 today) |
| `src/napari_spotiflow_tracking/_tests/test_calculation.py` | Unit tests for the pure helper | **Modify**: replace the `TestApplyCalculation` class body |
| `src/napari_spotiflow_tracking/_cleanup_widget.py` | Pre-Processing Qt widget | **Modify**: swap `QDoubleSpinBox → QSpinBox` in the Calculation group (lines 92-123 today), update `_run_calc_mean` and `_run_calc_apply` (lines 383-411 today) |
| `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` | Widget smoke tests | **Modify**: update the `TestCalculationWorkflow` class to reflect dtype preservation + integer values |

---

## Task 1: Revise the pure helper `apply_calculation` — TDD

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tests/test_calculation.py` (replace the test class)
- Modify: `src/napari_spotiflow_tracking/_preprocessing.py` (replace the helper)

- [ ] **Step 1: Replace the test file with the revised tests**

Open `src/napari_spotiflow_tracking/_tests/test_calculation.py` and replace its **entire content** with:

```python
import numpy as np
import pytest


class TestApplyCalculation:
    def test_add_returns_sum_same_dtype(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((10, 10), dtype=np.uint16)
        result = apply_calculation(arr, "add", 5)
        assert np.array_equal(result, np.full((10, 10), 6, dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_subtract_returns_difference_same_dtype(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((10, 10), 10, dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 3)
        assert np.array_equal(result, np.full((10, 10), 7, dtype=np.uint16))
        assert result.dtype == np.uint16

    @pytest.mark.parametrize(
        "in_dtype",
        [np.uint8, np.uint16, np.float32, np.float64, np.int32],
    )
    def test_output_dtype_matches_input(self, in_dtype):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=in_dtype)
        result = apply_calculation(arr, "add", 1)
        assert result.dtype == in_dtype

    def test_3d_stack(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        stack = np.full((5, 10, 10), 2.0, dtype=np.float32)
        result = apply_calculation(stack, "add", 3)
        assert result.shape == (5, 10, 10)
        assert np.allclose(result, 5.0)
        assert result.dtype == np.float32

    def test_subtract_clips_at_zero_for_unsigned(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 5)
        assert np.array_equal(result, np.zeros((4, 4), dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_add_clips_at_max_for_unsigned(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((4, 4), 65000, dtype=np.uint16)
        result = apply_calculation(arr, "add", 1000)
        assert np.array_equal(result, np.full((4, 4), 65535, dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="Unknown op"):
            apply_calculation(arr, "multiply", 2)

    def test_result_is_independent_copy(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "add", 0)
        result[0, 0] = 999
        assert arr[0, 0] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`

If `pytest` is not on PATH, use `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`.

Expected: 5 tests fail. The 7 still-existing tests behave like this against the *current* float64-output helper:

- `test_add_returns_sum_same_dtype` → FAIL (`result.dtype == np.float64`, expected `np.uint16`).
- `test_subtract_returns_difference_same_dtype` → FAIL (same reason).
- `test_output_dtype_matches_input` → 4 of 5 parametrized variants FAIL; `np.float64` passes.
- `test_3d_stack` → FAIL (`result.dtype == np.float64`, expected `np.float32`).
- `test_subtract_clips_at_zero_for_unsigned` → FAIL (current returns `-5.0`, expected `0`).
- `test_add_clips_at_max_for_unsigned` → FAIL (current returns `66000.0`, expected `65535`).
- `test_unknown_op_raises` → PASS (helper still raises).
- `test_result_is_independent_copy` → PASS (current `astype` already copies).

Total: 8 failures expected.

- [ ] **Step 3: Replace the helper implementation**

In `src/napari_spotiflow_tracking/_preprocessing.py`, locate the existing block (currently lines 402-428):

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

**Replace it entirely** with:

```python
# ── Scalar calculation (add / subtract) ──────────────────────────────


def apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray:
    """Add or subtract an integer scalar from an image.

    Returns a new array with the *same dtype as the input*. For integer
    input dtypes, underflow and overflow are clipped to the dtype's
    representable range (e.g. a ``uint16`` image clipped to ``[0, 65535]``).
    For floating-point inputs the math is performed and cast back to the
    input dtype with no clipping.

    Args:
        image: 2D or 3D numpy array.
        op: ``"add"`` or ``"subtract"``.
        value: integer scalar (the widget restricts this to ``>= 0``; the
            helper itself accepts any int).

    Returns:
        New array of the same shape and dtype as ``image``. Always a copy
        — never a view.

    Raises:
        ValueError: if ``op`` is not one of ``"add"`` or ``"subtract"``.
    """
    if op not in ("add", "subtract"):
        raise ValueError(f"Unknown op: {op!r}")

    sign = 1 if op == "add" else -1

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        # Compute in int64 to avoid intermediate overflow during the math.
        wide = image.astype(np.int64) + sign * int(value)
        clipped = np.clip(wide, info.min, info.max)
        return clipped.astype(image.dtype)

    # Floating-point input: do the math, then cast back to the input dtype.
    return (image + sign * value).astype(image.dtype, copy=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_calculation.py -v`

Expected: all 12 tests pass (8 method definitions, with `test_output_dtype_matches_input` parametrized over 5 dtypes).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_preprocessing.py src/napari_spotiflow_tracking/_tests/test_calculation.py
git commit -m "refactor: dtype-preserving apply_calculation with clipping"
```

---

## Task 2: Revise the Calculation widget UI + tests — TDD

**Files:**
- Modify: `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` (replace the `TestCalculationWorkflow` class body)
- Modify: `src/napari_spotiflow_tracking/_cleanup_widget.py` (swap spinbox type, update two methods)

- [ ] **Step 1: Replace the `TestCalculationWorkflow` class with the revised tests**

In `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py`, locate the existing `class TestCalculationWorkflow:` block at the **end** of the file. **Replace the entire class** (everything from `class TestCalculationWorkflow:` to the end of the file) with:

```python
class TestCalculationWorkflow:
    def test_section_exists_with_defaults(self, preproc_widget):
        from qtpy.QtWidgets import QSpinBox

        widget, _ = preproc_widget
        assert widget._calc_op_combo is not None
        assert widget._calc_value is not None
        assert widget._calc_apply_btn is not None
        # Spinbox is integer-only and non-negative
        assert isinstance(widget._calc_value, QSpinBox)
        assert widget._calc_value.minimum() == 0
        assert widget._calc_value.maximum() >= 65535
        items = [
            widget._calc_op_combo.itemText(i)
            for i in range(widget._calc_op_combo.count())
        ]
        assert items == ["Add", "Subtract"]
        assert widget._calc_op_combo.currentText() == "Add"
        assert widget._calc_apply_btn.text() == "Apply"

    def test_mean_button_populates_spinbox_with_int(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.full((10, 10), 4, dtype=np.uint16)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._run_calc_mean()

        # Integer assertion (no approx) — Mean rounds to int.
        assert widget._calc_value.value() == 4

    def test_mean_button_rounds_float_mean(self, preproc_widget):
        widget, viewer = preproc_widget

        # Two 1s and two 2s → mean is 1.5 → rounds to 2 via banker's rounding,
        # but np.mean here yields exactly 1.5; round() rounds half to even → 2.
        image = np.array([[1, 1], [2, 2]], dtype=np.uint16)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._run_calc_mean()

        assert widget._calc_value.value() == 2

    def test_apply_add_creates_layer_with_offset_same_dtype(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.uint16)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5)
        widget._run_calc_apply()

        cropped = viewer.layers["img (+5)"]
        assert np.array_equal(cropped.data, np.full((10, 10), 5, dtype=np.uint16))
        assert cropped.data.dtype == np.uint16

    def test_apply_subtract_real_data_decreases_intensity(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.full((10, 10), 100, dtype=np.uint16)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Subtract")
        widget._calc_value.setValue(30)
        widget._run_calc_apply()

        cropped = viewer.layers["img (-30)"]
        assert np.array_equal(cropped.data, np.full((10, 10), 70, dtype=np.uint16))
        assert cropped.data.dtype == np.uint16

    def test_apply_subtract_clips_at_zero(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.uint16)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Subtract")
        widget._calc_value.setValue(3)
        widget._run_calc_apply()

        # Layer is created with name "(-3)" but data is clipped to 0.
        cropped = viewer.layers["img (-3)"]
        assert np.array_equal(cropped.data, np.zeros((10, 10), dtype=np.uint16))
        assert cropped.data.dtype == np.uint16

    def test_apply_with_no_image_does_nothing(self, preproc_widget):
        widget, viewer = preproc_widget

        before_names = [l.name for l in viewer.layers]
        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5)
        widget._run_calc_apply()

        after_names = [l.name for l in viewer.layers]
        assert after_names == before_names

    def test_mean_with_no_image_does_nothing(self, preproc_widget):
        widget, _ = preproc_widget

        before = widget._calc_value.value()
        widget._run_calc_mean()
        assert widget._calc_value.value() == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py::TestCalculationWorkflow -v`

Expected: tests fail because:

- `test_section_exists_with_defaults` → FAIL (`isinstance(_calc_value, QSpinBox)` is False — it is a `QDoubleSpinBox`).
- `test_mean_button_populates_spinbox_with_int` → FAIL or PASS by accident (current code calls `setValue(4.0)` on a `QDoubleSpinBox` which keeps `4.0`; the int comparison `4.0 == 4` is `True`, so it may pass).
- `test_mean_button_rounds_float_mean` → FAIL (current code writes `1.5`, not `2`).
- `test_apply_add_creates_layer_with_offset_same_dtype` → FAIL (current data dtype is `np.float64`, not `np.uint16`).
- `test_apply_subtract_real_data_decreases_intensity` → FAIL (dtype + value-pattern mismatch; current produces `float64`).
- `test_apply_subtract_clips_at_zero` → FAIL (current returns `-3.0`, not `0`).
- `test_apply_with_no_image_does_nothing` → PASS.
- `test_mean_with_no_image_does_nothing` → PASS.

If `test_mean_button_populates_spinbox_with_int` accidentally passes against the old code, that's fine — the failing-test signal comes from the other six.

- [ ] **Step 3: Swap `QDoubleSpinBox` → `QSpinBox` in `_setup_ui`**

In `src/napari_spotiflow_tracking/_cleanup_widget.py`, locate the existing `value_row` block in the Calculation group (currently lines 103-109):

```python
        value_row = QHBoxLayout()
        value_row.addWidget(QLabel("Value:"))
        self._calc_value = QDoubleSpinBox()
        self._calc_value.setRange(-1e9, 1e9)
        self._calc_value.setDecimals(4)
        self._calc_value.setValue(0.0)
        value_row.addWidget(self._calc_value)
```

**Replace it** with:

```python
        value_row = QHBoxLayout()
        value_row.addWidget(QLabel("Value:"))
        self._calc_value = QSpinBox()
        self._calc_value.setRange(0, 2**31 - 1)
        self._calc_value.setValue(0)
        value_row.addWidget(self._calc_value)
```

(Note: `QSpinBox` is already imported at the top of `_cleanup_widget.py` — used by the N2V epoch / patch / window-size spinboxes.)

- [ ] **Step 4: Update `_run_calc_mean` to round to int**

Locate the existing `_run_calc_mean` method (currently lines 385-391):

```python
    def _run_calc_mean(self):
        layer_name, image = self._get_image()
        if image is None:
            return
        mean_val = float(np.mean(image))
        self._calc_value.setValue(mean_val)
        self._status_label.setText(f"Mean = {mean_val:g}")
```

**Replace it** with:

```python
    def _run_calc_mean(self):
        layer_name, image = self._get_image()
        if image is None:
            return
        mean_val = int(round(float(np.mean(image))))
        self._calc_value.setValue(mean_val)
        self._status_label.setText(f"Mean = {mean_val}")
```

- [ ] **Step 5: Update `_run_calc_apply` to use integer suffixes**

Locate the existing `_run_calc_apply` method (currently lines 393-411):

```python
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

**Replace it** with:

```python
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
        suffix = f"({sign}{value})"
        self.viewer.add_image(result, name=f"{layer_name} {suffix}")
        show_info(f"Calculated: {op} {value}")
        self._status_label.setText(f"Applied {op} {value}.")
```

(Only the three format strings change: `{value:g}` → `{value}`. Since `value` is now an `int` from `QSpinBox.value()`, `{value}` formats it cleanly without a trailing `.0`.)

- [ ] **Step 6: Run the new test class to verify it passes**

Run: `pytest src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py::TestCalculationWorkflow -v`

Expected: all 8 tests in `TestCalculationWorkflow` pass.

- [ ] **Step 7: Run the full test suite to catch regressions**

Run: `pytest src/napari_spotiflow_tracking/_tests/ -v`

Expected: 70 tests pass total (50 pre-existing + 12 helper tests + 8 widget tests in `TestCalculationWorkflow`). No regressions in `test_detection_widget`, `test_segmentation`, `test_fitting`, `test_tracking`, `test_tracking_widget`, `test_crop`, or the other parts of `test_cleanup_widget`.

- [ ] **Step 8: Manual smoke test in napari (deferred to human)**

DO NOT execute. Manual napari testing will be done by the human operator. Note in your report that this step is pending human verification.

For the human's reference, the smoke test is:

```bash
.venv/Scripts/python -m napari
```

Then: Plugins → Spotiflow 2D Tracking → Pre-Processing → Calculation section.

1. Load a `uint16` TIFF or stack.
2. Verify the Value spinbox **does not accept negative values** and **does not show decimals**.
3. Click **Mean** — the field populates with an integer (rounded from the float mean).
4. Select **Subtract**, click **Apply** — a new layer named `<orig> (-<int>)` appears. Verify in the napari layer panel that the layer's dtype is `uint16` (same as input).
5. Subtract a value larger than the image's minimum (e.g. mean + 1000 on a darker region) — the affected pixels clip to `0` rather than wrap to large positive values.
6. Add a value that pushes some pixels above the dtype max (e.g. add `60000` to a `uint16` image with maxes near `5000`) — those pixels clip to `65535` rather than wrap.

- [ ] **Step 9: Commit**

```bash
git add src/napari_spotiflow_tracking/_cleanup_widget.py src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py
git commit -m "refactor: integer-only Calculation spinbox, dtype-preserving output"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Pure helper signature `apply_calculation(image, op, value: int)` returning input dtype → Task 1, Step 3.
   - Integer math via `int64` intermediate to avoid intermediate overflow → Task 1, Step 3.
   - Clipping to dtype `[min, max]` for integer inputs → Task 1, Step 3 + tested by `test_subtract_clips_at_zero_for_unsigned` and `test_add_clips_at_max_for_unsigned`.
   - Floating-point input passes through with no clipping, cast back to input dtype → Task 1, Step 3 + tested by `test_3d_stack` (asserts `dtype == np.float32`).
   - `ValueError` for unknown op → Task 1, Step 3 + `test_unknown_op_raises`.
   - `QDoubleSpinBox → QSpinBox`, range `(0, 2**31 - 1)`, default `0` → Task 2, Step 3 + `test_section_exists_with_defaults`.
   - Mean rounds to int via `int(round(float(np.mean(image))))` → Task 2, Step 4 + `test_mean_button_rounds_float_mean`.
   - Layer-name suffix uses `{value}` not `{value:g}` → Task 2, Step 5 + `test_apply_add_creates_layer_with_offset_same_dtype` (asserts `"img (+5)"`).
   - Negative values impossible by Qt range → enforced by `setRange(0, ...)`; no extra code path needed.
   - The cosmetic `(+-3)` / `(--3)` bug from the previous review is automatically gone (no negative values).

2. **Placeholder scan:** No TBDs, no "add appropriate error handling", no "similar to Task N". Every code block is complete and copy-pasteable. The "Note: `QSpinBox` is already imported" line is a check-this-not-an-action note.

3. **Type consistency:**
   - `apply_calculation(image, op, value)` — same signature in helper, helper tests, and widget call site. The `value` parameter is now `int` everywhere.
   - Widget attribute names `_calc_op_combo`, `_calc_value`, `_calc_apply_btn`, `_calc_mean_btn` — unchanged from the previous spec; tests reference them.
   - Widget method names `_run_calc_mean`, `_run_calc_apply` — unchanged.
   - Combo items `["Add", "Subtract"]` and `op = .lower()` → helper accepts `"add"` / `"subtract"` (unchanged).
   - Layer-name suffix format `(+5)` / `(-3)` consistent across helper-impl-agnostic widget code, the test assertions in Task 2, and the spec.
