# Pre-Processing Calculation Revision — Design

**Date:** 2026-04-23
**Status:** Approved for planning
**Component:** `napari_spotiflow_tracking._cleanup_widget.PreProcessingWidget` and
              `napari_spotiflow_tracking._preprocessing.apply_calculation`
**Supersedes (in part):** `2026-04-23-preprocessing-calculation-design.md`

## Summary

Revise the just-shipped Calculation feature so that the result keeps the same
dtype as the input image, the spinbox is integer-only with no negative values,
the Mean button rounds to an int, and overflow / underflow is handled by
clipping to the input dtype's range.

## Motivation

Microscopy images are almost always non-negative integers (typically `uint16`).
The original spec produced a `float64` output to avoid underflow on subtraction,
but in practice users expect the output to live in the same intensity space as
the input — both for visual consistency with other layers and for downstream
tools that assume integer microscopy data. A simpler model — integer values,
integer math, clip on overflow — matches user expectations and avoids creating
many large `float64` arrays.

## Behaviour Changes (from the original Calculation spec)

| Aspect | Before | After |
|---|---|---|
| Spinbox type | `QDoubleSpinBox`, 4 decimals | `QSpinBox`, integer only |
| Spinbox range | `-1e9` to `1e9` | `0` to `2**31 − 1` |
| Default value | `0.0` | `0` |
| Mean button output | `float(np.mean(image))` written via `setValue` | `int(round(float(np.mean(image))))` written via `setValue` |
| Helper output dtype | always `float64` | same as input image's dtype |
| Underflow / overflow | impossible (float64) | clipped to the input dtype's `[min, max]` range |
| Layer name suffix | `(+5)` / `(-3)` via `{value:g}` | `(+5)` / `(-3)` via `{value}` (already int) |
| Negative values in spinbox | allowed | not allowed (Subtract op handles "negative" intent) |
| Operation contract (`"add"` / `"subtract"`) | unchanged | unchanged |

Everything not listed above is unchanged: UI placement (between Crop and Device
sections), the two-button "Mean" + "Apply" pattern, error handling via
`show_error`, status-label updates, layer-name pattern.

## Architecture

### Pure helper

Modify `apply_calculation` in `src/napari_spotiflow_tracking/_preprocessing.py`
so that:

- It accepts an `int` `value` (signed, but the widget always passes `>= 0`).
- It computes the result in `np.int64` to avoid intermediate over/underflow,
  then clips to the input's dtype range, and finally casts back to that
  dtype.
- For floating-point input dtypes (`float32`, `float64`), the clip step is a
  no-op (clip to `±inf`); the result is just the math cast back to the input
  dtype.

```python
def apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray:
    """Add or subtract an integer scalar from an image.

    Returns an array with the *same dtype as the input*. Underflow and
    overflow are clipped to the input dtype's representable range.

    Args:
        image: 2D or 3D numpy array.
        op: ``"add"`` or ``"subtract"``.
        value: integer scalar (>= 0 in the widget; helper accepts any int).

    Returns:
        New array of the same shape and dtype as ``image``.

    Raises:
        ValueError: if ``op`` is not one of ``"add"`` or ``"subtract"``.
    """
    if op not in ("add", "subtract"):
        raise ValueError(f"Unknown op: {op!r}")

    sign = 1 if op == "add" else -1

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        # Compute in int64 to dodge intermediate overflow on the math itself
        wide = image.astype(np.int64) + sign * int(value)
        clipped = np.clip(wide, info.min, info.max)
        return clipped.astype(image.dtype)

    # Floating-point input: do the math, then cast back. No clipping needed.
    return (image + sign * value).astype(image.dtype, copy=True)
```

### Widget

In `_setup_ui`, swap the spinbox type and the value-row construction:

- Replace `self._calc_value = QDoubleSpinBox()` with `self._calc_value = QSpinBox()`.
- Set range `(0, 2**31 - 1)` (the full positive `int32` range; covers any
  microscopy intensity, well beyond `uint16`'s 65535).
- Remove `setDecimals(4)`.
- Default value `0`.

In `_run_calc_mean`, round the mean to int before `setValue`:

```python
def _run_calc_mean(self):
    layer_name, image = self._get_image()
    if image is None:
        return
    mean_val = int(round(float(np.mean(image))))
    self._calc_value.setValue(mean_val)
    self._status_label.setText(f"Mean = {mean_val}")
```

In `_run_calc_apply`, the body is structurally the same, but `value` is now
an `int` (from `QSpinBox.value()`), the suffix uses `{value}` instead of
`{value:g}`, and the layer name reflects integer values:

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

Note: because the spinbox no longer accepts negative values, the cosmetic
`(+-3)` / `(--3)` naming bug from the previous review is automatically gone.

## Imports

`QSpinBox` is already imported in `_cleanup_widget.py` (used by the existing
N2V epoch / patch / window-size spinboxes). `QDoubleSpinBox` import remains
because other sections use it. No import changes are required.

## Testing

### Pure helper — `test_calculation.py`

Update the existing test class so each test reflects the new contract.
Keep the test-class structure and most names; only the assertions and the
`in_dtype` parameter list change in detail.

1. **`test_add_returns_sum`** — `arr = np.ones((10, 10), dtype=np.uint16)`,
   `apply_calculation(arr, "add", 5)` → `np.full((10, 10), 6, dtype=np.uint16)`.
   Assert dtype is `np.uint16`.
2. **`test_subtract_returns_difference`** — `arr = np.full((10, 10), 10,
   dtype=np.uint16)`, `apply_calculation(arr, "subtract", 3)` →
   `np.full((10, 10), 7, dtype=np.uint16)`. Assert dtype is `np.uint16`.
3. **`test_output_dtype_matches_input`** (renamed from
   `test_output_dtype_is_float64`) — parametrize over
   `[np.uint8, np.uint16, np.float32, np.float64, np.int32]`. For each,
   assert `result.dtype == in_dtype`.
4. **`test_3d_stack`** — unchanged behaviour: `(5, 10, 10)` `float32` input,
   add 3, result has same shape and approx values.
5. **`test_subtract_clips_at_zero_for_unsigned`** (renamed from
   `test_subtract_below_zero_no_underflow`) — `arr = np.zeros((4, 4),
   dtype=np.uint16)`, `apply_calculation(arr, "subtract", 5)` →
   `np.zeros((4, 4), dtype=np.uint16)`. Verifies clipping to dtype min, not
   wraparound.
6. **`test_add_clips_at_max_for_unsigned`** (new) — `arr = np.full((4, 4),
   65000, dtype=np.uint16)`, `apply_calculation(arr, "add", 1000)` →
   `np.full((4, 4), 65535, dtype=np.uint16)`. Verifies clipping to dtype
   max, not wraparound.
7. **`test_unknown_op_raises`** — unchanged.
8. **`test_result_is_independent_copy`** — assertion subject changes from
   `np.float64` test to `np.uint16`: `arr = np.ones((4, 4), dtype=np.uint16)`,
   mutate result, assert input unchanged.
9. **Remove `test_add_negative_value`** — the helper still accepts negative
   `value` (it's just `int + sign * value`), but the widget can never pass
   one. The test is no longer load-bearing; drop it.

### Widget — `test_cleanup_widget.py` (`TestCalculationWorkflow`)

Update the existing tests:

- **`test_section_exists_with_defaults`** — assert `widget._calc_value` is a
  `QSpinBox` (not `QDoubleSpinBox`); assert `widget._calc_value.minimum() ==
  0` and `widget._calc_value.maximum() >= 65535`.
- **`test_mean_button_populates_spinbox`** — input image is `(10, 10)` of
  `4` (`uint16`); after `_run_calc_mean`, `widget._calc_value.value() == 4`.
  (Note: the previous test used `4.0` and `pytest.approx`; integer assertion
  is exact.)
- **`test_apply_add_creates_layer_with_offset`** — input is
  `np.zeros((10, 10), dtype=np.uint16)`; spinbox `5`; expect layer name
  `"img (+5)"` with `data == np.full((10, 10), 5, dtype=np.uint16)` AND
  `data.dtype == np.uint16`.
- **`test_apply_subtract_creates_layer_clipped`** (renamed from
  `test_apply_subtract_creates_layer_with_negative`) — input is
  `np.zeros((10, 10), dtype=np.uint16)`; spinbox `3`; expect layer name
  `"img (-3)"` with `data == np.zeros((10, 10), dtype=np.uint16)` (clipped),
  not `-3`. Verifies the dtype-preserving + clipping behaviour at the widget
  layer.
- **`test_apply_with_no_image_does_nothing`** — unchanged.
- **`test_mean_with_no_image_does_nothing`** — unchanged.

### New widget test (one)

- **`test_apply_subtract_real_data_decreases_intensity`** — input is
  `np.full((10, 10), 100, dtype=np.uint16)`; spinbox `30`; result data
  equals `np.full((10, 10), 70, dtype=np.uint16)` and dtype is `np.uint16`.
  This is the realistic happy path — clipping is the edge, intensity
  reduction is the common case — and locks in dtype preservation visibly.

## Non-Goals

- No rescaling (option B from the brainstorm) — the user explicitly
  preferred clipping over rescaling for predictability.
- No allowing negative spinbox values — Subtract is the explicit operation
  for "less".
- No changes to UI placement, the operation combo's items, or the Mean / Apply
  button text.
- No new dependencies.

## Error Handling Summary

| Condition | Action |
|---|---|
| No image selected | `_get_image()` shows `"No image selected."` and returns; widget callers early-return. (Unchanged.) |
| Spinbox cannot accept negative value | Enforced by `QSpinBox.setRange(0, ...)` — Qt physically prevents it. (No code path needed.) |
| `apply_calculation` receives an unknown op | Raises `ValueError`; widget catches and surfaces via `show_error`. (Unchanged from previous spec.) |
| Underflow on Subtract | Clipped to dtype min (e.g., `0` for unsigned). Silent — no error toast. |
| Overflow on Add | Clipped to dtype max (e.g., `65535` for `uint16`). Silent — no error toast. |
