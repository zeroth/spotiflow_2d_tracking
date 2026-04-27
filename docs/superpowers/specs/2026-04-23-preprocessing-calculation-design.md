# Pre-Processing Calculation Feature — Design

**Date:** 2026-04-23
**Status:** Approved for planning
**Component:** `napari_spotiflow_tracking._cleanup_widget.PreProcessingWidget`

## Summary

Add a **Calculation** section to the Pre-Processing widget that lets users add
or subtract a scalar from the selected image. The scalar can be entered
manually or filled from the image's overall mean by clicking a "Mean" helper
button.

## Motivation

Microscopy images often need a constant intensity offset applied — for example,
removing a known camera bias or shifting the dynamic range before downstream
detection. Today users have to drop into the napari console or run a script to
do this. A small, ergonomic UI control inside the existing Pre-Processing
widget makes the operation discoverable and keeps the user in the same
workflow as the rest of pre-processing.

## UI Placement

New `QGroupBox("Calculation")` inserted in `PreProcessingWidget._setup_ui`
**between the Crop group and the Background Removal group**, so it reads as a
simple intensity-arithmetic step early in the pipeline.

```
┌─ Input ────────────────────────────────────┐
│ ...                                         │
└─────────────────────────────────────────────┘
┌─ Crop ─────────────────────────────────────┐
│ ...                                         │
└─────────────────────────────────────────────┘
┌─ Calculation ──────────────────────────────┐
│ Operation: [Add ▾]                          │
│ Value:     [____0.000____]  [Mean]          │
│ [Apply]                                     │
└─────────────────────────────────────────────┘
┌─ Background Removal ───────────────────────┐
│ ...                                         │
└─────────────────────────────────────────────┘
```

## UI Components

| Widget | Behaviour |
|---|---|
| `QComboBox` `_calc_op_combo` | Items `["Add", "Subtract"]`. Default selection: `Add`. Future-proofed for additional ops without re-laying-out the section. |
| `QDoubleSpinBox` `_calc_value` | Range `-1e9` to `1e9`, 4 decimals, default `0.0`. Negative values are allowed (lets the user invert the sign manually if needed). |
| `QPushButton` "Mean" | Computes `float(np.mean(image))` over the *entire* selected image (all frames, all pixels) and writes the result into `_calc_value`. Does not add a layer. |
| `QPushButton` "Apply" `_calc_apply_btn` | Runs the selected operation with the current spinbox value. |

The image source is the same `_image_combo` already used by every other
section in the widget — no separate selector.

## Behaviour

### Mean button

1. Get the selected image via the existing `_get_image()` helper.
2. If no image is selected, `_get_image()` already shows
   `"No image selected."` and returns; nothing further happens.
3. Otherwise, set `self._calc_value.setValue(float(np.mean(image)))`.
4. Update `_status_label` with `"Mean = {value:g}"` for visible confirmation.

### Apply button

1. Get image via `_get_image()`. Return on no image.
2. Read `op = self._calc_op_combo.currentText().lower()` →
   `"add"` or `"subtract"`.
3. Read `value = self._calc_value.value()`.
4. Compute `result = apply_calculation(image, op, value)`.
5. Build output name with sign-baked-in:
   - Add → `f"{layer_name} (+{value:g})"`
   - Subtract → `f"{layer_name} (-{value:g})"`
   - (`{value:g}` collapses trailing zeros: `5.0` → `5`, `412.7` → `412.7`.)
6. `self.viewer.add_image(result, name=...)`.
7. `show_info(f"Calculated: {op} {value:g}")` and update `_status_label`.

### Pure helper

Add `apply_calculation` to `src/napari_spotiflow_tracking/_preprocessing.py`:

```python
def apply_calculation(image: np.ndarray, op: str, value: float) -> np.ndarray:
    """Add or subtract a scalar from an image (returns float64).

    Args:
        image: 2D or 3D numpy array.
        op: ``"add"`` or ``"subtract"``.
        value: scalar to add or subtract. May be negative.

    Returns:
        New ``float64`` array of the same shape as ``image``. Always a
        copy — never a view.

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

## Output Conventions

- **Dtype:** always `float64`. Consistent with `remove_background_*` and the
  Gaussian-blur step. Microscopy data is commonly `uint16` and subtraction can
  produce negatives — float promotion avoids silent underflow.
- **Layer naming:** mirrors sibling sections' suffix style: `(hp s=10)`,
  `(blur s=1)`, `(crop)` → here `(+5)` and `(-412.7)`.
- **Shape preservation:** result has the same shape as input. No clipping, no
  rescaling, no per-frame normalization.

## Explicit Non-Goals

- **No multiply / divide / clip / normalize / threshold operations.** YAGNI —
  add only if asked.
- **No per-frame mean.** The "Mean" button always returns one scalar across
  all dimensions. Per-frame normalization is a separate feature if/when it's
  needed.
- **No worker thread.** A single numpy expression on a `(T, Y, X)` stack is
  effectively instant; no `QThread`, no progress bar, no cancel.
- **No preview button.** Output is deterministic and instant — the result
  appears as a new layer the user can compare directly.
- **No dtype-preservation option.** Output is always `float64`.
- **No "Apply directly with mean" shortcut.** The Mean button only fills the
  field; the user clicks Apply to actually run the operation. This keeps the
  value visible and tweakable before applying.
- **No layer-source dropdown distinct from the Input section.** The widget
  uses the existing `_image_combo`.

## File Changes

| File | Change |
|---|---|
| `src/napari_spotiflow_tracking/_preprocessing.py` | Add the `apply_calculation` pure helper. |
| `src/napari_spotiflow_tracking/_cleanup_widget.py` | Add the Calculation `QGroupBox` to `_setup_ui`, plus `_run_calc_mean()` and `_run_calc_apply()` methods. Import `apply_calculation`. |
| `src/napari_spotiflow_tracking/_tests/test_calculation.py` | New file with unit tests for `apply_calculation`. |
| `src/napari_spotiflow_tracking/_tests/test_cleanup_widget.py` | Extend with widget-level tests for the Calculation section. |

No new dependencies.

## Testing

### Pure helper — `test_calculation.py` (new file)

1. **Add returns sum** — `arr = np.ones((10, 10), dtype=np.uint16)`,
   `apply_calculation(arr, "add", 5.0)` → `np.full((10, 10), 6.0)`.
2. **Subtract returns difference** — `arr = np.full((10, 10), 10, dtype=np.uint16)`,
   `apply_calculation(arr, "subtract", 3.0)` → `np.full((10, 10), 7.0)`.
3. **Output dtype is float64** for any input dtype (`uint8`, `uint16`,
   `float32`).
4. **3D stack** — `(5, 10, 10)` input → result has the same shape, op
   applied uniformly.
5. **Negative subtraction underflow** — `arr = np.zeros((4, 4), dtype=np.uint16)`,
   `apply_calculation(arr, "subtract", 5.0)` → `np.full((4, 4), -5.0)`. Verifies
   no underflow because of the float64 promotion.
6. **Negative value with add** — `apply_calculation(arr, "add", -3.0)` works
   (no special handling needed).
7. **Unknown op raises** — `apply_calculation(arr, "multiply", 2.0)` →
   `ValueError`.
8. **Result is independent of input** — mutating the returned array does
   not modify the input (verifies `astype(np.float64)` produced a copy).

### Widget — extend `test_cleanup_widget.py`

1. **Section exists** — `widget._calc_op_combo`, `widget._calc_value`,
   `widget._calc_apply_btn` are present, with combo items `["Add", "Subtract"]`
   and default `"Add"`.
2. **Mean button populates the spinbox** — Add a `(10, 10)` image of all
   `4.0`. Click `_run_calc_mean`. Spinbox value equals `4.0`.
3. **Apply Add produces correct layer** — Add a `(10, 10)` image of zeros,
   set op to "Add", value to `5`, click `_run_calc_apply`. Layer named
   `"img (+5)"` exists with values all `5.0`.
4. **Apply Subtract produces correct layer** — Same image, op "Subtract",
   value `3`, click. Layer `"img (-3)"` with all `-3.0`.
5. **Apply with no image selected** does nothing — no new layer is added,
   no exception is raised. (Existing `_get_image` short-circuits.)
6. **Mean with no image selected** does nothing — same contract.

## Error Handling Summary

| Condition | Action |
|---|---|
| No image selected (Mean or Apply) | `_get_image()` shows `"No image selected."` and returns. |
| Combo somehow holds an unknown op (defensive only) | `apply_calculation` raises `ValueError`; widget catches and surfaces via `show_error`. (Should not happen from the UI given the fixed combo items, but defended at the helper boundary.) |
