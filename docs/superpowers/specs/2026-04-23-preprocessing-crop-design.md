# Pre-Processing Crop Feature — Design

**Date:** 2026-04-23
**Status:** Approved for planning
**Component:** `napari_spotiflow_tracking._cleanup_widget.PreProcessingWidget`

## Summary

Add a **Crop** section to the Pre-Processing widget that lets users crop the
currently selected image using an axis-aligned bounding box derived from a
rectangle drawn in a napari Shapes layer. Rectangles only (no ellipse, polygon,
line, or path).

## Motivation

Many of the team's microscopy acquisitions include large empty regions or
multiple fields of view in one stack. Running spot detection and N2V denoising
on the full frame wastes compute and produces noise far from the biology of
interest. A lightweight ROI-crop step before the other pre-processing operations
lets users focus all downstream work on a region they draw interactively.

## UI Placement

New `QGroupBox("Crop")` inserted in `PreProcessingWidget._setup_ui` **between
the Input group and the Background Removal group**, so it reads as the first
pre-processing step.

```
┌─ Input ────────────────────────────────────┐
│ Image: [combo ▾] [⟳]                       │
└─────────────────────────────────────────────┘
┌─ Crop ─────────────────────────────────────┐
│ Shapes layer: [combo ▾] [⟳] [+ New]        │
│ [Apply Crop]                                │
└─────────────────────────────────────────────┘
┌─ Background Removal ───────────────────────┐
│ ...                                         │
└─────────────────────────────────────────────┘
```

## UI Components

| Widget | Behaviour |
|---|---|
| `QComboBox` `_shapes_combo` | Lists every `napari.layers.Shapes` in the viewer, same pattern as `_image_combo`. |
| `QPushButton` "⟳" (30 px) | Manually refreshes the combo — mirrors the existing Input refresh button. |
| `QPushButton` "+ New" | Creates an empty Shapes layer named `"Crop ROI"`, sets `mode = "add_rectangle"`, makes it the active selected layer, and selects it in `_shapes_combo`. |
| `QPushButton` "Apply Crop" | Runs the crop; adds a new image layer `{layer_name} (crop)`. |

The combo auto-refreshes via the existing `_on_layer_change` hook, which will
be extended to also call a new `_refresh_shapes_combo()` method.

## Behaviour (Apply Crop)

1. **Get inputs.** Selected image via existing `_get_image()`. Selected Shapes
   layer via `_shapes_combo`.
2. **Extract first rectangle.** Iterate `zip(shapes.data, shapes.shape_type)`
   and pick the first entry with `shape_type == "rectangle"`. If none exists,
   show error `"No rectangle found in shapes layer"` via `show_error` and
   return.
3. **Compute axis-aligned bounding box.** Each rectangle in napari is a
   (4, D) array of corner coordinates. Take the last two dimensions (Y, X) and
   compute `y_min, y_max = min/max`, `x_min, x_max = min/max`. This yields the
   AABB of any rectangle, including rotated ones — matching the user's intent
   of "box only".
4. **Round and clip.**
   - `y_min, x_min = max(0, int(round(v)))`
   - `y_max, x_max = min(bound, int(round(v)))` where `bound` is the image's
     Y or X size.
5. **Degenerate check.** If `y_max - y_min <= 0` or `x_max - x_min <= 0`,
   show error `"Crop region is empty (rectangle outside image bounds)"` and
   return.
6. **Slice.** `result = image[..., y_min:y_max, x_min:x_max]`. The `...`
   prefix makes this work for 2D `(Y, X)` and 3D `(T, Y, X)` alike. For image
   dims outside `{2, 3}`, show error `"Expected 2D or 3D (T,Y,X) image, got
   {image.ndim}D."` (same phrasing as sibling pre-processing steps).
7. **Output.** `self.viewer.add_image(result, name=f"{layer_name} (crop)")`.
   Set `show_info("Cropped to Y=[y_min:y_max] X=[x_min:x_max]")` and update
   `_status_label`.

## Explicit Non-Goals

- **Multiple rectangles:** only the first rectangle is used. No union, no
  multi-output, no error on multiple — just take the first.
- **Time subsetting:** the rectangle's time-slice binding is ignored. The
  crop is always applied across all frames of a `(T, Y, X)` stack.
- **Rotated-crop resampling:** rotated rectangles are reduced to their
  axis-aligned bounding box. No affine resampling / re-gridding.
- **Worker thread:** cropping is a pure NumPy slice (a view, effectively
  zero cost). No `QThread`, no progress bar, no cancel button.
- **Preview button:** the operation is deterministic and instant, so no
  quick-preview counterpart is needed.
- **Ellipse / polygon / line / path:** unsupported shape types are silently
  skipped while searching for the first rectangle.

## File Changes

| File | Change |
|---|---|
| `src/napari_spotiflow_tracking/_cleanup_widget.py` | Add Crop `QGroupBox` in `_setup_ui`, `_refresh_shapes_combo()`, `_new_shapes_layer()`, `_run_crop()`. Extend `_on_layer_change` to refresh the shapes combo. `_run_crop()` calls `first_rectangle_bbox` + `crop_image`, wrapping `ValueError` with `show_error`. |
| `src/napari_spotiflow_tracking/_preprocessing.py` | Add two pure helpers: `first_rectangle_bbox(shapes_data, shape_types, image_shape) -> (y_min, y_max, x_min, x_max)` — finds the first rectangle, computes its AABB, rounds, clips to the image, and raises `ValueError` on missing rectangle or empty region. And `crop_image(image, bbox) -> np.ndarray` — slices the image using `[..., y_min:y_max, x_min:x_max]`. Keeps the widget thin and makes the logic unit-testable without Qt. |
| `src/napari_spotiflow_tracking/_tests/test_crop.py` | New file. See Testing. |

No new dependencies — napari, numpy, qtpy are already used.

## Testing

New test file `src/napari_spotiflow_tracking/_tests/test_crop.py`. Tests target
the pure `first_rectangle_bbox` and `crop_image` helpers in `_preprocessing.py`
so they don't need Qt:

1. **2D crop (end-to-end)** — `(100, 100)` image, one rectangle with corners
   `[[10, 20], [10, 80], [60, 80], [60, 20]]` → bbox `(10, 60, 20, 80)`,
   cropped result shape `(50, 60)`.
2. **3D (T,Y,X) crop** — `(5, 100, 100)` stack, same rectangle → result
   shape `(5, 50, 60)`; T preserved.
3. **Clipping** — rectangle corners `[[-10, -10], [-10, 50], [200, 50],
   [200, -10]]` on a `(100, 100)` image → bbox `(0, 100, 0, 50)`, crop shape
   `(100, 50)`.
4. **Out-of-bounds** — rectangle entirely outside image bounds (e.g. corners
   all at y>=200 on a `(100, 100)` image) → `first_rectangle_bbox` raises
   `ValueError`.
5. **Non-rectangle shapes skipped** — `shapes_data = [ellipse_corners,
   rect_corners]`, `shape_types = ["ellipse", "rectangle"]` → uses the
   rectangle.
6. **Rotated rectangle** — 45°-rotated rectangle inscribed in a larger box →
   bbox is the AABB of the 4 corners.
7. **No rectangle present** — `shape_types = ["ellipse"]` → raises
   `ValueError`.
8. **3D shape coords** — rectangle stored with 3-D corner coords (T,Y,X)
   where T is constant → the helper takes the last two dims and produces the
   correct Y/X bbox.

A minimal widget smoke test in the existing test style is optional and can be
added if easy, but the pure-function tests are the contract.

## Error Handling Summary

| Condition | Action |
|---|---|
| No image selected | `show_error("No image selected.")` (existing `_get_image` path). |
| No shapes layer selected | `show_error("No shapes layer selected.")`. |
| Shapes layer has no rectangle | `show_error("No rectangle found in shapes layer.")`. |
| Rectangle clips to empty region | `show_error("Crop region is empty (rectangle outside image bounds).")`. |
| Image is neither 2D nor 3D | `show_error("Expected 2D or 3D (T,Y,X) image, got {n}D.")`. |

All error paths leave the viewer unchanged and return early — matching the
widget's existing style.
