# Bulk track analysis CLI — design

**Status:** approved
**Date:** 2026-05-08
**File:** `bulk_track_analysis.py` (project root, replaces existing stub)

## Goal

A standalone CLI that runs the full preprocessing → segmentation → tracking →
track-analysis pipeline over every TIFF stack in a directory, emitting one
sub-folder of artifacts per input file.

## Non-goals

- No GUI / napari integration.
- No new tests for this script (it is a thin CLI over functions already tested
  in the `napari_spotiflow_tracking` package).
- No subcommands — a single batch command.
- No recursive directory traversal.

## CLI surface

```
python bulk_track_analysis.py <input_dir> <output_dir> [options]
```

**Arguments:**
- `input_dir` — directory containing `.tif` / `.tiff` files. Non-recursive.
  Each file must be a 3D `(T, Y, X)` stack; 2D files are skipped with a
  warning.
- `output_dir` — created if missing. For every input `<stem>.tif` we create
  `<output_dir>/<stem>/` containing the per-file artifacts (see Outputs).

**Options (with defaults):**

| Flag | Default | Purpose |
| --- | --- | --- |
| `--walk-window` | `3` | Walking-average window (must be odd; coerced if even). |
| `--no-walk` | off | Skip the walking-average step. |
| `--gauss-sigma` | `1.0` | Gaussian-blur σ applied per frame in (Y, X). |
| `--no-blur` | off | Skip Gaussian blur. |
| `--no-submean` | off | Skip stack-mean subtraction. |
| `--prob-thresh` | `0.5` | Spotiflow probability threshold. |
| `--min-distance` | `2` | Spotiflow min-distance NMS. |
| `--device` | `cuda` | `cuda` or `cpu`. Auto-falls back to CPU if CUDA unavailable. |
| `--fit-win` | `2` | Half-window for `fit_all_frames_gpu`. |
| `--max-radius` | `2.0` | Cap for `make_gpu_spot_mask` spot radius. |
| `--search-range` | `5` | trackpy `search_range`. |
| `--memory` | `3` | trackpy `memory`. |
| `--min-track-length` | `2` | Minimum track length included in plots (CSV is unfiltered). |
| `--skip-existing` | off | Skip an input if its output sub-folder already exists. |
| `--verbose` | off | Print per-stage progress to stderr. |

## Pipeline (per input file)

1. **Load** stack with `tifffile.imread`. Skip with a warning if `ndim != 3`.
2. **Pre-process** (each step is a no-op when its `--no-*` flag is set;
   order: walking average → Gaussian blur → stack-mean subtraction):
   - `walking_average(stack, window=walk_window)` — temporal rolling mean.
     Window is coerced to the next odd value if given as even.
   - Per-frame `scipy.ndimage.gaussian_filter` with `sigma=gauss_sigma`,
     applied to the (Y, X) plane only.
   - `stack_mean = stack.mean(axis=0)` (a 2D image), subtracted from every
     frame. Result is computed in `float64`, then cast back to the input
     dtype (mirrors `apply_calculation` in `_preprocessing.py`):
     - **Integer input** — clip to `np.iinfo(dtype)` range before cast
       (so `uint16` negatives clamp to 0, no wrap-around).
     - **Float input** — direct cast, no clipping.
3. **Segment** — load Spotiflow `general` model **once at startup**. For each
   frame call `model.predict(frame, prob_thresh, min_distance)`. Collect
   `(frame, y, x)` rows into a DataFrame with columns `time, y, x` (the
   column name `time` is what `fit_all_frames_gpu` expects internally;
   it is renamed to `frame` before any CSV is written).
4. **Mask via GPU fit:**
   - `fit_all_frames_gpu(stack, df, win=fit_win)` → DataFrame with
     `amp, x_fit, y_fit, sigma_x, sigma_y, bg`.
   - `make_gpu_spot_mask(stack.shape, fitted_df, intensity_stack=stack,
     max_radius=max_radius)` → `(T, Y, X) int32` mask + DataFrame with
     `area, mean_intensity, max_intensity, spot_radius`.
5. **Track** — `track_blobs(spots_df, search_range, memory)` adds a
   `particle` column (integer track id).
6. **Track analysis** — `compute_track_properties(tracked_df,
   image=preprocessed_stack)` produces one row per track with
   `length, total_displacement, net_displacement, mean_speed,
   mean_intensity, max_intensity, intensity_std`.
7. **Plots** — generated from the tracked DataFrame and the per-track
   property table; the `min_track_length` filter applies to plots only,
   not to the CSVs.
8. **Write outputs** to `<output_dir>/<stem>/`.

The Spotiflow model load happens once outside the per-file loop. If the
model fails to load the script exits before processing.

Per-file exceptions are caught: the file is logged as `FAIL` in `run.log`
with the traceback, and the loop continues to the next file.

## Outputs (per input file)

`<output_dir>/<stem>/`:

- **`mask.tif`** — `(T, Y, X) int32` written with
  `tifffile.imwrite(..., bigtiff=True)`.
- **`spots.csv`** — pooled across all frames. Columns:
  `frame, y, x, sigma_x, sigma_y, amp, x_fit, y_fit, bg,
  area, mean_intensity, max_intensity, spot_radius`. (The DataFrame
  returned by `make_gpu_spot_mask`, with `time` renamed to `frame`.)
- **`tracks.csv`** — `spots.csv` + `particle` column from trackpy. Row
  count may be ≤ `spots.csv` because trackpy drops singletons.
- **`track_analysis.csv`** — one row per track:
  `track_id, length, total_displacement, net_displacement, mean_speed,
  mean_intensity, max_intensity, intensity_std`. **Unfiltered** — the
  `--min-track-length` flag is not applied here.
- **`length_hist.png`** — histogram of `length` across tracks
  (after `min_track_length` filter), 150 DPI.
- **`speed_hist.png`** — histogram of `mean_speed`, 150 DPI.
- **`length_per_track.png`** — one line per track of cumulative
  displacement vs frame. Capped at the top 200 longest tracks (sorted
  by length descending) to keep the plot legible.
- **`speed_per_track.png`** — one line per track of instantaneous speed
  (frame-to-frame Euclidean distance) vs frame. Same 200-track cap.

`<output_dir>/run.log` — one line per processed file:
`<stem>\tn_spots=<int>\tn_tracks=<int>\tstatus=OK|FAIL\t<reason if any>`.

## Code structure

Single file `bulk_track_analysis.py`, ~300–400 lines, no submodules.

```
# stdlib + third-party imports

# ── helpers ─────────────────────────────────────────────────────────
_select_device(prefer)
_walking_average(stack, window)            # copied from _preprocessing.py
_gaussian_blur_stack(stack, sigma)         # scipy.ndimage per-frame
_subtract_stack_mean(stack)                # cast-back-to-dtype variant

# ── segmentation (copied from _segmentation.py + _fitting.py) ──────
_load_spotiflow(name="general", device)
_detect_all_frames(stack, model, prob_thresh, min_dist)
_fit_all_frames_gpu(stack, df, win)
_make_gpu_spot_mask(stack_shape, df, intensity_stack, max_radius)

# ── tracking + analysis (copied from _tracking.py +
#    _track_properties.py) ────────────────────────────────────────────
_track_blobs(spots_df, search_range, memory)
_compute_track_properties(tracked_df, image)

# ── plotting ────────────────────────────────────────────────────────
_plot_length_hist(track_props, out_path)
_plot_speed_hist(track_props, out_path)
_plot_length_per_track(tracked_df, out_path, max_tracks=200)
_plot_speed_per_track(tracked_df, out_path, max_tracks=200)

# ── per-file pipeline ──────────────────────────────────────────────
process_file(tif_path, out_dir, model, args) -> dict   # log row

# ── CLI ─────────────────────────────────────────────────────────────
parse_args(argv)
main(argv=None)
```

Functions are copied (not imported) per the user's preference — the script
is intended to be standalone and shippable without depending on the napari
plugin package being installed.

`print` to stderr (gated by `--verbose`) is the only logging mechanism;
no `logging` module configuration. The structured `run.log` file is
written line-by-line as files complete.

## Dependencies

Already in the project's `.venv`:
`numpy`, `scipy`, `pandas`, `tifffile`, `trackpy`, `spotiflow`, `torch`,
`torchimize`, `matplotlib`, `tqdm` (optional; `tqdm` used only when
`--verbose`).

## Out of scope

- 2D-only inputs (skipped with warning).
- Recursive directory walking.
- Resume on partial failure within a single file (the unit of retry is
  one input file).
- Configurable plot styling beyond the fixed defaults.
- A subcommand interface (preprocess/segment/track/analyze run separately).
