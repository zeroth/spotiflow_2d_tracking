# Spot Tracking

Links detected spots across time into tracks using
[`trackpy`](http://soft-matter.github.io/trackpy/), then exposes per-track
analytics:

- **MSD** (mean squared displacement) with anomalous-diffusion fit `D · t^α`
- **Step detection** with adjustable window and threshold
- **Intensity over time** (sampled from a chosen Image layer)

**Source:** [`_tracking_widget.py`](https://github.com/) — class
`TrackingWidget`.

The widget has two tabs: **Control** (link spots into tracks) and **Analysis**
(filtering + per-track plots).

Tracks do not have to come from a run in this session — a `tracks.csv` written
earlier can be opened straight into the viewer. See
[Opening a tracks CSV](#opening-a-tracks-csv).

---

## Control tab

![Spot Tracking — Control tab](../assets/screenshots/spot-tracking--control.png){ width=440 }

### Input

| Field | Notes |
|---|---|
| **Points layer** | The output of *Spot Detection* (must have `frame`, `y`, `x`). |
| **Image layer** *(optional)* | Source for per-vertex intensity sampling along each track. |

### Tracking Parameters

| Field | Notes |
|---|---|
| **Search range** | Maximum pixel jump a particle is allowed to make between consecutive frames. |
| **Memory** | Number of frames a particle is allowed to disappear before its track is broken. |

### Action buttons

| Button | Notes |
|---|---|
| **Track** | Link spots into tracks. Creates a Tracks layer. |
| **Export Tracks to CSV** | Save the full per-vertex DataFrame (one row per `(track_id, frame)`). |

### Filter panel

Filter tracks by `length`, `mean_intensity` (the track's average intensity)
and `max_intensity` (its brightest vertex). Same UI as the Detection widget:
one range-slider per property. The two intensity sliders only appear when an
Image layer was selected for the **Track** run — that is what supplies the
per-vertex intensity samples. `length` also gets a histogram preview; the
intensity columns are filter-only, since the Tracks histogram plots raw
values and their scale would swamp the length bars. The viewer Tracks layer
can mirror the filter via **Show filtered in viewer**.

---

## Analysis tab

![Spot Tracking — Analysis tab](../assets/screenshots/spot-tracking--analysis.png){ width=440 }

### Tracks layer

Which Tracks layer the plots below read. Every Tracks layer carrying
per-vertex data is listed — one produced by **Track**, or one opened from a
CSV. A layer that has just been added is selected automatically, so after a
run or an open the analysis already points at the right one.

### Filtered rows

A table of the tracks currently passing the filter. **Click a row** to drive
the per-track plots below.

### Distributions

- **Track length distribution** — histogram of how long each track survives.
- **Mean speed distribution** — histogram with the formula label
  `mean_speed = (Σ ||pᵢ − pᵢ₋₁||) / (length − 1)`.

### Per-track quick analysis

For the selected row:

- **MSD** chart with an anomalous-diffusion fit overlay
  `MSD(t) = 4·D·t^α`. Failure to fit (e.g. too few points) is handled
  gracefully.
- **Step detection** chart — sliding-window mean step size; controls for
  *Window* and *Threshold* re-plot live.
- **Intensity over time** chart — sampled from the chosen Image layer at each
  track vertex.

Each plot ships with the matplotlib navigation toolbar (zoom / pan / save).

---

## Opening a tracks CSV

The plugin registers a napari **reader** for tracks CSVs, so an earlier result
opens like any other file — drag it onto the viewer, use **File → Open**, or
pass it on the command line:

```bash
napari "path/to/DE 10FPS 01-1/tracks.csv"
```

It reads both flavours the plugin writes: the **Export Tracks to CSV** table,
and the `tracks.csv` that `scripts/bulk_track_analysis.py` writes per stack. A
file needs
`frame`, `y`, `x` and `particle` (or `track_id`); any other CSV is declined
and napari opens it as it did before, so points and shapes tables are
unaffected.

The layer is named after the file's folder when the file is a plain
`tracks.csv`, since that folder names the stack. If a `track_analysis.csv`
sits beside it, its per-track properties — including `D`, `alpha` and
`motility_class` — are loaded with it instead of being recomputed. Rows
missing a coordinate are dropped, and the count is kept in
`metadata["n_dropped_rows"]`.

The result behaves exactly like a freshly tracked layer: it appears in the
**Tracks layer** dropdown and drives the filter panel and every Analysis plot.

---

## Outputs

| Output | Type |
|---|---|
| **Tracked Spots** | napari **Tracks** layer. Per-track properties (`length`, `mean_speed`, etc.) are attached as features; the underlying DataFrame is stored in `metadata["tracked_df"]`. |
| CSV export | Per-vertex DataFrame — one row per `(track_id, frame)` with position, frame, and (if image provided) intensity. |
