# PAPA

For **cyclical / alternating** experiments where frames belong to one of two
recurring groups (e.g. *DR* vs *PAPA* illumination). The widget groups your
detected spots by cycle pattern and produces per-cycle counts and ratios.

**Source:** [`_papa_widget.py`](https://github.com/) — class `PAPAWidget`.

The widget has two tabs: **Input** (configuration) and **Plots** (results).

---

## Input tab

![PAPA — Input tab](../assets/screenshots/papa--input.png){ width=440 }

### Input

| Field | Notes |
|---|---|
| **Points layer** | Detected spots (must have `frame`, `y`, `x`). Usually the output of *Spot Detection*. |

### Cycle Pattern

Define how frames repeat. A "cycle" is one full `G1 → G2 → (skip)` block.

| Field | Notes |
|---|---|
| **Group 1 size** | Number of frames in group 1 per cycle. |
| **Group 2 size** | Number of frames in group 2 per cycle. |
| **Skip frames** | Number of frames at the **start of each cycle** that are excluded from both groups (e.g. shutter / settling frames). |

Example: with `G1=10, G2=10, Skip=2` and 100 frames you get four full cycles
of length 22 plus a partial trailing cycle. Partial cycles are handled
gracefully.

### Labels

| Field | Notes |
|---|---|
| **Group 1 label** | Default `DR`. Used in plot legends and CSV column names. |
| **Group 2 label** | Default `PAPA`. |

### Action buttons

| Button | Notes |
|---|---|
| **Plot** | Compute the cycle annotations and switch to the **Plots** tab. |
| **Export CSV** | Save the per-frame annotated DataFrame. |

---

## Plots tab

![PAPA — Plots tab](../assets/screenshots/papa--plots.png){ width=440 }

Four stacked figures, each with the matplotlib navigation toolbar:

1. **Particles per frame** — line plot of detected particle count over the
   entire experiment, with vertical separators marking cycle boundaries and
   colored backgrounds for G1 / G2 / skip regions.
2. **Total counts per group per cycle** — bar chart, G1 and G2 side-by-side
   for every cycle.
3. **G1 / G2 ratio** per cycle.
4. **G2 / G1 ratio** per cycle.

---

## Outputs

| Output | Type |
|---|---|
| **CSV export** | One row per frame with columns `frame`, `n_particles`, `cycle`, `group` (`G1` / `G2` / `skip`), and group labels. |

PAPA does **not** add any new napari layers — it's a pure analysis widget.
