# napari-spotiflow-tracking

A [napari](https://napari.org) plugin for **2D spot detection and particle tracking** in time-lapse microscopy, combining [Spotiflow](https://github.com/weigertlab/spotiflow) (deep-learning spot detection) and [trackpy](http://soft-matter.github.io/trackpy/) (frame-to-frame linking) with an end-to-end UI for pre-processing, detection, tracking, and cycle analysis.

## Features

The plugin contributes **four dock widgets** under **Plugins → Spotiflow 2D Tracking**:

- **Pre-Processing** — crop, image arithmetic, background removal, Noise2Void denoising, Gaussian blur, and walking-average to produce a clean image stack. Each step adds a new layer, so the chain of layers records the processing history.
- **Spot Detection** — detect spots in 2D images or `(T, Y, X)` stacks using pretrained Spotiflow models (with a classical Laplacian-of-Gaussian fallback), optionally fit elliptical 2D Gaussians and generate per-spot labeled masks, and filter spots non-destructively by any region property (area, intensity, FWHM, solidity, …).
- **Spot Tracking** — link spots across frames with trackpy (displayed as a napari Tracks layer), then explore per-track analytics: MSD with anomalous-diffusion fit, step detection, and intensity profiles.
- **PAPA** — cycle-based comparison of alternating events (e.g. DR vs PAPA): group-wise particle counts and ratios per cycle.

CSV export is available from the Detection, Tracking, and PAPA widgets.

A typical pipeline runs the widgets in order:

```
Pre-Processing  →  Spot Detection  →  Spot Tracking  →  (optional) PAPA
```

## Documentation

Full user documentation (per-widget guides, an end-to-end workflow walkthrough, and screenshots) lives in [`documentation/`](documentation/) and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install -r documentation/requirements.txt
mkdocs serve -f documentation/mkdocs.yml   # live preview at http://127.0.0.1:8000
```

See [documentation/README.md](documentation/README.md) for the full build/preview commands.

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install PyTorch

Install PyTorch **before** the plugin (Spotiflow depends on it). Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the command for your OS and CUDA version.

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# or, e.g. CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install napari with a Qt backend

```bash
pip install "napari[all]"
```

### 4. Install the plugin

From Git (development mode):

```bash
git clone https://github.com/your-username/napari-spotiflow-tracking.git
cd napari-spotiflow-tracking
pip install -e .
```

`pip install -e .` pulls in the remaining dependencies (spotiflow, numpy, pandas, trackpy, scikit-image, scipy, …) declared in `pyproject.toml`, in editable mode so code changes are picked up without reinstalling.

### Optional extras

| Feature | Notes |
|---|---|
| Spotiflow GPU inference | Install a CUDA-enabled PyTorch matching your driver (step 2). |
| GPU Gaussian fit (`torchimize`) | `pip install torchimize` (with a CUDA-enabled PyTorch); the plugin auto-detects it and uses batched GPU fitting when a CUDA GPU is present. |
| Noise2Void denoising | Bundled via Spotiflow's dependencies; no extra install required. |

## Usage

Launch napari and open the widgets from **Plugins → Spotiflow 2D Tracking**:

```bash
napari
```

### Pre-Processing

Pick an Image layer and apply any combination of crop, arithmetic, background removal, Noise2Void denoising, blur, and walking-average. Each operation produces a new image layer.

### Spot Detection

1. Select the (pre-processed) image layer.
2. Choose a method — **Spotiflow** (pick a pretrained or custom model, tune the probability threshold and min distance) or **LoG** (classical fallback).
3. Optionally tick **Generate mask during detection** to fit a 2D Gaussian per spot and produce a Labels layer with region properties.
4. Click **Detect Spots** (or **Quick Preview** for the current frame).
5. Use the **Filter** panel to drop spots by property, the **Analysis** tab for the table and spots-per-frame chart, and **Export Blobs to CSV** to save.

### Spot Tracking

1. Select the Points layer from detection (optionally also the image layer for per-vertex intensity sampling).
2. Set **Search range** (max pixel jump between frames) and **Memory** (frames a spot may disappear before its track breaks).
3. Click **Track** — results appear as a **Tracked Spots** Tracks layer.
4. In the **Analysis** tab, filter tracks and click a row for per-track MSD (with anomalous-diffusion fit), step detection, and intensity plots.
5. **Export Tracks to CSV** to save the full per-vertex DataFrame.

### PAPA

For cyclical / alternating experiments: pick the detected Points layer, set the cycle pattern (group sizes and skip frames), label the groups, and **Plot** to see particles-per-frame and per-cycle bar charts. The first `group 1 size` frames are treated as an excluded lead-in — the readout before the first pulse — and the pattern then starts fresh, so cycle 1 opens with a group-1 block and a cycle runs `G1 → skip → G2 → skip`. Incomplete trailing cycles are not reported. **Export CSV** for the annotated per-frame and per-cycle tables.

## Bulk / headless processing

The [`scripts/`](scripts/) directory contains command-line helpers that reuse the plugin's processing functions without the napari UI — handy for batch runs on a server or overnight jobs. They require the plugin installed in the same environment (`pip install -e .`).

```bash
# Batch: preprocess → detect → fit → mask → track → analyze every
# .tif/.tiff stack in a directory, one output sub-folder per file.
python scripts/bulk_track_analysis.py <input_dir> <output_dir> [options]

# ...or a single stack, with --in instead of an input directory.
python scripts/bulk_track_analysis.py <output_dir> --in <stack.tif> [options]

# Redo the analysis over already-tracked experiments, at new settings.
python scripts/bulk_track_analysis.py --analysis-only <root_dir> [options]

# Re-write only the run-level summary over a root of already-processed
# experiments — no re-detection, no re-tracking.
python scripts/bulk_track_analysis.py --summary-only <root_dir> [options]

# Compare several finished runs against each other — one condition per
# sub-folder of <parent_dir>. Takes no input_dir or output_dir.
python scripts/bulk_track_analysis.py --compare <parent_dir> [options]

# PAPA cycle analysis for a single stack (masks, blob CSV, and graphs).
python scripts/papa_analysis.py <input.tif> <group1_size> <group2_size> [options]
```

Pass `--help` to either script for the full list of options. Give exactly one
of `input_dir` or `--in`; a single-file run writes its own output sub-folder
but no run-level summary, since one stack is not a set of replicates.

### Comparing conditions

`--summary-only` summarises one run; `--compare` puts several finished runs
side by side. Point it at the **parent** of those runs, one sub-folder per
condition, and each sub-folder is treated as a group:

```
Root_Round2/            <- pass this to --compare
├── DE/                 <- one finished run per condition
├── EK/
├── PL/
└── WT/
```

Both run layouts are recognised, so a condition nested as `DE/DE01/<stack>/`
and one flat as `PL/<stack>/` can sit side by side under the same parent.

Every metric is drawn three ways into `<parent>/comparison/`:

| File | What |
|---|---|
| `<root>_combined_<metric>_summary.png` | all conditions overlaid, raw counts |
| `<root>_combined_<metric>_summary_density.png` | the same, y as a fraction of each condition's tracks |
| `harmonised/<GROUP>_<metric>_summary.png` | each condition alone, on the axis range they **all** share |
| `<root>_comparison_summary.csv` | per-condition, per-experiment class counts |

The harmonised figures are the point of the mode: a run-level figure normally
scales to its own data, so two of them cannot be laid side by side and read
directly. Here the x range is the union of what each condition's own
percentile cut would give — no condition's tail is clipped by another's range
— and the y ceiling is the tallest bar any of them reaches.

Because conditions differ in how many tracks they yield, raw counts make the
largest one look dominant; the `_density` variants divide by each condition's
own track count, so the **shapes** are comparable. Read the counts version for
how much data each condition contributed, and the density version for how the
conditions actually differ.

Condition colours are fixed and colour-blind-safe for up to four conditions
overlaid; past that the script warns, since the hues start to collide.

### Choosing a detector

`--detector spotiflow` (the default) runs the deep-learning model, as the Spot
Detection widget does. `--detector log` swaps in the classical Laplacian of
Gaussian instead — no model download, no `torch` inference, and useful when
Spotiflow has not been trained on anything like your spots:

```bash
python scripts/bulk_track_analysis.py in/ out/ --detector log \
    --log-min-sigma 2 --log-max-sigma 10 --log-num-sigma 10 --log-threshold 0.09
```

The `--log-*` defaults match the widget's LoG panel. `--prob-thresh` and
`--min-distance` apply to Spotiflow only, and are ignored under `--detector
log`. Everything downstream — Gaussian fitting, masking, tracking, MSD — is the
same either way, and `run.log` records which detector produced the spots.

### Self-contained copies

`bulk_track_analysis_complete.py` and `papa_analysis_complete.py` are
standalone editions that inline every plugin function they use, so they run on
a machine where `napari-spotiflow-tracking` is not installed (the third-party
packages are still required). They take the same arguments as the originals.

The inlined section of `bulk_track_analysis_complete.py` is **generated** —
after changing `bulk_track_analysis.py` or the plugin's logic modules, run:

```bash
python scripts/_build_bulk_complete.py
```

Editing that copy by hand will be overwritten, and the two would then diverge
without any visible sign.

### MSD and motility classification

`bulk_track_analysis.py` fits every track's mean squared displacement with

```
MSD = 4·D·t^α + 4·σ²
```

and bins the scaling exponent α into **constrained** (α < 0.4), **diffusive**
(0.4 ≤ α ≤ 1.2) or **directed** (α > 1.2), following the motility analysis in
[Commun Biol 4, 137 (2021)](https://www.nature.com/articles/s42003-021-01740-y#Sec9).
Each output sub-folder gains `msd_curves.csv` (per-track MSD per lag, with its
standard error and pair count), `msd_fits.csv` (D, α, class per track) and
seven figures; `D`, `alpha` and `motility_class` are also merged into
`track_analysis.csv`.

| Figure | What it shows |
|---|---|
| `msd_regimes.png` | **Regime map** — normalised MSD on log-log over the three shaded α wedges, so a track's slope places it in its class |
| `msd_combined.png` | All track MSDs coloured by class, with the ensemble mean and its fit |
| `motility_classes.png` | Absolute class counts beside the 100 %-stacked share |
| `d_vs_alpha.png` | Joint distribution of D and α with marginal histograms and the class cuts |
| `msd_construction.png` | How an MSD curve is built — a trajectory with its Δt = τ…5τ displacements, and the MSD ± SEM they average to |
| `msd_archetypes.png` | One representative track per class: MSD ± SEM with its fit, a dashed Brownian reference, and the trajectory inset |
| `msd_shape_comparison.png` | The three class-mean MSD shapes overlaid — concave / straight / convex |

The last three follow Figure 1 of
[Schirripa Spagnolo & Luin, *Int. J. Mol. Sci.* **25**, 8660 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354962/)
(panels a/b, c–e and f respectively), computed from your own tracks. Their
representative track per class is the one whose α is the class median.

A bulk run also writes `motility_summary.png` and `motility_summary.csv` to the
**output root** — one stacked bar per input stack, for comparing conditions
directly.

Every root-level summary file is prefixed with the root folder's own name, so
`experiments/` yields `experiments_motility_summary.png`,
`experiments_d_vs_alpha_summary.png` and so on. They stay distinguishable once
copied out of their run folder and set side by side. Characters that are
awkward in a filename are replaced, so a root called `DE 10FPS s=1` prefixes
with `DE_10FPS_s_1_`.

Both the per-file plots and these root-level figures are cut at
`--min-track-length`, so they always describe the same set of tracks; the CSVs
stay unfiltered.

### Reworking a finished run

Detection and linking are the expensive part of the pipeline and rarely what
you want to change. Two flags rerun only the cheap tail over results you
already have, so trying a different `--min-track-length`, `--msd-max-lag` or
calibration costs seconds instead of a whole re-run:

```bash
# Refit the MSDs and redraw every per-file figure at a longer minimum.
python scripts/bulk_track_analysis.py --analysis-only experiments/ \
    --min-track-length 10 --pixel-size-nm 100 --frame-interval-s 0.5

# Then refresh the root-level figures to match.
python scripts/bulk_track_analysis.py --summary-only experiments/ \
    --min-track-length 10 --pixel-size-nm 100 --frame-interval-s 0.5
```

`--analysis-only` rereads each `tracks.csv` (and the `preprocessed.tif` beside
it, for the intensity columns) and rewrites `track_analysis.csv`,
`msd_curves.csv`, `msd_fits.csv` and the per-file figures. It leaves detection,
linking and the run-level summary alone — run `--summary-only` after it, as
above. Folders are independent: one failure is logged to `run.log` and the rest
carry on.

`--summary-only` writes just those root-level summaries over experiments that
were processed earlier, without re-running anything. It takes no input
directory: point it at a root, and every folder beneath it holding an
`msd_fits.csv` counts as one experiment, labelled by its path relative to the
root. So a root of previous run outputs aggregates across all of them at once:

```bash
python scripts/bulk_track_analysis.py --summary-only experiments/ \
    --pixel-size-nm 100 --frame-interval-s 0.5
```

with `experiments/ctrl/cell01`, `experiments/drug/cell03` and so on becoming
the points. Pass the same calibration flags as the original run — the summary
figures need them for physical units.

`--min-track-length` re-cuts these figures without reprocessing. Raising it
works from the CSVs alone; lowering it past what the original run used cannot
bring back tracks the MSD pass never fitted, which needs `--analysis-only`.

### Opening the results in napari

`tracks.csv` opens straight into the viewer as a Tracks layer — drag it in, use
**File → Open**, or `napari path/to/tracks.csv`. The plugin registers a reader
for it, and the Spot Tracking widget's Analysis tab drives off the opened layer
exactly as it would a fresh run. Where a `track_analysis.csv` sits beside it,
its per-track properties — including `D`, `alpha` and `motility_class` — come
along instead of being recomputed. Any CSV without the track columns is
declined, so other CSVs open as they always did.

In the regime map each curve is divided by its own fitted value at the first
lag, `4·D·t₁^α`, and the `4σ²` floor is subtracted first. Both steps are needed
for the wedges to mean what they look like: on log-log axes a track's class
depends on *slope* alone, while vertical position encodes D, so the wedges are
only a valid partition once every curve shares an anchor.

The σ term is the static localization error, which is only meaningful in
physical units — so supply both flags to work in µm and seconds:

```bash
python scripts/bulk_track_analysis.py in/ out/ \
    --pixel-size-nm 100 --frame-interval-s 0.5 --sigma-nm 35
```

With them, D is reported in µm²/s^α. Without them the fit falls back to
px²/frame^α and σ is dropped (a nanometre figure has no meaning in pixels);
the two flags must be given together. `--msd-max-lag` (default 25 frames) caps
the lag axis, and `--min-track-length` (default 5) sets the shortest track
admitted to the fits — short tracks give noisy α, so consider raising it.

The calibration reaches every output, not just the MSD fits. Given both flags,
each per-track figure is drawn in µm and seconds — `speed_hist.png` in µm/s,
`length_per_track.png` as µm against seconds, `length_hist.png` as a duration
in seconds — and `path_length_hist.png` shows the distance each track walked in
µm. Without them every one of those axes reads px and frames, exactly as it
did before the flags existed.

`track_analysis.csv` gains `duration_s`, `total_displacement_um`,
`net_displacement_um` and `mean_speed_um_s` **beside** the pixel columns rather
than over them. The pixel columns stay because `--summary-only` and `--compare`
read this table back and scale it themselves, and because `length` is a frame
count — the unit `--min-track-length` cuts on.

A track's duration spans the intervals *between* its localisations, so an
n-frame track lasts `(n - 1) × --frame-interval-s`. `length_summary.png` and
`length_vs_speed_summary.png` follow that convention too; a run summarised over
folders written before `duration_s` existed has it recomputed from the pixel
columns, so those figures stay correct without re-running the analysis.

## Running tests

```bash
pip install -e ".[testing]"
pytest
```
