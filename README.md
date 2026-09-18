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

## Batch / headless processing

`spotiflow-batch` runs the same processing without the napari UI, driven by a
YAML config — for batches on a server or overnight jobs. It is installed with
the plugin (`pip install -e .`):

```bash
spotiflow-batch init-config bleaching > my_run.yaml   # commented starter config
spotiflow-batch validate my_run.yaml                  # check config + inputs, run nothing
spotiflow-batch run my_run.yaml                       # process every movie
spotiflow-batch list-outputs                          # figures/tables the analyses can write
```

A run pre-processes each movie, detects spots, optionally fits 2D Gaussians and
masks, links tracks, and writes one folder per movie plus a run-level summary.

### The config

Ready-to-edit examples live in [`examples/batch/`](examples/batch/):
[`two_conditions.yaml`](examples/batch/two_conditions.yaml) (two conditions,
detection + tracking, µm and seconds) and
[`bleaching_single_movie.yaml`](examples/batch/bleaching_single_movie.yaml) (one
movie, photobleaching settings). Copy one, change the paths, and run it. The
full set of keys:

```yaml
modality: bleaching           # tracking | papa | bleaching — a preset of defaults

inputs:                       # each entry is one group; a folder or a single movie
  - {path: 'D:/Data/control', label: Control, pattern: '*.tif', recursive: false}
  - {path: 'D:/Data/drug', label: Drug}
output_dir: 'D:/Results/run1'
skip_existing: false          # resume an interrupted run
device: cuda                  # falls back to cpu with a warning

calibration:                  # both set → lengths in µm, times in s
  pixel_size_nm: null
  frame_interval_s: null

preprocessing:
  steps:                      # applied top to bottom; [] = raw
    - crop: {y: [0, 263], x: [0, 263]}
    - walking_average: {window: 4}
    - subtract: {value: mean}          # add | subtract; a number or "mean"
    - background: {method: high_pass, sigma: 10}   # or {method: erosion, disk_size: 5}
    - n2v: {n_epochs: 10, patch_size: 64}
    - gaussian_blur: {sigma: 1.0}

detection:
  method: spotiflow           # spotiflow | log
  spotiflow: {model: general, prob_thresh: 0.5, min_distance: 2}
  log: {min_sigma: 2, max_sigma: 10, num_sigma: 10, threshold: 0.09}
  fit: {enabled: true, window: 2, max_radius: 2.0}   # 2D Gaussian + masks; torchimize when available, else scipy

tracking:
  enabled: true
  search_range: 3
  memory: 10
  min_track_length: 5

measure_on: raw               # raw | preprocessed — where intensities are read

outputs:
  formats: [png, svg]
  dpi: 300
  save_preprocessed: false
  save_masks: true
```

Any value can be overridden on the command line, so one config covers a sweep:

```bash
spotiflow-batch run my_run.yaml --set tracking.memory=5 --set detection.spotiflow.prob_thresh=0.4
```

Values resolve as built-in defaults < modality preset < your YAML < `--set`.
Unknown keys warn and are ignored; wrong types, missing inputs and impossible
combinations are reported together, before any movie is read. `validate` runs
exactly those checks and nothing else.

### Modalities

`modality` picks a preset of defaults, which your config then overrides:

| Modality | Preprocessing | Tracking |
|---|---|---|
| `tracking` | walking average 4 → subtract mean → blur σ 1 | on, search 5, memory 3 |
| `papa` | none | off |
| `bleaching` | subtract mean → blur σ 1 | on, search 3, memory 10 |

### Choosing a detector

`method: spotiflow` runs the deep-learning model, as the Spot Detection widget
does. `method: log` swaps in the classical Laplacian of Gaussian — no model
download and no `torch` inference, useful when Spotiflow has not been trained on
anything like your spots. The `log:` defaults match the widget's LoG panel, and
the `spotiflow:` settings are ignored under `method: log`. Everything
downstream is the same either way, and `run.log` records what ran.

### Outputs

```
<output_dir>/
  resolved_config.yaml   every value actually used, including merged defaults
  run_summary.csv        one row per movie: group, status, n_spots, n_tracks, runtime, error
  run_overview.png       spots and tracks per movie, coloured by group
  run.log
  <group>/<movie>/
    spots.csv, tracks.csv
    preprocessed.tif                 only when save_preprocessed: true
    mask.tif                         written unless save_masks: false
    movie_status.json               marks the movie complete, for skip_existing
```

Every spot is fitted with a 2D Gaussian by default (`detection.fit.enabled: true`), so
`spots.csv` carries the fit and region-property columns (`amp`, `x_fit`, `y_fit`,
`sigma_x`, `sigma_y`, `bg`, `area`, `mean_intensity`, `max_intensity`,
`spot_radius`), and `mask.tif` is written unless `outputs.save_masks` is set to
`false`. Fitting uses the batched `torchimize` backend on a GPU device when it is
importable, and a per-frame `scipy` fallback otherwise (including whenever
`device: cpu`, since the batched solver's CPU path has a known numerical-stability
issue), so it needs no extra install to be on by default. The two backends produce
the same schema and closely matching values — not bit-identical, since the scipy
fit bounds its search while the batched fit does not — and the fallback is slower
(roughly 5x on one 15,457-spot, 827-frame movie in testing) but not prohibitively so.

One movie failing does not stop the run: its traceback goes to `run.log`, its
row reads `FAIL`, and the rest carry on. The command exits 0 when everything
worked, 1 when any movie failed, and 2 when the config itself is wrong — so a
wrapping script can tell the difference. With `skip_existing: true` a rerun
skips movies that finished and reuses spots and tracks for movies that did not.

### Opening the results in napari

`tracks.csv` opens straight into the viewer as a Tracks layer — drag it in, use
**File → Open**, or `napari path/to/tracks.csv`. The plugin registers a reader
for it, and the Spot Tracking widget's Analysis tab drives off the opened layer
exactly as it would a fresh run. Where a per-track properties CSV sits beside
it, those properties come along instead of being recomputed. Any CSV without
the track columns is declined, so other CSVs open as they always did.

### Analyses

Detection and tracking run once per movie, and every enabled analysis reads
those results, writing tables and figures per movie, per group, and across
groups. `spotiflow-batch list-outputs <name>` lists what an analysis can write,
and `figures`/`tables` select from it (`all`, `none`, or a list of names).

**`msd`** — mean squared displacement and motility classification, enabled by
the `tracking` preset:

```yaml
analyses:
  msd:
    max_lag: 25           # longest lag fitted, in frames
    min_length: 5         # shortest track admitted to the fits
    sigma_nm: 35          # static localization error, used only with calibration
    max_tracks_plot: 200  # cap for the per-track figures
    figures: all
    tables: all
```

Every track's MSD is fitted with `MSD = 4·D·t^α + 4·σ²`, and α bins the track
into **constrained** (α < 0.4), **diffusive** (0.4 ≤ α ≤ 1.2) or **directed**
(α > 1.2), following the motility analysis in
[Commun Biol 4, 137 (2021)](https://www.nature.com/articles/s42003-021-01740-y#Sec9).
Per movie you get `msd_curves.csv`, `msd_fits.csv` and `track_analysis.csv`
beside 15 figures — the regime map, the combined MSD curves, class shares, the
D–α joint distribution, length/speed/displacement distributions, and the three
explanatory figures (construction, archetypes, shape comparison) that follow
Figure 1 of
[Schirripa Spagnolo & Luin, *Int. J. Mol. Sci.* **25**, 8660 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354962/).

Each group gets the same metrics pooled over its movies in
`<group>/_summary/msd/`, and a run with more than one group also writes
`_comparison/msd/`: overlaid group distributions, their density versions, the
harmonised per-group figures on a shared axis range, and
`comparison_summary.csv`. Group colours are assigned in `inputs` order and are
the same in every figure of the run.

With `calibration` set, D is reported in µm²/s^α and every axis reads µm and
seconds; without it, px and frames (and σ is dropped, since a nanometre figure
has no meaning in pixels).

PAPA cycle counts and photobleaching-step analysis are being ported next.

## Running tests

```bash
pip install -e ".[testing]"
pytest
```
