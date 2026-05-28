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
| GPU Gaussian fit (`gpufit`) | See the [gpufit docs](https://github.com/gpufit/Gpufit); the plugin auto-detects it. |
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

For cyclical / alternating experiments: pick the detected Points layer, set the cycle pattern (group sizes and skip frames), label the groups, and **Plot** to see particles-per-frame and per-cycle bar charts. **Export CSV** for the annotated per-frame table.

## Bulk / headless processing

The [`scripts/`](scripts/) directory contains command-line helpers that reuse the plugin's processing functions without the napari UI — handy for batch runs on a server or overnight jobs. They require the plugin installed in the same environment (`pip install -e .`).

```bash
# Batch: preprocess → detect → fit → mask → track → analyze every
# .tif/.tiff stack in a directory, one output sub-folder per file.
python scripts/bulk_track_analysis.py <input_dir> <output_dir> [options]

# PAPA cycle analysis for a single stack (masks, blob CSV, and graphs).
python scripts/papa_analysis.py <input.tif> <group1_size> <group2_size> [options]
```

Pass `--help` to either script for the full list of options.

## Running tests

```bash
pip install -e ".[testing]"
pytest
```
