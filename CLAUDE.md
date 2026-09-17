# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`napari-spotiflow-tracking` is a [napari](https://napari.org) plugin for **2D spot detection and particle tracking** in time-lapse microscopy. It combines [Spotiflow](https://github.com/weigertlab/spotiflow) (deep-learning spot detection, with a classical Laplacian-of-Gaussian fallback) and [trackpy](http://soft-matter.github.io/trackpy/) (frame-to-frame linking), exposed as four dock widgets plus headless command-line scripts for bulk processing.

It is a proper Python package (`pyproject.toml`, setuptools) with a `pytest` suite and MkDocs documentation.

## Running

```bash
# Editable install (pulls deps from pyproject.toml); installs the napari plugin
pip install -e .
pip install "napari[all]"   # if napari + a Qt backend aren't already present

napari                       # then open Plugins → Spotiflow 2D Tracking → <widget>
```

The four widgets are registered in `src/napari_spotiflow_tracking/napari.yaml`:
**Pre-Processing**, **Spot Detection**, **Spot Tracking**, **PAPA**. The same
manifest registers a **reader** contribution (`_reader.py`) that opens a tracks
CSV as a Tracks layer — it claims `*.csv` but declines any file without the
track columns, so other CSVs still reach napari's builtin reader.

### Tests

```bash
pip install -e ".[testing]"
pytest                       # testpaths = src/napari_spotiflow_tracking/_tests
```

### Documentation

MkDocs Material site under `documentation/`. Build/preview:

```bash
pip install -r documentation/requirements.txt
mkdocs serve -f documentation/mkdocs.yml
```

### Bulk / headless scripts

`scripts/` holds CLI helpers that reuse the plugin's logic modules with **no UI**.
They require the plugin installed in the same environment.

The config-driven pipeline `spotiflow-batch` (package `batch/`) is the unified
replacement being built in phases (spec:
`docs/superpowers/specs/2026-09-17-unified-batch-pipeline-design.md`):

```bash
spotiflow-batch init-config bleaching > my.yaml   # commented starter config
spotiflow-batch validate my.yaml
spotiflow-batch run my.yaml --set tracking.memory=5
```

```bash
python scripts/bulk_track_analysis.py <input_dir> <output_dir> [options]   # batch pipeline over a folder
python scripts/papa_analysis.py <input.tif> <group1_size> <group2_size> [options]
```

## Architecture

**Key convention: UI is separated from logic.** Each widget module (Qt) is a thin
shell over pure-Python logic modules that have **no Qt imports** and are unit-tested
in isolation. When adding functionality, put the algorithm in a logic module and
call it from the widget (and from the scripts). Heavy/optional dependencies
(`torch`, `spotiflow`, `careamics`) are imported lazily inside functions.

**UI layer (Qt widgets):**
- `_cleanup_widget.py` — Pre-Processing widget
- `_detection_widget.py` — Spot Detection widget
- `_tracking_widget.py` — Spot Tracking widget
- `_papa_widget.py` — PAPA widget
- `_filter_panel.py` — reusable non-destructive property-filter sub-panel
- `_workers.py` — `QThread` background workers (detection, masking, N2V) so long ops don't block the UI
- `_mpl_theme.py` — napari dark-theme styling for embedded matplotlib figures

**Logic layer (pure Python, no Qt — unit-tested):**
- `_preprocessing.py` — crop, scalar arithmetic, background removal (high-pass / erosion), Noise2Void denoising, Gaussian blur, walking average; `_get_device` helper
- `_segmentation.py` — detection backends: Spotiflow (`detect_spots`, `load_model`) and LoG (`detect_spots_log`); `create_spot_mask`, intensity sampling
- `_fitting.py` — 2D Gaussian fitting + labeled masks. scipy backend (default) with an optional batched-GPU `torchimize` backend; stack-wide `fit_all_frames_gpu` / `make_gpu_spot_mask`
- `_tracking.py` — trackpy wrapper (`track_blobs`) + napari Tracks conversion
- `_tracks_io.py` — read a tracks CSV (widget export or the bulk script's
  `tracks.csv`) back into Tracks-layer form; backs the npe2 reader
- `_track_properties.py` — per-track geometry/intensity, MSD (+ anomalous-diffusion fit), step detection
- `_papa_analysis.py` — PAPA cycle logic: `build_cycle_info`, per-frame counts, per-cycle counts/ratios

**Data flow:** detection produces a Points layer (and optional Labels mask); tracking
links it into a Tracks layer; CSV export is available from the Detection, Tracking,
and PAPA widgets. Frames are 0-indexed in napari but 1-indexed (`time`/`frame`) in
the analysis DataFrames and CSVs.

## Key Dependencies

numpy, spotiflow (≥0.6.0), trackpy, scikit-image, scipy, pandas, matplotlib, qtpy, superqt.
Optional: `torchimize` (batched GPU Gaussian fit, auto-detected), `careamics` (Noise2Void, via Spotiflow's deps).
