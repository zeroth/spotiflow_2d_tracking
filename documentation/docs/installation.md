# Installation

## Requirements

- Python ≥ 3.9
- A Qt backend for napari (PyQt5 / PySide2 / PyQt6 / PySide6)
- Optional: a CUDA-capable GPU for Spotiflow inference and GPU-accelerated Gaussian fitting

## Install the plugin

From the repository root:

```bash
# Recommended: use a virtual environment
python -m venv .venv
.venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
# .venv/bin/activate             # macOS / Linux

pip install -e .
pip install "napari[all]"        # if you don't already have napari + a Qt backend
```

The `-e` flag installs the plugin in editable mode, so code changes are picked
up without reinstalling.

## Verify

```bash
napari
```

In napari, open **Plugins → Spotiflow 2D Tracking** — you should see four
entries: *Pre-Processing*, *Spot Detection*, *Spot Tracking*, *PAPA*.

## Optional extras

| Feature | Install |
|---|---|
| Spotiflow GPU inference | Install a CUDA-enabled PyTorch matching your driver, then `pip install spotiflow` |
| GPU Gaussian fit (`torchimize`) | `pip install torchimize` (with a CUDA-enabled PyTorch); the plugin auto-detects it and uses batched GPU fitting when a CUDA GPU is present. |
| Noise2Void denoising | Bundled via `spotiflow`'s dependencies; no extra install required. |
| Running tests | `pip install -e ".[testing]"` |

## Project layout (for contributors)

```
src/napari_spotiflow_tracking/
├── napari.yaml                  # plugin manifest (widget registration)
├── _cleanup_widget.py           # Pre-Processing widget
├── _detection_widget.py         # Spot Detection widget
├── _tracking_widget.py          # Spot Tracking widget
├── _papa_widget.py              # PAPA widget
├── _filter_panel.py             # Reusable property-filter sub-panel
├── _segmentation.py             # Detection backends (Spotiflow + LoG)
├── _tracking.py                 # trackpy wrapper
├── _papa_analysis.py            # PAPA cycle analysis
├── _fitting.py                  # Gaussian fit (scipy CPU / torchimize GPU)
├── _preprocessing.py            # Crop, arithmetic, background, blur, average
├── _track_properties.py         # MSD, step detection, intensity sampling
├── _workers.py                  # Background workers (detect / mask / N2V)
└── _tests/                      # pytest suite
```
