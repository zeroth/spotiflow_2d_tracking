# Napari Spotiflow Tracking

A [napari](https://napari.org) plugin for **2D spot detection and particle tracking** in time-lapse microscopy.

It combines:

- **[Spotiflow](https://github.com/weigertlab/spotiflow)** — deep-learning spot detection (with classical Laplacian-of-Gaussian as a fallback).
- **[trackpy](http://soft-matter.github.io/trackpy/)** — frame-to-frame linking of detected spots into tracks.
- An end-to-end UI inside napari for **pre-processing**, **detection**, **tracking**, and **PAPA cycle analysis**.

---

## The four widgets

The plugin contributes four dock widgets, each available from napari's
**Plugins** menu under **Spotiflow 2D Tracking**:

| Widget | What it does |
|---|---|
| [**Pre-Processing**](widgets/preprocessing.md) | Crop, arithmetic, background removal, Noise2Void denoising, Gaussian blur, walking-average — get a clean image stack. |
| [**Spot Detection**](widgets/detection.md) | Detect spots (Spotiflow or LoG), optionally fit Gaussians and generate per-spot masks. |
| [**Spot Tracking**](widgets/tracking.md) | Link detected spots across time into tracks using trackpy, plus per-track MSD / step-detection / intensity analytics. |
| [**PAPA**](widgets/papa.md) | Cycle-based comparison of alternating events (e.g. DR vs PAPA) — group-wise counts and ratios per cycle. |

A typical pipeline runs them in order:

```
Pre-Processing  →  Spot Detection  →  Spot Tracking  →  (optional) PAPA
```

See the [Workflow](workflow.md) page for an end-to-end walkthrough.

---

## Quick start

```bash
# Install plugin into your napari environment
pip install -e .

# Launch napari and open the plugin from
# Plugins → Spotiflow 2D Tracking → <widget name>
napari
```

For full setup, see [Installation](installation.md).

---

## About this documentation

These docs are generated with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
Widget screenshots are produced automatically by a script that opens each
widget in a napari viewer and grabs the panel. See
[Maintaining these docs](maintaining.md) for how to regenerate when widgets
change or new ones are added.
