"""
Capture screenshots of every widget contributed by `napari-spotiflow-tracking`,
populated with real results from a sample dataset.

What it does
------------
1. Loads `documentation/scripts/data/sample_papa_200_200.tif` into napari as an
   Image layer. By default only the first `--max-frames` frames are loaded
   (keeps a screenshot refresh under a few minutes); pass `--max-frames 0` to
   load the whole stack.

2. Runs each widget end-to-end in order, so later widgets pick up the layers
   produced by earlier ones:

       Pre-Processing   →  open, screenshot   (no actions; data is enough)
       Spot Detection   →  Spotiflow detect, screenshot Control + Analysis
       Spot Tracking    →  trackpy link, screenshot Control + Analysis
       PAPA             →  set group sizes 200/200, plot, screenshot tabs

3. Saves PNGs under `documentation/docs/assets/screenshots/`. Filenames are
   derived from the widget display name (and the tab name, for tabbed
   widgets), so they remain stable as widgets evolve:

       pre-processing.png
       spot-detection--control.png
       spot-detection--analysis.png
       spot-tracking--control.png
       spot-tracking--analysis.png
       papa--input.png
       papa--plots.png

Usage
-----
    .venv/Scripts/python.exe documentation/scripts/capture_widgets.py
    .venv/Scripts/python.exe documentation/scripts/capture_widgets.py --max-frames 0
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #

DOC_ROOT = Path(__file__).resolve().parents[1]                # documentation/
REPO_ROOT = DOC_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
NAPARI_YAML = SRC_ROOT / "napari_spotiflow_tracking" / "napari.yaml"
SCREENSHOT_DIR = DOC_ROOT / "docs" / "assets" / "screenshots"
SAMPLE_TIFF = DOC_ROOT / "scripts" / "data" / "sample_papa_200_200.tif"

# --------------------------------------------------------------------------- #
# Capture knobs                                                               #
# --------------------------------------------------------------------------- #

DOCK_WIDTH = 440         # default width each dock is resized to before grabbing
DOCK_WIDE = 900          # width for tabs with wide matplotlib figures (e.g. PAPA Plots)
DOCK_MIN_HEIGHT = 700    # minimum height so short widgets still look right
DOCK_MAX_HEIGHT = 2400   # absolute cap (tall analysis tabs can be this big)
SETTLE_MS = 250          # ms to pump the event loop after each action

# Default frame window. The full 1826-frame stack covers ~4 full PAPA cycles
# of 400 frames; 800 frames gives two clean cycles and keeps detection fast.
DEFAULT_MAX_FRAMES = 800

# Worker poll timeout (seconds). Spotiflow on GPU should finish well under this.
WORKER_TIMEOUT_S = 1800


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "widget"


def load_widget_manifest() -> list[dict]:
    with NAPARI_YAML.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    widgets = (manifest.get("contributions") or {}).get("widgets") or []
    return [{"display_name": w["display_name"], "command_id": w["command"]} for w in widgets]


def plugin_package_name() -> str:
    with NAPARI_YAML.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    return manifest["name"]


# --------------------------------------------------------------------------- #
# Qt helpers                                                                  #
# --------------------------------------------------------------------------- #


def _pump(app, ms: int = SETTLE_MS) -> None:
    """Pump the Qt event loop for `ms` ms so paints/signals settle."""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait_for_worker(app, worker, timeout_s: float = WORKER_TIMEOUT_S) -> None:
    """Block until a QThread-based worker is no longer running.

    We pump events while waiting so its `finished` signal can fire on the
    main thread (that's the slot that adds layers / updates plots).
    """
    deadline = time.monotonic() + timeout_s
    while worker.isRunning():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Worker did not finish within {timeout_s}s")
        app.processEvents()
        time.sleep(0.05)
    # Drain any tail signals so layer-add slots run before we screenshot.
    _pump(app, ms=300)


def _grab(widget, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    if pix.isNull():
        raise RuntimeError(f"QWidget.grab() returned null pixmap for {out_path}")
    if not pix.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    print(f"  saved {out_path.relative_to(DOC_ROOT)} ({pix.width()}x{pix.height()})")


def _find_tabwidget(widget):
    from qtpy.QtWidgets import QTabWidget
    return widget.findChild(QTabWidget)


def _enforce_canvas_heights(widget, width: int) -> None:
    """Force each visible matplotlib FigureCanvas to honor its figsize.

    Matplotlib's FigureCanvas uses a `Preferred` Qt size policy, which means
    its layout-allocated height can collapse far below the figsize-implied
    aspect ratio when packed alongside other widgets in a QVBoxLayout. That
    is why PAPA's `figsize=(10, 4)` time-series plot rendered as a 60px
    sliver despite a 900px-wide dock.

    We set a `minimumHeight` on each canvas equal to `width * (h/w)` so it
    can't shrink below its natural aspect ratio. Canvases that already have
    an explicit `setFixedHeight(...)` (e.g. tracking widget's distribution
    plots) are left alone.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    for canvas in widget.findChildren(FigureCanvasQTAgg):
        if not canvas.isVisible():
            continue
        # Skip if the canvas already has a fixed height (max < Qt's "no limit").
        if canvas.maximumHeight() < 10_000_000:
            continue
        # Prefer the original figsize we snapshotted in _open_widget. Fall
        # back to whatever the figure reports now (likely already squashed).
        orig = getattr(canvas, "_docgen_orig_figsize", None)
        if orig is not None:
            w_in, h_in = orig
        else:
            try:
                w_in, h_in = canvas.figure.get_size_inches()
            except Exception:
                continue
        if w_in <= 0 or h_in <= 0:
            continue
        usable_w = max(width - 40, 200)
        desired_h = int(usable_w * (h_in / w_in))
        desired_h = max(180, min(desired_h, 900))
        if canvas.minimumHeight() < desired_h:
            canvas.setMinimumHeight(desired_h)


def _fit_dock_to_content(dock, widget, app, width: int = DOCK_WIDTH) -> None:
    """Resize the dock so the active tab's full content is visible (no scrollbar).

    The widgets use `QScrollArea` heavily, so a naive sizeHint() reports the
    viewport size, not the inner content size. We dig out the inner widget of
    any visible QScrollArea, ask IT for its size hint at our target width, and
    resize the dock around that. Falls back to the widget's own sizeHint if
    no scroll area is found.
    """
    from qtpy.QtWidgets import QScrollArea

    widget.setMinimumWidth(width)
    widget.resize(width, max(widget.height(), DOCK_MIN_HEIGHT))
    dock.resize(width, max(dock.height(), DOCK_MIN_HEIGHT))
    _pump(app, ms=50)

    # Force matplotlib canvases to honor their figsize aspect ratio. Must
    # happen before sizeHint queries so the scroll-area inner reports a
    # height that includes the full canvas heights.
    _enforce_canvas_heights(widget, width)
    _pump(app, ms=50)

    inner_h = 0
    # Visible scroll areas only (we don't care about scroll areas in hidden tabs).
    for sa in widget.findChildren(QScrollArea):
        if not sa.isVisible():
            continue
        inner = sa.widget()
        if inner is None:
            continue
        # Force a relayout at our target width so wrapped heights are correct.
        inner.setMinimumWidth(width - 30)
        inner.adjustSize()
        if inner.layout() is not None:
            inner.layout().activate()
        # Try every height signal Qt gives us and take the biggest. Different
        # Qt layouts surface their child mins via different channels:
        #   - sizeHint() ignores child setMinimumHeight on Preferred policies
        #   - minimumSizeHint() picks them up
        #   - layout().sizeHint() sums child sizeHints + spacings/margins
        candidates = [
            inner.sizeHint().height(),
            inner.minimumSizeHint().height(),
        ]
        if inner.layout() is not None:
            candidates.append(inner.layout().sizeHint().height())
            candidates.append(inner.layout().minimumSize().height())
        h = max(candidates)
        if h > inner_h:
            inner_h = h

    # Compute needed height. We can't trust any sizeHint() on tabbed widgets
    # because Qt's QStackedWidget (inside QTabWidget) sizes every page to the
    # max of all pages — so a tall Plots tab pollutes the Input tab's hint.
    # Two cases:
    #   - Tab WITH a scroll area: we have a reliable inner.sizeHint() reading.
    #   - Tab WITHOUT a scroll area: content is small enough to fit in
    #     DOCK_MIN_HEIGHT; don't pad with bogus whitespace.
    if inner_h > 0:
        needed = inner_h + 80
    else:
        needed = DOCK_MIN_HEIGHT
    needed = max(needed, DOCK_MIN_HEIGHT)
    needed = min(needed, DOCK_MAX_HEIGHT)

    dock.resize(width, needed)
    widget.resize(width, needed)
    _pump(app, ms=150)


def _grab_all_tabs(dock, widget, app, slug: str,
                   tab_widths: dict[str, int] | None = None) -> list[Path]:
    """Grab one PNG per tab (or one PNG total if not tabbed).

    Resizes the dock per-tab so each tab gets to show its full content.
    `tab_widths` is an optional `{tab_text: width_px}` map for tabs that need
    a wider dock than the default (e.g. tabs with wide matplotlib figures).
    """
    saved: list[Path] = []
    tab_widths = tab_widths or {}
    tabs = _find_tabwidget(widget)
    if tabs is None or tabs.count() <= 1:
        _fit_dock_to_content(dock, widget, app)
        out = SCREENSHOT_DIR / f"{slug}.png"
        _grab(widget, out)
        saved.append(out)
        return saved

    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        _pump(app)
        tab_text = tabs.tabText(i) or f"tab{i+1}"
        width = tab_widths.get(tab_text, DOCK_WIDTH)
        _fit_dock_to_content(dock, widget, app, width=width)
        tab_slug = slugify(tab_text)
        out = SCREENSHOT_DIR / f"{slug}--{tab_slug}.png"
        _grab(widget, out)
        saved.append(out)
    tabs.setCurrentIndex(0)
    _pump(app, ms=50)
    return saved


def _open_widget(viewer, plugin_name: str, display_name: str):
    dock, widget = viewer.window.add_plugin_dock_widget(
        plugin_name, widget_name=display_name
    )
    # Snapshot original matplotlib figsizes IMMEDIATELY. Once Qt does its
    # first layout pass, FigureCanvasQTAgg's resize event calls
    # `figure.set_size_inches(...)` and the original aspect ratio is lost
    # for good — e.g. PAPA's `figsize=(10, 4)` shrinks to `(8.6, 0.3)` after
    # being squeezed into a narrow dock. We need the originals to compute
    # a sensible per-canvas minimum height later.
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        for canvas in widget.findChildren(FigureCanvasQTAgg):
            try:
                canvas._docgen_orig_figsize = tuple(canvas.figure.get_size_inches())
            except Exception:
                pass
    except Exception:
        pass

    # Detach (float) the dock so it becomes a top-level window and is no
    # longer clamped by the napari main window's height. Without this, even
    # a maximized napari window will silently cap our dock at ~1000 px tall
    # — clipping the PAPA Plots tab's bar charts at the bottom.
    try:
        dock.setFloating(True)
    except Exception:
        pass
    widget.setMinimumWidth(DOCK_WIDTH)
    dock.resize(DOCK_WIDTH, DOCK_MIN_HEIGHT)
    widget.resize(DOCK_WIDTH, DOCK_MIN_HEIGHT)
    return dock, widget


def _select_in_combo(combo, name: str) -> bool:
    """Set a QComboBox to the entry matching `name`. Returns True if found."""
    idx = combo.findText(name)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return True
    return False


# --------------------------------------------------------------------------- #
# Scenes                                                                      #
# --------------------------------------------------------------------------- #


def scene_preprocessing(viewer, app, plugin_name: str) -> list[Path]:
    """Pre-Processing: just show the widget with the sample image loaded."""
    print("Scene: Pre-Processing")
    dock, widget = _open_widget(viewer, plugin_name, "Pre-Processing")
    # Refresh layer combo so the freshly added image appears in the dropdown.
    if hasattr(widget, "_refresh_image_combo"):
        widget._refresh_image_combo()
    _pump(app)
    saved = _grab_all_tabs(dock, widget, app, slugify("Pre-Processing"))
    viewer.window.remove_dock_widget(dock)
    _pump(app, ms=50)
    return saved


def scene_detection(viewer, app, plugin_name: str, image_layer_name: str) -> list[Path]:
    """Spot Detection: select the image, run Spotiflow detect, capture tabs."""
    print("Scene: Spot Detection")
    dock, widget = _open_widget(viewer, plugin_name, "Spot Detection")

    if hasattr(widget, "_refresh_image_combo"):
        widget._refresh_image_combo()
    _pump(app)

    if not _select_in_combo(widget._image_combo, image_layer_name):
        raise RuntimeError(f"Image layer {image_layer_name!r} not found in detection dropdown")

    # Use defaults: Spotiflow / general / auto threshold. GPU box is auto-set
    # by the widget based on torch availability.
    print("  running Spotiflow detection (this may take a while)...")
    widget._detect_btn.click()
    _pump(app, ms=300)

    if widget._worker is not None:
        _wait_for_worker(app, widget._worker)

    # The Filter panel sometimes needs a kick to pick up the newly added
    # "Detected Spots" Points layer.
    if hasattr(widget, "_filter_panel"):
        try:
            widget._filter_panel._refresh_layer_combo()
        except Exception:
            pass
    _pump(app, ms=400)

    saved = _grab_all_tabs(dock, widget, app, slugify("Spot Detection"))
    viewer.window.remove_dock_widget(dock)
    _pump(app, ms=50)
    return saved


def scene_tracking(viewer, app, plugin_name: str,
                   points_layer_name: str, image_layer_name: str) -> list[Path]:
    """Spot Tracking: select points + image, run trackpy, capture tabs."""
    print("Scene: Spot Tracking")
    dock, widget = _open_widget(viewer, plugin_name, "Spot Tracking")

    if hasattr(widget, "_refresh_points_combo"):
        widget._refresh_points_combo()
    if hasattr(widget, "_refresh_image_combo"):
        widget._refresh_image_combo()
    _pump(app)

    if not _select_in_combo(widget._points_combo, points_layer_name):
        raise RuntimeError(f"Points layer {points_layer_name!r} not found")
    _select_in_combo(widget._image_combo, image_layer_name)  # optional

    print("  running trackpy linking...")
    widget._track_btn.click()
    _pump(app, ms=400)

    # Tracking is synchronous (no QThread), but layer-add events still need a
    # turn through the event loop before the filter table populates.
    _pump(app, ms=600)

    # Click the first row of the filtered table to drive the per-track
    # MSD / step / intensity plots so the Analysis tab isn't empty.
    try:
        table = widget._filter_panel.table_view
        proxy = table.model()
        if proxy is not None and proxy.rowCount() > 0:
            idx = proxy.index(0, 0)
            table.selectionModel().setCurrentIndex(
                idx,
                table.selectionModel().ClearAndSelect | table.selectionModel().Rows,
            )
            _pump(app, ms=400)
    except Exception as exc:
        print(f"  (couldn't auto-select track row: {exc})")

    saved = _grab_all_tabs(dock, widget, app, slugify("Spot Tracking"))
    viewer.window.remove_dock_widget(dock)
    _pump(app, ms=50)
    return saved


def scene_papa(viewer, app, plugin_name: str, points_layer_name: str,
               g1_size: int = 200, g2_size: int = 200,
               skip: int = 0) -> list[Path]:
    """PAPA: set 200/200 group sizes, click Plot, capture tabs."""
    print("Scene: PAPA")
    dock, widget = _open_widget(viewer, plugin_name, "PAPA")

    if hasattr(widget, "_refresh_points_combo"):
        widget._refresh_points_combo()
    _pump(app)

    if not _select_in_combo(widget._points_combo, points_layer_name):
        raise RuntimeError(f"Points layer {points_layer_name!r} not found in PAPA dropdown")

    widget._group1_size.setText(str(g1_size))
    widget._group2_size.setText(str(g2_size))
    widget._skip_frames.setText(str(skip))
    _pump(app, ms=100)

    print(f"  computing PAPA cycles (G1={g1_size}, G2={g2_size}, skip={skip})...")
    widget._plot_btn.click()
    _pump(app, ms=600)

    # The Plots tab has a `figsize=(10,4)` time-series figure that collapses
    # at the default 440px dock width — give it a wider canvas.
    saved = _grab_all_tabs(
        dock, widget, app, slugify("PAPA"),
        tab_widths={"Plots": DOCK_WIDE},
    )
    viewer.window.remove_dock_widget(dock)
    _pump(app, ms=50)
    return saved


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #


def load_sample_image(viewer, max_frames: int) -> str:
    import tifffile
    if not SAMPLE_TIFF.exists():
        raise FileNotFoundError(f"Sample TIFF not found: {SAMPLE_TIFF}")

    with tifffile.TiffFile(str(SAMPLE_TIFF)) as tf:
        series = tf.series[0]
        full_shape = series.shape
        print(f"Sample TIFF: {SAMPLE_TIFF.name}  shape={full_shape}  dtype={series.dtype}")

        if max_frames > 0 and full_shape[0] > max_frames:
            print(f"  loading first {max_frames} frames (use --max-frames 0 for full stack)")
            data = series.asarray(key=slice(0, max_frames))
        else:
            print("  loading full stack")
            data = series.asarray()

    name = "Sample (PAPA 200/200)"
    viewer.add_image(data, name=name)
    print(f"  added Image layer {name!r} shape={data.shape}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Cap the loaded stack length. 0 = full stack. Default: {DEFAULT_MAX_FRAMES}.")
    parser.add_argument("--keep-open", action="store_true",
                        help="Don't close napari at the end (handy for debugging).")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip Spotiflow detect (Tracking/PAPA will be empty). For dev only.")
    args = parser.parse_args(argv)

    import napari
    from qtpy.QtWidgets import QApplication

    plugin_name = plugin_package_name()
    widget_names = [w["display_name"] for w in load_widget_manifest()]
    print(f"Plugin: {plugin_name!r}")
    print(f"Widgets: {widget_names}")
    print(f"Output: {SCREENSHOT_DIR}")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=True)
    # Maximize the main window so wide docks (e.g. PAPA Plots tab at 900px)
    # aren't constrained by the parent. A constrained main window will silently
    # clamp the dock back down.
    try:
        viewer.window._qt_window.showMaximized()
    except Exception:
        try:
            viewer.window._qt_window.resize(1800, 1100)
        except Exception:
            pass
    _pump(app, ms=400)

    saved_all: list[Path] = []
    failed: list[str] = []

    image_name = load_sample_image(viewer, args.max_frames)
    _pump(app, ms=200)

    # --- Pre-Processing ---------------------------------------------------
    try:
        saved_all.extend(scene_preprocessing(viewer, app, plugin_name))
    except Exception as exc:
        print(f"  FAILED Pre-Processing: {exc}", file=sys.stderr)
        failed.append("Pre-Processing")

    # --- Detection --------------------------------------------------------
    points_name = "Detected Spots"
    if args.skip_detection:
        # Synthesize a tiny Points layer so Tracking/PAPA have something.
        import numpy as np
        rng = np.random.default_rng(0)
        n_frames = int(viewer.layers[image_name].data.shape[0])
        rows = []
        for f in range(n_frames):
            for _ in range(rng.integers(3, 8)):
                rows.append([f, rng.uniform(50, 600), rng.uniform(50, 400)])
        viewer.add_points(np.asarray(rows), name=points_name, size=6)
        _pump(app, ms=200)
    else:
        try:
            saved_all.extend(scene_detection(viewer, app, plugin_name, image_name))
        except Exception as exc:
            print(f"  FAILED Detection: {exc}", file=sys.stderr)
            failed.append("Spot Detection")

    # --- Tracking ---------------------------------------------------------
    if points_name in viewer.layers:
        try:
            saved_all.extend(scene_tracking(viewer, app, plugin_name, points_name, image_name))
        except Exception as exc:
            print(f"  FAILED Tracking: {exc}", file=sys.stderr)
            failed.append("Spot Tracking")
    else:
        print(f"  skip Tracking — no {points_name!r} layer (detection failed)")
        failed.append("Spot Tracking (skipped — no points)")

    # --- PAPA -------------------------------------------------------------
    if points_name in viewer.layers:
        try:
            saved_all.extend(scene_papa(viewer, app, plugin_name, points_name,
                                        g1_size=200, g2_size=200, skip=0))
        except Exception as exc:
            print(f"  FAILED PAPA: {exc}", file=sys.stderr)
            failed.append("PAPA")
    else:
        print("  skip PAPA — no points layer")
        failed.append("PAPA (skipped — no points)")

    if args.keep_open:
        print("\n--keep-open set; press Ctrl+C to exit.")
        napari.run()
    else:
        viewer.close()
        _pump(app, ms=50)

    print(f"\nCaptured {len(saved_all)} screenshot(s).")
    if failed:
        print(f"FAILED widgets: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
