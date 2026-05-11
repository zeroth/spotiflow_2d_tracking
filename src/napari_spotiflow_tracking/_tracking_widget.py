from __future__ import annotations

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_info, show_error
import pandas as pd
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari_spotiflow_tracking._mpl_theme import style_figure

from napari_spotiflow_tracking._tracking import track_blobs, tracks_to_napari
from napari_spotiflow_tracking._filter_panel import FilterPanel
from napari_spotiflow_tracking._track_properties import compute_track_properties


class TrackingWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_tracked_df: pd.DataFrame | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        self._tabs = QTabWidget()
        outer_layout.addWidget(self._tabs)

        # === Control tab ============================================
        control_tab = QWidget()
        control_outer = QVBoxLayout()
        control_tab.setLayout(control_outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        control_outer.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout()
        inner.setLayout(layout)
        scroll.setWidget(inner)

        self._tabs.addTab(control_tab, "Control")

        # Input layers
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        row.addWidget(self._points_combo)
        refresh_btn = QPushButton("\u21BB")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_points_combo)
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image layer:"))
        self._image_combo = QComboBox()
        img_row.addWidget(self._image_combo)
        img_refresh_btn = QPushButton("\u21BB")
        img_refresh_btn.setFixedWidth(30)
        img_refresh_btn.setToolTip("Refresh layer list (optional, for intensity stats)")
        img_refresh_btn.clicked.connect(self._refresh_image_combo)
        img_row.addWidget(img_refresh_btn)
        input_layout.addLayout(img_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Tracking parameters
        params_group = QGroupBox("Tracking Parameters")
        params_layout = QVBoxLayout()

        row_sr = QHBoxLayout()
        row_sr.addWidget(QLabel("Search range:"))
        self._search_range = QDoubleSpinBox()
        self._search_range.setRange(0.1, 100.0)
        self._search_range.setSingleStep(0.5)
        self._search_range.setValue(2.0)
        row_sr.addWidget(self._search_range)
        params_layout.addLayout(row_sr)

        row_mem = QHBoxLayout()
        row_mem.addWidget(QLabel("Memory:"))
        self._memory = QSpinBox()
        self._memory.setRange(0, 50)
        self._memory.setValue(3)
        row_mem.addWidget(self._memory)
        params_layout.addLayout(row_mem)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Track button
        self._track_btn = QPushButton("Track")
        self._track_btn.clicked.connect(self._run_tracking)
        layout.addWidget(self._track_btn)

        # Export button
        self._export_btn = QPushButton("Export Tracks to CSV")
        self._export_btn.clicked.connect(self._export_tracks)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

        # Status
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Embedded property filter
        self._filter_panel = FilterPanel(self.viewer, "Tracks", "track_id", parent=self)
        layout.addWidget(self._filter_panel)

        layout.addStretch()
        self._refresh_points_combo()
        self._refresh_image_combo()

        # === Analysis tab ===========================================
        self._build_analysis_tab()

    def _build_analysis_tab(self):
        analysis_tab = QWidget()
        analysis_outer = QVBoxLayout()
        analysis_tab.setLayout(analysis_outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        analysis_outer.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout()
        inner.setLayout(layout)
        scroll.setWidget(inner)

        # Filtered rows table — re-parented from the FilterPanel.
        layout.addWidget(QLabel("Filtered rows:"))
        layout.addWidget(self._filter_panel.table_view)

        # Track length distribution.
        layout.addWidget(QLabel("Track length distribution:"))
        self._length_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._length_canvas = FigureCanvas(self._length_fig)
        self._length_canvas.setFixedHeight(220)
        self._length_toolbar = NavigationToolbar(self._length_canvas, self)
        layout.addWidget(self._length_toolbar)
        layout.addWidget(self._length_canvas)

        # Mean-speed distribution (chart labeled "Mean speed", which
        # is the per-track velocity = total displacement / (length-1)).
        layout.addWidget(QLabel("Mean speed distribution:"))
        speed_formula = QLabel(
            "<i>mean_speed = (Σ ‖p<sub>i</sub> − p<sub>i−1</sub>‖) / (length − 1)</i>"
            " &nbsp;— sum of frame-to-frame Euclidean displacements divided"
            " by the number of steps."
        )
        speed_formula.setWordWrap(True)
        speed_formula.setTextFormat(Qt.RichText)
        layout.addWidget(speed_formula)
        self._speed_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._speed_canvas = FigureCanvas(self._speed_fig)
        self._speed_canvas.setFixedHeight(220)
        self._speed_toolbar = NavigationToolbar(self._speed_canvas, self)
        layout.addWidget(self._speed_toolbar)
        layout.addWidget(self._speed_canvas)

        # === Per-track quick analysis ==============================
        layout.addWidget(QLabel("<b>Per-track quick analysis</b>"))
        self._per_track_status = QLabel(
            "Click a row in the Filtered rows table above to analyze a track."
        )
        self._per_track_status.setWordWrap(True)
        layout.addWidget(self._per_track_status)

        # MSD chart.
        layout.addWidget(QLabel("MSD (lag time vs ⟨r²⟩):"))
        self._msd_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._msd_canvas = FigureCanvas(self._msd_fig)
        self._msd_canvas.setFixedHeight(220)
        self._msd_toolbar = NavigationToolbar(self._msd_canvas, self)
        layout.addWidget(self._msd_toolbar)
        layout.addWidget(self._msd_canvas)

        # Step-detection chart with adjustable knobs.
        layout.addWidget(QLabel("Step detection (intensity time series):"))
        knob_row = QHBoxLayout()
        knob_row.addWidget(QLabel("Window:"))
        self._step_window = QSpinBox()
        self._step_window.setRange(1, 200)
        self._step_window.setValue(20)
        self._step_window.setToolTip("Gaussian filter window for derivative")
        self._step_window.valueChanged.connect(self._on_step_params_changed)
        knob_row.addWidget(self._step_window)
        knob_row.addWidget(QLabel("Threshold:"))
        self._step_threshold = QDoubleSpinBox()
        self._step_threshold.setRange(0.0, 1.0)
        self._step_threshold.setSingleStep(0.05)
        self._step_threshold.setDecimals(2)
        self._step_threshold.setValue(0.5)
        self._step_threshold.setToolTip(
            "Normalized peak threshold (0..1) for step detection"
        )
        self._step_threshold.valueChanged.connect(self._on_step_params_changed)
        knob_row.addWidget(self._step_threshold)
        knob_row.addStretch()
        layout.addLayout(knob_row)
        self._step_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._step_canvas = FigureCanvas(self._step_fig)
        self._step_canvas.setFixedHeight(220)
        self._step_toolbar = NavigationToolbar(self._step_canvas, self)
        layout.addWidget(self._step_toolbar)
        layout.addWidget(self._step_canvas)

        # Mean-intensity-over-time chart.
        layout.addWidget(QLabel("Intensity over time:"))
        self._intensity_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._intensity_canvas = FigureCanvas(self._intensity_fig)
        self._intensity_canvas.setFixedHeight(220)
        self._intensity_toolbar = NavigationToolbar(
            self._intensity_canvas, self,
        )
        layout.addWidget(self._intensity_toolbar)
        layout.addWidget(self._intensity_canvas)

        # Cache for the currently-displayed track id.
        self._current_track_id: int | None = None

        # Subscribe to row-click selection on the filter panel's table.
        self._filter_panel.selectionChanged.connect(
            self._on_track_selected
        )

        layout.addStretch()
        self._tabs.addTab(analysis_tab, "Analysis")

        self._filter_panel.filteredDataChanged.connect(
            self._on_filter_data_changed
        )

    def _on_filter_data_changed(self, df):
        self._redraw_length(df)
        self._redraw_speed(df)

    def _redraw_length(self, df):
        fig = self._length_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Length (frames)")
        ax.set_ylabel("Number of tracks")
        if df is None or len(df) == 0 or "length" not in df.columns:
            style_figure(fig)
            self._length_canvas.draw_idle()
            return
        values = df["length"].dropna().to_numpy()
        if values.size > 0:
            ax.hist(values, bins=20, color="#7fcf9f", edgecolor="#3fa05f")
        style_figure(fig)
        self._length_canvas.draw_idle()

    def _redraw_speed(self, df):
        fig = self._speed_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Mean speed (px/frame)")
        ax.set_ylabel("Number of tracks")
        if df is None or len(df) == 0 or "mean_speed" not in df.columns:
            style_figure(fig)
            self._speed_canvas.draw_idle()
            return
        values = df["mean_speed"].dropna().to_numpy()
        if values.size > 0:
            ax.hist(values, bins=20, color="#cf7f7f", edgecolor="#9f3f3f")
        style_figure(fig)
        self._speed_canvas.draw_idle()

    def _on_track_selected(self, track_id: int):
        self._current_track_id = int(track_id)
        self._refresh_per_track_charts()

    def _on_step_params_changed(self, *_):
        # Just redraw the step-detection chart; MSD/intensity don't depend
        # on these knobs.
        if self._current_track_id is None:
            return
        intensity = self._get_track_intensity(self._current_track_id)
        self._redraw_step_chart(intensity)

    def _get_track_subdf(self, track_id: int):
        """Return the chronological per-vertex DataFrame for a track."""
        layers = [
            l for l in self.viewer.layers
            if l.name == "Tracked Spots"
        ]
        if not layers:
            return None
        meta_df = layers[0].metadata.get("tracked_df")
        if not isinstance(meta_df, pd.DataFrame):
            return None
        sub = meta_df[meta_df["particle"] == int(track_id)].sort_values("frame")
        return sub

    def _get_track_intensity(self, track_id: int):
        sub = self._get_track_subdf(track_id)
        if sub is None or "intensity" not in sub.columns:
            return None
        vals = sub["intensity"].to_numpy(dtype=float)
        return vals

    def _refresh_per_track_charts(self):
        if self._current_track_id is None:
            return
        sub = self._get_track_subdf(self._current_track_id)
        if sub is None or len(sub) == 0:
            self._per_track_status.setText(
                f"Track {self._current_track_id} not found in tracked_df."
            )
            return

        n = len(sub)
        self._per_track_status.setText(
            f"Track {self._current_track_id} — {n} vertices."
        )

        coords_xy = sub[["x", "y"]].to_numpy(dtype=float)
        self._redraw_msd_chart(coords_xy)

        intensity = self._get_track_intensity(self._current_track_id)
        self._redraw_step_chart(intensity)
        self._redraw_intensity_chart(intensity)

    def _redraw_msd_chart(self, coords_xy):
        from napari_spotiflow_tracking._track_properties import (
            msd as _msd,
            msd_basic_fit,
        )

        fig = self._msd_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Lag time (frames)")
        ax.set_ylabel("MSD ⟨r²⟩ (px²)")
        if coords_xy.shape[0] <= 5:
            ax.text(
                0.5, 0.5,
                "Track too short (need >5 vertices)",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._msd_canvas.draw_idle()
            return
        try:
            series = _msd(coords_xy, limit=25)
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"MSD failed: {exc}",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._msd_canvas.draw_idle()
            return
        ax.plot(
            series.index.to_numpy(), series.to_numpy(),
            color="#7fcf9f", marker="o", linewidth=1.5,
            label="MSD",
        )
        # Anomalous-diffusion fit: msd = 4 * D * t^alpha
        try:
            fit = msd_basic_fit(series, delta=1.0)
            ax.plot(
                fit["x"], fit["y_fit"],
                color="#cf7f7f", linewidth=2.0, linestyle="--",
                label=(
                    f"fit: D={fit['diffusion_coefficient']:.3g}, "
                    f"α={fit['alpha']:.3g}"
                ),
            )
        except Exception:
            # Fit may diverge on very short or pathological MSD curves —
            # just show the raw data in that case.
            pass
        ax.legend(loc="upper left", fontsize=8)
        style_figure(fig)
        self._msd_canvas.draw_idle()

    def _redraw_step_chart(self, intensity):
        from napari_spotiflow_tracking._track_properties import find_steps

        fig = self._step_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity")
        if intensity is None:
            ax.text(
                0.5, 0.5,
                "No intensity data — re-run Track with an Image layer selected.",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._step_canvas.draw_idle()
            return
        if intensity.size < 3:
            ax.text(
                0.5, 0.5, "Track too short for step detection.",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._step_canvas.draw_idle()
            return
        try:
            window = int(self._step_window.value())
            threshold = float(self._step_threshold.value())
            _, fitx, _ = find_steps(intensity, window=window, threshold=threshold)
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"Step detection failed: {exc}",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._step_canvas.draw_idle()
            return
        x = np.arange(intensity.size)
        ax.plot(x, intensity, color="#cccccc", linewidth=1.0,
                label="raw")
        ax.plot(x, fitx, color="#cf7f7f", linewidth=2.0,
                label="step fit")
        ax.legend(loc="upper right", fontsize=8)
        style_figure(fig)
        self._step_canvas.draw_idle()

    def _redraw_intensity_chart(self, intensity):
        fig = self._intensity_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Frame (index along track)")
        ax.set_ylabel("Intensity")
        if intensity is None or intensity.size == 0:
            ax.text(
                0.5, 0.5,
                "No intensity data — re-run Track with an Image layer selected.",
                transform=ax.transAxes, ha="center", va="center",
            )
            style_figure(fig)
            self._intensity_canvas.draw_idle()
            return
        x = np.arange(intensity.size)
        ax.plot(x, intensity, color="#7fb3ff", marker="o", linewidth=1.0)
        style_figure(fig)
        self._intensity_canvas.draw_idle()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_points_combo()
        self._refresh_image_combo()

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)

    def _refresh_image_combo(self):
        prev = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        idx = self._image_combo.findText(prev)
        if idx >= 0:
            self._image_combo.setCurrentIndex(idx)

    def _run_tracking(self):
        layer_name = self._points_combo.currentText()
        if not layer_name:
            self._status_label.setText("No Points layer selected.")
            show_error("No Points layer selected.")
            return

        layer = self.viewer.layers[layer_name]
        data = np.asarray(layer.data)

        if data.ndim != 2 or data.shape[1] != 3:
            msg = "Points layer must have 3 columns (frame, y, x)."
            self._status_label.setText(msg)
            show_error(msg)
            return

        df = pd.DataFrame(data, columns=["frame", "y", "x"])

        try:
            tracked_df = track_blobs(
                df,
                search_range=self._search_range.value(),
                memory=self._memory.value(),
            )
        except Exception as e:
            self._status_label.setText(f"Tracking failed: {e}")
            show_error(f"Tracking failed: {e}")
            return

        tracks_array = tracks_to_napari(tracked_df)
        n_tracks = tracked_df["particle"].nunique()
        n_frames = int(df["frame"].nunique())

        image_name = self._image_combo.currentText()
        image = None
        if image_name and image_name in self.viewer.layers:
            image = np.asarray(self.viewer.layers[image_name].data)

        try:
            track_props = compute_track_properties(tracked_df, image=image)
        except ValueError as exc:
            show_error(f"Track property compute failed: {exc}; falling back to no image.")
            track_props = compute_track_properties(tracked_df, image=None)

        # Augment tracked_df with per-vertex intensity (when image is given)
        # so the per-track quick analysis can read time series directly.
        if image is not None:
            try:
                if image.ndim == 3:
                    t_max = int(tracked_df["frame"].max())
                    if image.shape[0] <= t_max:
                        raise ValueError(
                            "Image T-axis does not span tracked frames"
                        )
                ys = np.clip(
                    np.rint(tracked_df["y"].to_numpy()).astype(int),
                    0, image.shape[-2] - 1,
                )
                xs = np.clip(
                    np.rint(tracked_df["x"].to_numpy()).astype(int),
                    0, image.shape[-1] - 1,
                )
                if image.ndim == 3:
                    frames = tracked_df["frame"].astype(int).to_numpy()
                    vals = image[frames, ys, xs]
                else:
                    vals = image[ys, xs]
                tracked_df = tracked_df.copy()
                tracked_df["intensity"] = vals.astype(float)
            except Exception as exc:
                show_error(
                    f"Per-vertex intensity sampling failed: {exc}; "
                    "intensity time series will be unavailable."
                )

        self.viewer.add_tracks(
            tracks_array,
            name="Tracked Spots",
            metadata={"properties": track_props, "tracked_df": tracked_df},
        )

        self._last_tracked_df = tracked_df
        self._export_btn.setEnabled(True)
        msg = f"Found {n_tracks} tracks across {n_frames} frames."
        self._status_label.setText(msg)
        show_info(msg)

    def _export_tracks(self):
        if self._last_tracked_df is None or len(self._last_tracked_df) == 0:
            show_error("No tracks to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save tracks CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        self._last_tracked_df.to_csv(path, index=False)
        show_info(f"Exported {self._last_tracked_df['particle'].nunique()} tracks to {path}")
