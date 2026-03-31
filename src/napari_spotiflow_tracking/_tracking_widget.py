from __future__ import annotations

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_info, show_error
import pandas as pd
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_spotiflow_tracking._tracking import track_blobs, tracks_to_napari


class TrackingWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_tracked_df: pd.DataFrame | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input points layer
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

        layout.addStretch()
        self._refresh_points_combo()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_points_combo()

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)

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

        self.viewer.add_tracks(tracks_array, name="Tracked Spots")
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
