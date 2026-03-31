from __future__ import annotations

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_info, show_error
from napari.utils import progress
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_spotiflow_tracking._preprocessing import (
    remove_background,
    walking_average,
)


class PreProcessingWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input image
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Image:"))
        self._image_combo = QComboBox()
        row.addWidget(self._image_combo)
        refresh_btn = QPushButton("\u21BB")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_image_combo)
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Background removal
        bg_group = QGroupBox("Background Removal")
        bg_layout = QVBoxLayout()

        row_sigma = QHBoxLayout()
        row_sigma.addWidget(QLabel("Sigma:"))
        self._bg_sigma = QDoubleSpinBox()
        self._bg_sigma.setRange(0.5, 200.0)
        self._bg_sigma.setSingleStep(1.0)
        self._bg_sigma.setValue(10.0)
        self._bg_sigma.setToolTip("Gaussian filter sigma — larger values remove broader background")
        row_sigma.addWidget(self._bg_sigma)
        bg_layout.addLayout(row_sigma)

        self._bg_btn = QPushButton("Remove Background")
        self._bg_btn.clicked.connect(self._run_remove_background)
        bg_layout.addWidget(self._bg_btn)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)

        # Walking average
        wa_group = QGroupBox("Walking Average")
        wa_layout = QVBoxLayout()

        row_win = QHBoxLayout()
        row_win.addWidget(QLabel("Window size:"))
        self._window_size = QSpinBox()
        self._window_size.setRange(1, 99)
        self._window_size.setSingleStep(2)
        self._window_size.setValue(3)
        row_win.addWidget(self._window_size)
        wa_layout.addLayout(row_win)

        self._wa_btn = QPushButton("Apply Walking Average")
        self._wa_btn.clicked.connect(self._run_walking_average)
        wa_layout.addWidget(self._wa_btn)

        wa_group.setLayout(wa_layout)
        layout.addWidget(wa_group)

        # Status
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        layout.addStretch()
        self._refresh_image_combo()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_image_combo()

    def _refresh_image_combo(self):
        prev = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        idx = self._image_combo.findText(prev)
        if idx >= 0:
            self._image_combo.setCurrentIndex(idx)

    def _get_image(self):
        """Get the selected image layer data, or None with error shown."""
        layer_name = self._image_combo.currentText()
        if not layer_name:
            show_error("No image selected.")
            return None, None
        layer = self.viewer.layers[layer_name]
        return layer_name, np.asarray(layer.data)

    def _run_remove_background(self):
        layer_name, image = self._get_image()
        if image is None:
            return

        sigma = self._bg_sigma.value()
        show_info("Removing background...")

        if image.ndim == 2:
            result = remove_background(image, sigma=sigma)
        elif image.ndim == 3:
            frames = []
            for t in progress(range(image.shape[0]), desc="Removing background"):
                frames.append(remove_background(image[t], sigma=sigma))
            result = np.stack(frames, axis=0)
        else:
            show_error(f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D.")
            return

        self.viewer.add_image(result, name=f"{layer_name} (bg removed)")
        show_info("Done — background removed.")
        self._status_label.setText("Background removed.")

    def _run_walking_average(self):
        layer_name, image = self._get_image()
        if image is None:
            return

        if image.ndim != 3:
            show_error("Walking average requires a 3D (T,Y,X) stack.")
            return

        window = self._window_size.value()
        show_info(f"Applying walking average (window={window})...")

        result = walking_average(image, window=window)

        self.viewer.add_image(result, name=f"{layer_name} (avg w={window})")
        show_info("Done — walking average applied.")
        self._status_label.setText("Walking average applied.")
