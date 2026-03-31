from __future__ import annotations

import os
from pathlib import Path

import napari
import napari.layers
import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
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

from napari_spotiflow_tracking._segmentation import load_model
from napari_spotiflow_tracking._workers import DetectionWorker


class DetectionWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: DetectionWorker | None = None
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

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.addItems(["general", "synth_complex", "hybiss", "fluo_live"])
        model_layout.addWidget(self._model_combo)
        custom_row = QHBoxLayout()
        self._custom_model_path = QLabel("")
        self._custom_model_path.setWordWrap(True)
        custom_row.addWidget(self._custom_model_path)
        browse_btn = QPushButton("Custom model...")
        browse_btn.clicked.connect(self._browse_custom_model)
        custom_row.addWidget(browse_btn)
        model_layout.addLayout(custom_row)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()

        row_prob = QHBoxLayout()
        row_prob.addWidget(QLabel("Probability threshold:"))
        self._prob_thresh = QDoubleSpinBox()
        self._prob_thresh.setRange(0.0, 1.0)
        self._prob_thresh.setSingleStep(0.05)
        self._prob_thresh.setValue(0.5)
        row_prob.addWidget(self._prob_thresh)
        params_layout.addLayout(row_prob)

        row_dist = QHBoxLayout()
        row_dist.addWidget(QLabel("Min distance:"))
        self._min_distance = QSpinBox()
        self._min_distance.setRange(1, 50)
        self._min_distance.setValue(2)
        row_dist.addWidget(self._min_distance)
        params_layout.addLayout(row_dist)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Device
        self._use_gpu = QCheckBox("Use GPU (CUDA)")
        try:
            import torch
            self._use_gpu.setChecked(torch.cuda.is_available())
            self._use_gpu.setEnabled(torch.cuda.is_available())
        except ImportError:
            self._use_gpu.setChecked(False)
            self._use_gpu.setEnabled(False)
        layout.addWidget(self._use_gpu)

        # Detect button
        self._detect_btn = QPushButton("Detect Spots")
        self._detect_btn.clicked.connect(self._run_detection)
        layout.addWidget(self._detect_btn)

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

    def _browse_custom_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select custom model folder")
        if path:
            self._custom_model_path.setText(path)
            if self._model_combo.findText("custom") < 0:
                self._model_combo.addItem("custom")
            self._model_combo.setCurrentText("custom")

    def _run_detection(self):
        layer_name = self._image_combo.currentText()
        if not layer_name:
            self._status_label.setText("No image selected.")
            return

        layer = self.viewer.layers[layer_name]
        image = np.asarray(layer.data)

        if image.ndim not in (2, 3):
            self._status_label.setText(
                f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D."
            )
            return

        model_name = self._model_combo.currentText()
        if model_name == "custom":
            model_name = self._custom_model_path.text()
            if not model_name or not Path(model_name).is_dir():
                self._status_label.setText("Custom model path is invalid.")
                return

        device = "cuda" if self._use_gpu.isChecked() else "cpu"
        self._status_label.setText("Loading model...")
        self._detect_btn.setEnabled(False)

        try:
            model = load_model(model_name, device=device)
        except Exception as e:
            self._status_label.setText(f"Model load failed: {e}")
            self._detect_btn.setEnabled(True)
            return

        self._worker = DetectionWorker(
            image=image,
            model=model,
            prob_thresh=self._prob_thresh.value(),
            min_distance=self._min_distance.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.errored.connect(self._on_detection_error)
        self._worker.start()

    def _on_progress(self, stage: str, current: int, total: int):
        self._status_label.setText(f"{stage}: {current}/{total}")

    def _on_detection_finished(self, points: np.ndarray, masks: np.ndarray):
        self._detect_btn.setEnabled(True)
        n_spots = len(points)
        self._status_label.setText(f"Detected {n_spots} spots.")

        if n_spots > 0:
            self.viewer.add_points(
                points,
                name="Detected Spots",
                size=3,
                face_color="red",
            )
        self.viewer.add_labels(masks, name="Spot Masks", opacity=0.4)

    def _on_detection_error(self, msg: str):
        self._detect_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg[:200]}")
