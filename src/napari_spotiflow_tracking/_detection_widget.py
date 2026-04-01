from __future__ import annotations

import os
from pathlib import Path

import napari
import napari.layers
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_error
from napari.utils import progress
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
from napari_spotiflow_tracking._workers import DetectionWorker, MaskGenerationWorker


class DetectionWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: DetectionWorker | None = None
        self._mask_worker: MaskGenerationWorker | None = None
        self._last_points: np.ndarray | None = None
        self._pbr = None  # napari progress bar
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

        # Detection method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Spotiflow", "LoG"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        layout.addLayout(method_row)

        # Spotiflow parameters
        self._spotiflow_group = QGroupBox("Spotiflow Parameters")
        spoti_layout = QVBoxLayout()

        # Model selection
        spoti_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["general", "synth_complex", "hybiss", "fluo_live"])
        spoti_layout.addWidget(self._model_combo)
        custom_row = QHBoxLayout()
        self._custom_model_path = QLabel("")
        self._custom_model_path.setWordWrap(True)
        custom_row.addWidget(self._custom_model_path)
        browse_btn = QPushButton("Custom model...")
        browse_btn.clicked.connect(self._browse_custom_model)
        custom_row.addWidget(browse_btn)
        spoti_layout.addLayout(custom_row)

        row_prob = QHBoxLayout()
        self._auto_prob_cb = QCheckBox("Auto")
        self._auto_prob_cb.setChecked(True)
        self._auto_prob_cb.setToolTip("Let Spotiflow choose the probability threshold automatically")
        self._auto_prob_cb.toggled.connect(lambda checked: self._prob_thresh.setEnabled(not checked))
        row_prob.addWidget(QLabel("Probability threshold:"))
        self._prob_thresh = QDoubleSpinBox()
        self._prob_thresh.setRange(0.0, 1.0)
        self._prob_thresh.setSingleStep(0.05)
        self._prob_thresh.setValue(0.5)
        self._prob_thresh.setEnabled(False)
        row_prob.addWidget(self._prob_thresh)
        row_prob.addWidget(self._auto_prob_cb)
        spoti_layout.addLayout(row_prob)

        row_dist = QHBoxLayout()
        row_dist.addWidget(QLabel("Min distance:"))
        self._min_distance = QSpinBox()
        self._min_distance.setRange(1, 50)
        self._min_distance.setValue(2)
        row_dist.addWidget(self._min_distance)
        spoti_layout.addLayout(row_dist)

        # Device
        self._use_gpu = QCheckBox("Use GPU (CUDA)")
        try:
            import torch
            self._use_gpu.setChecked(torch.cuda.is_available())
            self._use_gpu.setEnabled(torch.cuda.is_available())
        except ImportError:
            self._use_gpu.setChecked(False)
            self._use_gpu.setEnabled(False)
        spoti_layout.addWidget(self._use_gpu)

        self._spotiflow_group.setLayout(spoti_layout)
        layout.addWidget(self._spotiflow_group)

        # LoG parameters
        self._log_group = QGroupBox("LoG Parameters")
        log_layout = QVBoxLayout()

        row_min_s = QHBoxLayout()
        row_min_s.addWidget(QLabel("Min sigma:"))
        self._log_min_sigma = QDoubleSpinBox()
        self._log_min_sigma.setRange(0.1, 50.0)
        self._log_min_sigma.setValue(2.0)
        row_min_s.addWidget(self._log_min_sigma)
        log_layout.addLayout(row_min_s)

        row_max_s = QHBoxLayout()
        row_max_s.addWidget(QLabel("Max sigma:"))
        self._log_max_sigma = QDoubleSpinBox()
        self._log_max_sigma.setRange(0.1, 100.0)
        self._log_max_sigma.setValue(10.0)
        row_max_s.addWidget(self._log_max_sigma)
        log_layout.addLayout(row_max_s)

        row_num_s = QHBoxLayout()
        row_num_s.addWidget(QLabel("Num sigma:"))
        self._log_num_sigma = QSpinBox()
        self._log_num_sigma.setRange(1, 50)
        self._log_num_sigma.setValue(10)
        row_num_s.addWidget(self._log_num_sigma)
        log_layout.addLayout(row_num_s)

        row_thresh = QHBoxLayout()
        row_thresh.addWidget(QLabel("Threshold:"))
        self._log_threshold = QDoubleSpinBox()
        self._log_threshold.setRange(0.001, 1.0)
        self._log_threshold.setSingleStep(0.01)
        self._log_threshold.setDecimals(3)
        self._log_threshold.setValue(0.09)
        row_thresh.addWidget(self._log_threshold)
        log_layout.addLayout(row_thresh)

        self._log_group.setLayout(log_layout)
        self._log_group.setVisible(False)
        layout.addWidget(self._log_group)

        # Mask generation
        mask_group = QGroupBox("Mask Generation")
        mask_layout = QVBoxLayout()

        self._generate_mask_cb = QCheckBox("Generate mask during detection")
        self._generate_mask_cb.setChecked(False)
        mask_layout.addWidget(self._generate_mask_cb)

        # Fitting backend
        from napari_spotiflow_tracking._fitting import _available_backends
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Fitting backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(_available_backends())
        self._backend_combo.setToolTip(
            "scipy: CPU parallel | jaxfit: GPU via JAX | gpufit: CUDA batched"
        )
        backend_row.addWidget(self._backend_combo)
        mask_layout.addLayout(backend_row)

        # Generate mask from existing Points layer
        mask_row = QHBoxLayout()
        mask_row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        mask_row.addWidget(self._points_combo)
        mask_layout.addLayout(mask_row)

        self._gen_mask_btn = QPushButton("Generate Mask from Points")
        self._gen_mask_btn.clicked.connect(self._generate_mask_from_points)
        mask_layout.addWidget(self._gen_mask_btn)

        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # Detect button
        self._detect_btn = QPushButton("Detect Spots")
        self._detect_btn.clicked.connect(self._run_detection)
        layout.addWidget(self._detect_btn)

        # Export button
        self._export_btn = QPushButton("Export Blobs to CSV")
        self._export_btn.clicked.connect(self._export_blobs)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

        # Status
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        layout.addStretch()
        self._refresh_image_combo()
        self._refresh_points_combo()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_image_combo()
        self._refresh_points_combo()

    def _refresh_image_combo(self):
        prev = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        idx = self._image_combo.findText(prev)
        if idx >= 0:
            self._image_combo.setCurrentIndex(idx)

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)

    def _on_method_changed(self, method: str):
        is_spotiflow = method == "Spotiflow"
        self._spotiflow_group.setVisible(is_spotiflow)
        self._log_group.setVisible(not is_spotiflow)

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

        method = self._method_combo.currentText().lower()
        model = None

        self._detect_btn.setEnabled(False)
        self._gen_mask_btn.setEnabled(False)

        if method == "spotiflow":
            model_name = self._model_combo.currentText()
            if model_name == "custom":
                model_name = self._custom_model_path.text()
                if not model_name or not Path(model_name).is_dir():
                    self._status_label.setText("Custom model path is invalid.")
                    self._detect_btn.setEnabled(True)
                    self._gen_mask_btn.setEnabled(True)
                    return

            device = "cuda" if self._use_gpu.isChecked() else "cpu"
            self._status_label.setText("Loading model...")
            show_info("Loading model...")

            try:
                model = load_model(model_name, device=device)
            except Exception as e:
                self._status_label.setText(f"Model load failed: {e}")
                show_error(f"Model load failed: {e}")
                self._detect_btn.setEnabled(True)
                self._gen_mask_btn.setEnabled(True)
                return
        else:
            self._status_label.setText("Detecting spots (LoG)...")
            show_info("Detecting spots (LoG)...")

        self._worker = DetectionWorker(
            image=image,
            model=model,
            method=method,
            prob_thresh=None if self._auto_prob_cb.isChecked() else self._prob_thresh.value(),
            min_distance=self._min_distance.value(),
            generate_mask=self._generate_mask_cb.isChecked(),
            fit_backend=self._backend_combo.currentText(),
            log_min_sigma=self._log_min_sigma.value(),
            log_max_sigma=self._log_max_sigma.value(),
            log_num_sigma=self._log_num_sigma.value(),
            log_threshold=self._log_threshold.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.errored.connect(self._on_detection_error)
        self._worker.start()

    def _on_progress(self, stage: str, current: int, total: int):
        self._status_label.setText(f"{stage}: {current}/{total}")
        if self._pbr is None and total > 1:
            self._pbr = progress(total=total, desc=stage)
        if self._pbr is not None:
            self._pbr.set_description(stage)
            self._pbr.n = current
            self._pbr.refresh()

    def _on_detection_finished(self, points: np.ndarray, masks):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._detect_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._last_points = points
        self._export_btn.setEnabled(len(points) > 0)
        n_spots = len(points)
        self._status_label.setText(f"Detected {n_spots} spots.")
        show_info(f"Done — detected {n_spots} spots.")

        if n_spots > 0:
            self.viewer.add_points(
                points,
                name="Detected Spots",
                size=3,
                face_color="red",
            )
        if masks is not None:
            self.viewer.add_labels(masks, name="Spot Masks", opacity=0.4)

    def _generate_mask_from_points(self):
        """Generate a mask from an existing Points layer using Gaussian fitting."""
        points_name = self._points_combo.currentText()
        if not points_name:
            show_error("No Points layer selected.")
            return

        image_name = self._image_combo.currentText()
        if not image_name:
            show_error("No image selected (needed for Gaussian fitting).")
            return

        points_layer = self.viewer.layers[points_name]
        image_layer = self.viewer.layers[image_name]
        points_data = np.asarray(points_layer.data)
        image_data = np.asarray(image_layer.data)

        self._gen_mask_btn.setEnabled(False)
        self._detect_btn.setEnabled(False)
        show_info("Generating masks...")

        self._mask_worker = MaskGenerationWorker(
            image=image_data,
            points=points_data,
            fit_backend=self._backend_combo.currentText(),
        )
        self._mask_worker.progress.connect(self._on_mask_progress)
        self._mask_worker.finished.connect(self._on_mask_finished)
        self._mask_worker.errored.connect(self._on_mask_error)
        self._mask_worker.start()

    def _on_mask_progress(self, stage: str, current: int, total: int):
        self._status_label.setText(f"{stage}: {current}/{total}")
        if self._pbr is None and total > 1:
            self._pbr = progress(total=total, desc=stage)
        if self._pbr is not None:
            self._pbr.set_description(stage)
            self._pbr.n = current
            self._pbr.refresh()

    def _on_mask_finished(self, mask):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self.viewer.add_labels(np.asarray(mask), name="Spot Masks", opacity=0.4)
        show_info("Done — mask generated.")
        self._status_label.setText("Mask generated.")

    def _on_mask_error(self, msg: str):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        show_error(f"Mask generation failed: {msg[:200]}")
        self._status_label.setText("Mask generation failed.")

    def _export_blobs(self):
        if self._last_points is None or len(self._last_points) == 0:
            show_error("No blobs to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save blobs CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        # Check if mask and image layers exist for region properties
        mask_layer = None
        image_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels) and "Spot Masks" in layer.name:
                mask_layer = layer
            if isinstance(layer, napari.layers.Image) and layer.name == self._image_combo.currentText():
                image_layer = layer

        if mask_layer is not None and image_layer is not None:
            self._export_with_regionprops(path, mask_layer, image_layer)
        else:
            # Fallback: export coordinates only
            if self._last_points.shape[1] == 3:
                columns = ["frame", "y", "x"]
            else:
                columns = ["y", "x"]
            df = pd.DataFrame(self._last_points, columns=columns)
            df.to_csv(path, index=False)
            show_info(f"Exported {len(df)} blobs to {path}")

    def _export_with_regionprops(self, path: str, mask_layer, image_layer):
        """Export blobs with region properties from mask and image."""
        from skimage.measure import regionprops_table

        properties = [
            "label", "area", "centroid", "mean_intensity", "max_intensity",
            "min_intensity", "bbox", "equivalent_diameter", "perimeter",
            "solidity",
        ]

        mask_data = np.asarray(mask_layer.data)
        image_data = np.asarray(image_layer.data)

        if mask_data.ndim == 2:
            show_info("Computing region properties...")
            table = regionprops_table(
                mask_data, intensity_image=image_data, properties=properties,
            )
            df = pd.DataFrame(table)
        elif mask_data.ndim == 3:
            show_info("Computing region properties...")
            frames = []
            for t in progress(range(mask_data.shape[0]), desc="Computing region properties"):
                table = regionprops_table(
                    mask_data[t], intensity_image=image_data[t],
                    properties=properties,
                )
                frame_df = pd.DataFrame(table)
                frame_df.insert(0, "frame", t)
                frames.append(frame_df)
            df = pd.concat(frames, ignore_index=True)
        else:
            show_error(f"Unsupported mask dimensions: {mask_data.ndim}D")
            return

        df.to_csv(path, index=False)
        show_info(f"Exported {len(df)} blobs with region properties to {path}")

    def _on_detection_error(self, msg: str):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._detect_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg[:200]}")
        show_error(f"Detection failed: {msg[:200]}")
