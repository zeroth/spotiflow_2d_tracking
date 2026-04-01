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

        mask_btn_row = QHBoxLayout()
        self._gen_mask_btn = QPushButton("Generate Mask from Points")
        self._gen_mask_btn.clicked.connect(self._generate_mask_from_points)
        mask_btn_row.addWidget(self._gen_mask_btn)
        self._cancel_mask_btn = QPushButton("Cancel")
        self._cancel_mask_btn.clicked.connect(self._cancel_mask_generation)
        self._cancel_mask_btn.setEnabled(False)
        mask_btn_row.addWidget(self._cancel_mask_btn)
        mask_layout.addLayout(mask_btn_row)

        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # ── Spot Filtering ────────────────────────────────────────────
        filter_group = QGroupBox("Spot Filtering")
        filter_layout = QVBoxLayout()

        filter_layout.addWidget(QLabel(
            "Filter spots by region properties (from mask metadata)."
        ))

        # Labels layer selector
        filter_mask_row = QHBoxLayout()
        filter_mask_row.addWidget(QLabel("Mask layer:"))
        self._filter_mask_combo = QComboBox()
        filter_mask_row.addWidget(self._filter_mask_combo)
        filter_layout.addLayout(filter_mask_row)

        # Min/max area
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Area:"))
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 100000)
        self._min_area.setValue(0)
        self._min_area.setPrefix("min: ")
        area_row.addWidget(self._min_area)
        self._max_area = QSpinBox()
        self._max_area.setRange(0, 100000)
        self._max_area.setValue(100000)
        self._max_area.setPrefix("max: ")
        area_row.addWidget(self._max_area)
        filter_layout.addLayout(area_row)

        # Min/max mean intensity
        intens_row = QHBoxLayout()
        intens_row.addWidget(QLabel("Mean intensity:"))
        self._min_intensity = QDoubleSpinBox()
        self._min_intensity.setRange(0, 1e6)
        self._min_intensity.setValue(0)
        self._min_intensity.setPrefix("min: ")
        self._min_intensity.setDecimals(1)
        intens_row.addWidget(self._min_intensity)
        self._max_intensity = QDoubleSpinBox()
        self._max_intensity.setRange(0, 1e6)
        self._max_intensity.setValue(1e6)
        self._max_intensity.setPrefix("max: ")
        self._max_intensity.setDecimals(1)
        intens_row.addWidget(self._max_intensity)
        filter_layout.addLayout(intens_row)

        self._filter_btn = QPushButton("Apply Filter")
        self._filter_btn.clicked.connect(self._apply_filter)
        filter_layout.addWidget(self._filter_btn)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Detect button + quick preview + cancel
        detect_row = QHBoxLayout()
        self._detect_btn = QPushButton("Detect Spots")
        self._detect_btn.clicked.connect(self._run_detection)
        detect_row.addWidget(self._detect_btn)
        self._preview_btn = QPushButton("Quick Preview")
        self._preview_btn.setToolTip(
            "Run detection on the current time point only"
        )
        self._preview_btn.clicked.connect(self._run_preview)
        detect_row.addWidget(self._preview_btn)
        self._cancel_detect_btn = QPushButton("Cancel")
        self._cancel_detect_btn.clicked.connect(self._cancel_detection)
        self._cancel_detect_btn.setEnabled(False)
        detect_row.addWidget(self._cancel_detect_btn)
        layout.addLayout(detect_row)

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
        self._refresh_filter_mask_combo()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_image_combo()
        self._refresh_points_combo()
        self._refresh_filter_mask_combo()

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

    def _refresh_filter_mask_combo(self):
        prev = self._filter_mask_combo.currentText()
        self._filter_mask_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._filter_mask_combo.addItem(layer.name)
        idx = self._filter_mask_combo.findText(prev)
        if idx >= 0:
            self._filter_mask_combo.setCurrentIndex(idx)

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

    def _run_preview(self):
        """Run detection on the current time point only for quick parameter testing."""
        layer_name = self._image_combo.currentText()
        if not layer_name:
            self._status_label.setText("No image selected.")
            return

        layer = self.viewer.layers[layer_name]
        image = np.asarray(layer.data)

        if image.ndim == 2:
            frame_image = image
            frame_idx = None
        elif image.ndim == 3:
            frame_idx = int(self.viewer.dims.current_step[0])
            frame_image = image[frame_idx]
        else:
            self._status_label.setText(
                f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D."
            )
            return

        method = self._method_combo.currentText().lower()
        model = None

        self._detect_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._gen_mask_btn.setEnabled(False)

        if method == "spotiflow":
            model_name = self._model_combo.currentText()
            if model_name == "custom":
                model_name = self._custom_model_path.text()
                if not model_name or not Path(model_name).is_dir():
                    self._status_label.setText("Custom model path is invalid.")
                    self._detect_btn.setEnabled(True)
                    self._preview_btn.setEnabled(True)
                    self._gen_mask_btn.setEnabled(True)
                    return

            device = "cuda" if self._use_gpu.isChecked() else "cpu"
            self._status_label.setText("Loading model...")
            show_info("Loading model for preview...")

            try:
                model = load_model(model_name, device=device)
            except Exception as e:
                self._status_label.setText(f"Model load failed: {e}")
                show_error(f"Model load failed: {e}")
                self._detect_btn.setEnabled(True)
                self._preview_btn.setEnabled(True)
                self._gen_mask_btn.setEnabled(True)
                return
        else:
            suffix = f" (frame {frame_idx})" if frame_idx is not None else ""
            show_info(f"Preview: detecting spots (LoG){suffix}...")

        # Run on single 2D frame
        self._worker = DetectionWorker(
            image=frame_image,
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
        self._cancel_detect_btn.setEnabled(True)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.errored.connect(self._on_detection_error)
        self._worker.start()

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
        self._cancel_detect_btn.setEnabled(True)
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
        self._preview_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._cancel_detect_btn.setEnabled(False)
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
        self._cancel_mask_btn.setEnabled(True)
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

    def _on_mask_finished(self, mask, props_df):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_mask_btn.setEnabled(False)

        mask_layer = self.viewer.add_labels(
            np.asarray(mask), name="Spot Masks", opacity=0.4,
        )
        # Store regionprops in layer metadata for downstream filtering
        mask_layer.metadata["regionprops"] = props_df

        n_regions = len(props_df)
        show_info(f"Done — mask generated ({n_regions} regions with properties).")
        self._status_label.setText(f"Mask generated ({n_regions} regions).")

    def _on_mask_error(self, msg: str):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_mask_btn.setEnabled(False)
        show_error(f"Mask generation failed: {msg[:200]}")
        self._status_label.setText("Mask generation failed.")

    def _cancel_mask_generation(self):
        if self._mask_worker is not None and self._mask_worker.isRunning():
            self._mask_worker.progress.disconnect(self._on_mask_progress)
            self._mask_worker.finished.disconnect(self._on_mask_finished)
            self._mask_worker.errored.disconnect(self._on_mask_error)
            self._mask_worker.requestInterruption()
            self._mask_worker.terminate()
            self._mask_worker.wait(3000)
        self._mask_worker = None
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._gen_mask_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_mask_btn.setEnabled(False)
        show_info("Mask generation cancelled.")
        self._status_label.setText("Mask generation cancelled.")

    # ── Spot Filtering ────────────────────────────────────────────────

    def _apply_filter(self):
        """Filter spots by area and intensity using regionprops from mask metadata."""
        mask_name = self._filter_mask_combo.currentText()
        if not mask_name:
            show_error("No mask layer selected.")
            return

        mask_layer = self.viewer.layers[mask_name]
        props_df = mask_layer.metadata.get("regionprops")
        if props_df is None or len(props_df) == 0:
            show_error("No regionprops in mask metadata. Generate mask first.")
            return

        min_area = self._min_area.value()
        max_area = self._max_area.value()
        min_intens = self._min_intensity.value()
        max_intens = self._max_intensity.value()

        # Filter
        filtered = props_df[
            (props_df["area"] >= min_area)
            & (props_df["area"] <= max_area)
            & (props_df["mean_intensity"] >= min_intens)
            & (props_df["mean_intensity"] <= max_intens)
        ]

        n_before = len(props_df)
        n_after = len(filtered)
        n_removed = n_before - n_after

        # Build filtered Points layer from centroids
        if "frame" in filtered.columns:
            points_data = filtered[["frame", "y", "x"]].to_numpy()
        else:
            points_data = filtered[["y", "x"]].to_numpy()

        self.viewer.add_points(
            points_data,
            name="Filtered Spots",
            size=3,
            face_color="green",
        )

        # Store filtered props in the new points layer metadata
        pts_layers = [l for l in self.viewer.layers if l.name == "Filtered Spots"]
        if pts_layers:
            pts_layers[-1].metadata["regionprops"] = filtered.reset_index(drop=True)

        show_info(f"Filter: kept {n_after}/{n_before} spots (removed {n_removed}).")
        self._status_label.setText(
            f"Filtered: {n_after}/{n_before} spots kept."
        )

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
            df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        elif mask_data.ndim == 3:
            show_info("Computing region properties...")
            frames = []
            for t in progress(range(mask_data.shape[0]), desc="Computing region properties"):
                table = regionprops_table(
                    mask_data[t], intensity_image=image_data[t],
                    properties=properties,
                )
                frame_df = pd.DataFrame(table)
                frame_df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
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
        self._preview_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._cancel_detect_btn.setEnabled(False)
        self._status_label.setText(f"Error: {msg[:200]}")
        show_error(f"Detection failed: {msg[:200]}")

    def _cancel_detection(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.progress.disconnect(self._on_progress)
            self._worker.finished.disconnect(self._on_detection_finished)
            self._worker.errored.disconnect(self._on_detection_error)
            self._worker.requestInterruption()
            self._worker.terminate()
            self._worker.wait(3000)
        self._worker = None
        if self._mask_worker is not None and self._mask_worker.isRunning():
            self._mask_worker.progress.disconnect(self._on_mask_progress)
            self._mask_worker.finished.disconnect(self._on_mask_finished)
            self._mask_worker.errored.disconnect(self._on_mask_error)
            self._mask_worker.requestInterruption()
            self._mask_worker.terminate()
            self._mask_worker.wait(3000)
        self._mask_worker = None
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._detect_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._gen_mask_btn.setEnabled(True)
        self._cancel_detect_btn.setEnabled(False)
        show_info("Cancelled.")
        self._status_label.setText("Cancelled.")
