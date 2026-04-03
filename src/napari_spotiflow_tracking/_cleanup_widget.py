from __future__ import annotations

import napari
import napari.layers
import numpy as np
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

from napari_spotiflow_tracking._preprocessing import (
    remove_background_high_pass,
    remove_background_high_pass_stack,
    remove_background_erosion,
    walking_average,
)
from napari_spotiflow_tracking._workers import N2VWorker


class PreProcessingWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._n2v_worker: N2VWorker | None = None
        self._n2v_layer_name: str | None = None
        self._pbr = None
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

        # ── Background Removal ────────────────────────────────────────
        bg_group = QGroupBox("Background Removal")
        bg_layout = QVBoxLayout()

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._bg_method = QComboBox()
        self._bg_method.addItems(["High-pass (Gaussian)", "Erosion (Morphological)"])
        self._bg_method.currentTextChanged.connect(self._on_bg_method_changed)
        method_row.addWidget(self._bg_method)
        bg_layout.addLayout(method_row)

        # High-pass params
        self._highpass_row = QHBoxLayout()
        self._highpass_row_label = QLabel("Sigma:")
        self._highpass_row.addWidget(self._highpass_row_label)
        self._bg_sigma = QDoubleSpinBox()
        self._bg_sigma.setRange(0.5, 200.0)
        self._bg_sigma.setSingleStep(1.0)
        self._bg_sigma.setValue(10.0)
        self._bg_sigma.setToolTip("Gaussian filter sigma — larger values remove broader background")
        self._highpass_row.addWidget(self._bg_sigma)
        bg_layout.addLayout(self._highpass_row)

        # Erosion params
        self._erosion_row = QHBoxLayout()
        self._erosion_row_label = QLabel("Disk size:")
        self._erosion_row.addWidget(self._erosion_row_label)
        self._disk_size = QSpinBox()
        self._disk_size.setRange(1, 100)
        self._disk_size.setValue(10)
        self._disk_size.setToolTip("Radius of disk structuring element for erosion")
        self._erosion_row.addWidget(self._disk_size)
        bg_layout.addLayout(self._erosion_row)
        # Hide erosion params by default
        self._erosion_row_label.setVisible(False)
        self._disk_size.setVisible(False)

        self._bg_btn = QPushButton("Remove Background")
        self._bg_btn.clicked.connect(self._run_remove_background)
        bg_layout.addWidget(self._bg_btn)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)

        # ── Noise2Void ────────────────────────────────────────────────
        n2v_group = QGroupBox("Noise2Void Denoising")
        n2v_layout = QVBoxLayout()

        # Model path (optional)
        model_row = QHBoxLayout()
        self._n2v_model_path = QLabel("(train from scratch)")
        self._n2v_model_path.setWordWrap(True)
        model_row.addWidget(self._n2v_model_path)
        browse_btn = QPushButton("Load model...")
        browse_btn.clicked.connect(self._browse_n2v_model)
        model_row.addWidget(browse_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self._clear_n2v_model)
        model_row.addWidget(clear_btn)
        n2v_layout.addLayout(model_row)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._n2v_mode = QComboBox()
        self._n2v_mode.addItems(["2D", "3D"])
        self._n2v_mode.setToolTip(
            "2D: each frame is an independent sample (SYX). "
            "3D: stack is treated as a single volume (ZYX)."
        )
        mode_row.addWidget(self._n2v_mode)
        n2v_layout.addLayout(mode_row)

        # Training params
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("Epochs:"))
        self._n2v_epochs = QSpinBox()
        self._n2v_epochs.setRange(1, 1000)
        self._n2v_epochs.setValue(100)
        train_row.addWidget(self._n2v_epochs)
        train_row.addWidget(QLabel("Patch size:"))
        self._n2v_patch = QSpinBox()
        self._n2v_patch.setRange(16, 512)
        self._n2v_patch.setSingleStep(16)
        self._n2v_patch.setValue(64)
        train_row.addWidget(self._n2v_patch)
        n2v_layout.addLayout(train_row)

        # Train frames subset
        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Train frames:"))
        self._n2v_train_frames = QSpinBox()
        self._n2v_train_frames.setRange(0, 10000)
        self._n2v_train_frames.setValue(0)
        self._n2v_train_frames.setSpecialValueText("All")
        self._n2v_train_frames.setToolTip(
            "Number of evenly spaced frames to train on (0 = all). "
            "Use 2-5 for fast training on large stacks."
        )
        frames_row.addWidget(self._n2v_train_frames)
        n2v_layout.addLayout(frames_row)

        n2v_btn_row = QHBoxLayout()
        self._n2v_btn = QPushButton("Denoise (N2V)")
        self._n2v_btn.clicked.connect(self._run_n2v)
        n2v_btn_row.addWidget(self._n2v_btn)
        self._n2v_preview_btn = QPushButton("Quick Preview")
        self._n2v_preview_btn.setToolTip(
            "Apply N2V on the current time point only to test epoch settings"
        )
        self._n2v_preview_btn.clicked.connect(self._run_n2v_preview)
        n2v_btn_row.addWidget(self._n2v_preview_btn)
        self._n2v_cancel_btn = QPushButton("Cancel")
        self._n2v_cancel_btn.clicked.connect(self._cancel_n2v)
        self._n2v_cancel_btn.setEnabled(False)
        n2v_btn_row.addWidget(self._n2v_cancel_btn)
        n2v_layout.addLayout(n2v_btn_row)

        n2v_group.setLayout(n2v_layout)
        layout.addWidget(n2v_group)

        # ── Walking Average ───────────────────────────────────────────
        wa_group = QGroupBox("Walking Average")
        wa_layout = QVBoxLayout()

        row_win = QHBoxLayout()
        row_win.addWidget(QLabel("Window size:"))
        self._window_size = QSpinBox()
        self._window_size.setRange(1, 99)
        self._window_size.setSingleStep(1)
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

    # ── Events ────────────────────────────────────────────────────────

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

    def _on_bg_method_changed(self, method: str):
        is_highpass = "High-pass" in method
        self._highpass_row_label.setVisible(is_highpass)
        self._bg_sigma.setVisible(is_highpass)
        self._erosion_row_label.setVisible(not is_highpass)
        self._disk_size.setVisible(not is_highpass)

    def _browse_n2v_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select N2V model", "", "Model files (*.zip *.ckpt);;All files (*)"
        )
        if path:
            self._n2v_model_path.setText(path)

    def _clear_n2v_model(self):
        self._n2v_model_path.setText("(train from scratch)")

    def _get_image(self):
        """Get the selected image layer data, or None with error shown."""
        layer_name = self._image_combo.currentText()
        if not layer_name:
            show_error("No image selected.")
            return None, None
        layer = self.viewer.layers[layer_name]
        return layer_name, np.asarray(layer.data)

    # ── Background Removal ────────────────────────────────────────────

    def _run_remove_background(self):
        layer_name, image = self._get_image()
        if image is None:
            return

        method = self._bg_method.currentText()
        device = "cuda" if self._use_gpu.isChecked() else "cpu"

        if "High-pass" in method:
            sigma = self._bg_sigma.value()
            show_info(f"Removing background (high-pass, sigma={sigma})...")

            if image.ndim == 2:
                result = remove_background_high_pass(image, sigma=sigma, device=device)
            elif image.ndim == 3:
                result = remove_background_high_pass_stack(image, sigma=sigma, device=device)
            else:
                show_error(f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D.")
                return

            suffix = f"hp s={sigma}"

        else:  # Erosion
            disk_size = self._disk_size.value()
            show_info(f"Removing background (erosion, disk={disk_size})...")

            if image.ndim == 2:
                result = remove_background_erosion(image, disk_size=disk_size)
            elif image.ndim == 3:
                frames = []
                for t in progress(range(image.shape[0]), desc="Removing background"):
                    frames.append(remove_background_erosion(image[t], disk_size=disk_size))
                result = np.stack(frames, axis=0)
            else:
                show_error(f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D.")
                return

            suffix = f"erosion d={disk_size}"

        self.viewer.add_image(result, name=f"{layer_name} ({suffix})")
        show_info("Done — background removed.")
        self._status_label.setText("Background removed.")

    # ── Noise2Void ────────────────────────────────────────────────────

    def _run_n2v_preview(self):
        """Run N2V on the current time point only for quick parameter testing.

        Respects 2D/3D mode:
        - 2D + 3D stack: extracts current frame as 2D (Y,X)
        - 3D + 4D stack: extracts current time point as 3D volume (Z,Y,X)
        - 2D + 2D image: uses as-is
        - 3D + 3D stack: uses as-is (single 3D volume)
        """
        layer_name, image = self._get_image()
        if image is None:
            return

        mode = self._n2v_mode.currentText()
        is_3d = mode == "3D"

        if image.ndim == 2:
            # Single 2D image — always 2D mode
            frame_image = image
            frame_idx = None
        elif image.ndim == 3:
            if is_3d:
                # 3D mode: treat entire stack as a single volume
                frame_image = image
                frame_idx = None
            else:
                # 2D mode: extract current frame
                frame_idx = int(self.viewer.dims.current_step[0])
                frame_image = image[frame_idx]
        elif image.ndim == 4:
            # T,Z,Y,X — extract current time point
            frame_idx = int(self.viewer.dims.current_step[0])
            frame_image = image[frame_idx]  # gives Z,Y,X volume
            if not is_3d:
                # 2D mode on 4D: take middle Z slice
                z_idx = frame_image.shape[0] // 2
                frame_image = frame_image[z_idx]
                frame_idx = (frame_idx, z_idx)
        else:
            show_error(f"Unsupported image dimensions: {image.ndim}D.")
            return

        model_path = self._n2v_model_path.text()
        if model_path == "(train from scratch)":
            model_path = None

        suffix = f" (frame {frame_idx})" if frame_idx is not None else ""
        self._n2v_layer_name = f"{layer_name} N2V preview{suffix}"
        self._n2v_preview_btn.setEnabled(False)
        self._n2v_btn.setEnabled(False)
        self._n2v_cancel_btn.setEnabled(True)
        self._pbr = progress(total=0, desc=f"N2V preview{suffix}")

        self._n2v_worker = N2VWorker(
            image=frame_image,
            model_path=model_path,
            n_epochs=self._n2v_epochs.value(),
            patch_size=self._n2v_patch.value(),
            mode=mode,
        )
        self._n2v_worker.progress.connect(self._on_n2v_progress)
        self._n2v_worker.finished.connect(self._on_n2v_finished)
        self._n2v_worker.errored.connect(self._on_n2v_error)
        self._n2v_worker.start()

    def _run_n2v(self):
        layer_name, image = self._get_image()
        if image is None:
            return

        if image.ndim not in (2, 3):
            show_error(f"Expected 2D or 3D (T,Y,X) image, got {image.ndim}D.")
            return

        model_path = self._n2v_model_path.text()
        if model_path == "(train from scratch)":
            model_path = None

        self._n2v_layer_name = layer_name
        self._n2v_preview_btn.setEnabled(False)
        self._n2v_btn.setEnabled(False)
        self._n2v_cancel_btn.setEnabled(True)
        self._pbr = progress(total=0, desc="N2V denoising")

        train_frames = self._n2v_train_frames.value() or None
        self._n2v_worker = N2VWorker(
            image=image,
            model_path=model_path,
            n_epochs=self._n2v_epochs.value(),
            patch_size=self._n2v_patch.value(),
            mode=self._n2v_mode.currentText(),
            train_frames=train_frames,
        )
        self._n2v_worker.progress.connect(self._on_n2v_progress)
        self._n2v_worker.finished.connect(self._on_n2v_finished)
        self._n2v_worker.errored.connect(self._on_n2v_error)
        self._n2v_worker.start()

    def _on_n2v_progress(self, msg: str):
        self._status_label.setText(msg)
        if self._pbr is not None:
            self._pbr.set_description(msg)

    def _on_n2v_finished(self, result):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._n2v_btn.setEnabled(True)
        self._n2v_preview_btn.setEnabled(True)
        self._n2v_cancel_btn.setEnabled(False)
        name = self._n2v_layer_name or ""
        if "preview" not in name.lower():
            name = f"{name} (N2V denoised)"
        self.viewer.add_image(np.asarray(result), name=name)
        show_info(f"Done — {name}.")
        self._status_label.setText(f"{name}.")

    def _on_n2v_error(self, msg: str):
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._n2v_btn.setEnabled(True)
        self._n2v_preview_btn.setEnabled(True)
        self._n2v_cancel_btn.setEnabled(False)
        show_error(f"N2V failed: {msg[:300]}")
        self._status_label.setText("N2V failed.")

    def _cancel_n2v(self):
        if self._n2v_worker is not None and self._n2v_worker.isRunning():
            self._n2v_worker.progress.disconnect(self._on_n2v_progress)
            self._n2v_worker.finished.disconnect(self._on_n2v_finished)
            self._n2v_worker.errored.disconnect(self._on_n2v_error)
            self._n2v_worker.cancel()
            self._n2v_worker.wait(5000)
        self._n2v_worker = None
        if self._pbr is not None:
            self._pbr.close()
            self._pbr = None
        self._n2v_btn.setEnabled(True)
        self._n2v_preview_btn.setEnabled(True)
        self._n2v_cancel_btn.setEnabled(False)
        show_info("N2V cancelled.")
        self._status_label.setText("N2V cancelled.")

    # ── Walking Average ───────────────────────────────────────────────

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
