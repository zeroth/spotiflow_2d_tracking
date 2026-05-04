"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
import napari.layers
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FilterPanel(QWidget):
    """Reusable filter panel embedded in Detection and Tracking widgets."""

    selectionChanged = Signal(int)

    def __init__(
        self,
        viewer: napari.Viewer,
        layer_kind: Literal["Points", "Tracks"],
        id_column: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._viewer = viewer
        self._layer_kind = layer_kind
        self._id_column = id_column

        self._setup_ui()
        self._connect_events()
        self._refresh_layer_combo()
        self._update_status()

    def _layer_class(self):
        if self._layer_kind == "Points":
            return napari.layers.Points
        return napari.layers.Tracks

    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        group = QGroupBox("Filter")
        layout = QVBoxLayout()
        group.setLayout(layout)
        outer.addWidget(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        row.addWidget(self._layer_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layer_combo)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

    def _connect_events(self):
        self._viewer.layers.events.inserted.connect(self._on_layer_event)
        self._viewer.layers.events.removed.connect(self._on_layer_event)

    def _on_layer_event(self, event=None):
        self._refresh_layer_combo()

    def _refresh_layer_combo(self):
        prev = self._layer_combo.currentText()
        self._layer_combo.clear()
        cls = self._layer_class()
        for layer in self._viewer.layers:
            if isinstance(layer, cls):
                self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(prev)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._update_status()

    def _on_layer_changed(self, name: str):
        self._update_status()

    def _update_status(self):
        if self._layer_combo.count() == 0:
            self._status_label.setText(f"No {self._layer_kind} layers in the viewer.")
        else:
            self._status_label.setText("")
