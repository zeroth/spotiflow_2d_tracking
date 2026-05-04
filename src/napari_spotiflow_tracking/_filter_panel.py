"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
import napari.layers
import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, QLabeledRangeSlider


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

        sliders_group = QGroupBox("Range filters")
        self._sliders_layout = QVBoxLayout()
        sliders_group.setLayout(self._sliders_layout)
        layout.addWidget(sliders_group)
        self._sliders_group = sliders_group

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._sliders: dict = {}
        self._constant_labels: dict = {}
        self._source_df: pd.DataFrame | None = None
        self._source_layer = None
        self._filter_ranges: dict = {}

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
        self._clear_sliders()
        self._source_df = None
        self._source_layer = None
        self._filter_ranges = {}

        if not name or name not in self._viewer.layers:
            self._update_status()
            return

        layer = self._viewer.layers[name]
        if not isinstance(layer, self._layer_class()):
            self._update_status()
            return

        props = layer.metadata.get("properties")
        if props is None:
            self._status_label.setText(
                "Selected layer has no properties metadata. "
                "Run Detection / Tracking to populate it."
            )
            self._sliders_group.setVisible(False)
            return
        if not isinstance(props, pd.DataFrame):
            self._status_label.setText(
                "Properties metadata is not a DataFrame; ignored."
            )
            self._sliders_group.setVisible(False)
            return

        self._source_layer = layer
        self._source_df = props.copy()
        self._sliders_group.setVisible(True)
        self._build_sliders(props)
        self._update_status()

    def _clear_sliders(self):
        for slider in self._sliders.values():
            slider.setParent(None)
        for label in self._constant_labels.values():
            label.setParent(None)
        self._sliders = {}
        self._constant_labels = {}
        while self._sliders_layout.count():
            item = self._sliders_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _is_filterable_column(self, col: str) -> bool:
        if col == self._id_column:
            return False
        if self._layer_kind == "Points" and col in {"frame", "y", "x"}:
            return False
        if self._layer_kind == "Tracks" and col in {"t", "y", "x", "track_id"}:
            return False
        return True

    def _build_sliders(self, df: pd.DataFrame):
        for col in df.columns:
            if not self._is_filterable_column(col):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{col}: constant {col_min}"))
                container = QWidget()
                container.setLayout(row)
                self._sliders_layout.addWidget(container)
                self._constant_labels[col] = container
                continue

            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            if pd.api.types.is_integer_dtype(df[col]):
                slider = QLabeledRangeSlider(Qt.Horizontal)
                slider.setRange(int(col_min), int(col_max))
                slider.setValue((int(col_min), int(col_max)))
            else:
                slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
                slider.setRange(float(col_min), float(col_max))
                slider.setValue((float(col_min), float(col_max)))
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._sliders_layout.addWidget(container)
            self._sliders[col] = slider
            self._filter_ranges[col] = (col_min, col_max)

    def _update_status(self):
        if self._layer_combo.count() == 0:
            self._status_label.setText(f"No {self._layer_kind} layers in the viewer.")
        elif self._source_df is None:
            # Layer chosen but no valid metadata; status was already set by
            # _on_layer_changed.
            pass
        elif len(self._sliders) == 0 and len(self._constant_labels) == 0:
            self._status_label.setText("No filterable properties on this layer.")
        elif len(self._sliders) == 0:
            self._status_label.setText("All filterable columns are constant.")
        else:
            n_total = len(self._source_df)
            self._status_label.setText(f"{n_total} / {n_total} rows match")
