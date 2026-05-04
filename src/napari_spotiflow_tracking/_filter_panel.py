"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
import napari.layers
import pandas as pd
from qtpy.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    QVariant,
    Signal,
)
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, QLabeledRangeSlider


class _PropertyTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame = pd.DataFrame()

    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True).copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        if role == Qt.UserRole + 1:
            return float(self._df.iat[index.row(), index.column()])
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            if 0 <= section < self._df.shape[1]:
                return str(self._df.columns[section])
        return str(section)


class _PropertyTableProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ranges: dict = {}

    def setRanges(self, ranges: dict):
        self._ranges = dict(ranges)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None or not self._ranges:
            return True
        for col, (lo, hi) in self._ranges.items():
            col_idx = list(model._df.columns).index(col)
            val = model._df.iat[source_row, col_idx]
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v < lo or v > hi:
                return False
        return True


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

        self._table_model = _PropertyTableModel(self)
        self._proxy = _PropertyTableProxy(self)
        self._proxy.setSourceModel(self._table_model)
        self._table_view = QTableView()
        self._table_view.setModel(self._proxy)
        self._table_view.setSelectionMode(QTableView.SingleSelection)
        self._table_view.setSelectionBehavior(QTableView.SelectRows)
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        layout.addWidget(self._table_view)

        sel_model = self._table_view.selectionModel()
        sel_model.currentRowChanged.connect(self._on_table_selection_changed)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset Filters")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        btn_row.addWidget(self._reset_btn)
        self._export_btn = QPushButton("Export Filtered CSV")
        self._export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh)

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
        self._table_model.setDataFrame(props)
        self._proxy.setRanges({})
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
            slider.valueChanged.connect(
                lambda v, c=col: self._on_slider_changed(c, v)
            )

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

    def _on_slider_changed(self, col: str, value):
        lo, hi = value
        self._filter_ranges[col] = (lo, hi)
        self._refresh_timer.start()

    def _refresh(self):
        if self._source_df is None or len(self._sliders) == 0:
            self._update_status()
            return

        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)

        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

    def _on_table_selection_changed(self, current, previous):
        if not current.isValid():
            return
        source_index = self._proxy.mapToSource(current)
        if not source_index.isValid():
            return
        col_idx = list(self._table_model._df.columns).index(self._id_column)
        row_id = int(self._table_model._df.iat[source_index.row(), col_idx])
        self.selectionChanged.emit(row_id)

    def _on_reset_clicked(self):
        if self._source_df is None:
            return
        for col, slider in self._sliders.items():
            col_min = self._source_df[col].min()
            col_max = self._source_df[col].max()
            if pd.api.types.is_integer_dtype(self._source_df[col]):
                slider.setValue((int(col_min), int(col_max)))
                self._filter_ranges[col] = (int(col_min), int(col_max))
            else:
                slider.setValue((float(col_min), float(col_max)))
                self._filter_ranges[col] = (float(col_min), float(col_max))
        self._refresh()

    def _on_export_clicked(self):
        if self._source_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save filtered CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)
        filtered.to_csv(path, index=False)
