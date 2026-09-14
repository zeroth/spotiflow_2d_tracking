"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
import napari.layers
import numpy as np
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
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, QLabeledRangeSlider
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from napari_spotiflow_tracking._mpl_theme import style_figure

_POINTS_FILTER_WHITELIST = frozenset({
    "intensity", "mean_intensity", "max_intensity", "min_intensity", "area",
})
_TRACKS_FILTER_WHITELIST = frozenset({
    "length", "mean_intensity", "max_intensity",
})

_POINTS_HISTOGRAM_WHITELIST = _POINTS_FILTER_WHITELIST  # all 5 properties
# Tracks histograms plot raw values on a shared axis, so intensity columns
# (whose scale dwarfs track length) stay filter-only.
_TRACKS_HISTOGRAM_WHITELIST = frozenset({"length"})


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
    filteredDataChanged = Signal(object)  # emits the filtered DataFrame

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

        hist_group = QGroupBox("Histogram")
        hist_outer = QVBoxLayout()
        hist_group.setLayout(hist_outer)

        self._hist_fig = Figure(figsize=(6, 3), tight_layout=True)
        self._hist_canvas = FigureCanvas(self._hist_fig)
        self._hist_canvas.setFixedHeight(220)
        hist_outer.addWidget(self._hist_canvas)

        layout.addWidget(hist_group)
        self._hist_group = hist_group
        self._hist_checkboxes: dict = {}
        self._hist_colors: dict = {}

        self._table_model = _PropertyTableModel(self)
        self._proxy = _PropertyTableProxy(self)
        self._proxy.setSourceModel(self._table_model)
        # NOTE: the table view is intentionally NOT added to FilterPanel's
        # own layout. Host widgets place it elsewhere (e.g. an Analysis
        # tab) by pulling it via the public `table_view` property; Qt
        # re-parents on addWidget.
        self._table_view = QTableView()
        self._table_view.setModel(self._proxy)
        self._table_view.setSelectionMode(QTableView.SingleSelection)
        self._table_view.setSelectionBehavior(QTableView.SelectRows)
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        sel_model = self._table_view.selectionModel()
        sel_model.currentRowChanged.connect(self._on_table_selection_changed)

        self._show_filtered_cb = QCheckBox("Show filtered in viewer")
        self._show_filtered_cb.setChecked(False)
        self._show_filtered_cb.toggled.connect(self._on_show_filtered_toggled)
        layout.addWidget(self._show_filtered_cb)

        self._sibling_layer = None

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
        self._check_source_removed()
        self._refresh_layer_combo()

    def _check_source_removed(self):
        if self._source_layer is None:
            return
        try:
            still_in_viewer = self._source_layer in self._viewer.layers
        except Exception:
            still_in_viewer = False
        if not still_in_viewer:
            self._remove_sibling_layer()
            self._source_layer = None
            self._source_df = None

    def _refresh_layer_combo(self):
        prev = self._layer_combo.currentText()
        self._layer_combo.clear()
        cls = self._layer_class()
        for layer in self._viewer.layers:
            if not isinstance(layer, cls):
                continue
            # Skip our own sibling — it's not a useful filter source.
            # Check by identity first (fast path) and by name as a fallback
            # for the window between add_tracks() firing events.inserted and
            # _sibling_layer being assigned.
            sibling_name = self._sibling_name()
            if (self._sibling_layer is not None and layer is self._sibling_layer) or (
                sibling_name and layer.name == sibling_name
            ):
                continue
            self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(prev)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._update_status()

    @property
    def table_view(self) -> QTableView:
        """Return the table view widget so host widgets can place it.

        FilterPanel constructs and owns the table view + proxy + model
        but doesn't add the view to its own layout. Host widgets call
        e.g. ``analysis_layout.addWidget(panel.table_view)`` to place it
        in an Analysis tab; Qt re-parents on addWidget.
        """
        return self._table_view

    def refresh_current_layer(self) -> None:
        """Re-read the current source layer's ``metadata['properties']``.

        Call from the host widget after merging new columns into a layer's
        metadata so the panel rebuilds sliders/histogram with the new
        columns. No-op when no source layer is currently loaded.
        """
        if self._source_layer is None:
            return
        name = self._source_layer.name
        # Bypass the same-source early-return guard in _on_layer_changed
        # by clearing the cached layer reference first.
        self._source_layer = None
        self._on_layer_changed(name)

    def _on_layer_changed(self, name: str):
        # Defensive: if a transient currentTextChanged signal selects the same
        # source layer we're already showing, don't tear down the user's
        # sliders. This protects against the clear+rebuild cycle in
        # _refresh_layer_combo when sibling layers churn.
        #
        # Two transient cases to guard:
        # 1. name == "" — emitted by QComboBox.clear() mid-rebuild; if we
        #    already have a valid source, ignore this transient blank signal.
        # 2. name matches the current source — the combo re-selected the same
        #    layer after a clear+addItem cycle; nothing has changed.
        if self._source_layer is not None:
            if not name:
                return
            if (
                self._source_layer.name == name
                and name in self._viewer.layers
                and self._viewer.layers[name] is self._source_layer
            ):
                return

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
        self._redraw_histogram()
        self._update_status()
        # Emit the initial unfiltered DataFrame for any Analysis tab
        # subscriber so charts populate as soon as a layer is selected.
        self.filteredDataChanged.emit(props.copy())

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
        # Histogram checkboxes are now nested inside the slider-row
        # containers; they're freed automatically when those containers
        # are reparented above. Just drop the dict references.
        self._hist_checkboxes = {}
        self._hist_colors = {}

    def _is_filterable_column(self, col: str) -> bool:
        if self._layer_kind == "Points":
            return col in _POINTS_FILTER_WHITELIST
        return col in _TRACKS_FILTER_WHITELIST

    def _is_histogram_column(self, col: str) -> bool:
        if self._layer_kind == "Points":
            return col in _POINTS_HISTOGRAM_WHITELIST
        return col in _TRACKS_HISTOGRAM_WHITELIST

    def _normalize_histogram(self) -> bool:
        return self._layer_kind == "Points"

    # Layout constants for slider rows. The histogram checkbox lives on
    # the left of each row; non-histogram rows get a same-width spacer so
    # the column labels and sliders line up across rows.
    _ROW_CB_WIDTH = 22
    _ROW_LABEL_WIDTH = 130

    def _build_sliders(self, df: pd.DataFrame):
        cmap = colormaps.get_cmap("tab10")
        color_idx = 0
        for col in df.columns:
            if not self._is_filterable_column(col):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_min = df[col].min()
            col_max = df[col].max()

            row = QHBoxLayout()

            # Per-row histogram checkbox (or fixed-width spacer for alignment).
            if self._is_histogram_column(col) and col_min != col_max:
                cb = QCheckBox()
                cb.setChecked(True)
                cb.setToolTip(f"Show {col} in histogram")
                cb.setFixedWidth(self._ROW_CB_WIDTH)
                cb.toggled.connect(self._redraw_histogram)
                row.addWidget(cb)
                self._hist_checkboxes[col] = cb
            else:
                spacer = QWidget()
                spacer.setFixedWidth(self._ROW_CB_WIDTH)
                row.addWidget(spacer)

            label = QLabel(col)
            label.setMinimumWidth(self._ROW_LABEL_WIDTH)
            label.setMaximumWidth(self._ROW_LABEL_WIDTH)
            row.addWidget(label)

            if col_min == col_max:
                # Constant column: no slider, just a value indicator.
                row.addWidget(QLabel(f"constant {col_min}"))
                row.addStretch()
                container = QWidget()
                container.setLayout(row)
                self._sliders_layout.addWidget(container)
                self._constant_labels[col] = container
                continue

            color = cmap(color_idx % 10)
            color_idx += 1
            self._hist_colors[col] = color

            if pd.api.types.is_integer_dtype(df[col]):
                slider = QLabeledRangeSlider(Qt.Horizontal)
                slider.setRange(int(col_min), int(col_max))
                slider.setValue((int(col_min), int(col_max)))
            else:
                slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
                slider.setRange(float(col_min), float(col_max))
                slider.setValue((float(col_min), float(col_max)))
            # QLabeledRangeSlider's default horizontal size policy is
            # Preferred, which makes the layout respect its small intrinsic
            # width and ignore stretch. Force horizontal Expanding so the
            # track fills the available row width.
            slider.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            # Hide the static min/max edge labels — they duplicate the
            # floating handle labels and steal track width when values are
            # wide (e.g. 1339.99). superqt's _AbstractLabeledSlider hides
            # them when passed a falsy EdgeLabelMode.
            slider.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode(0))
            row.addWidget(slider, stretch=3)

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
            self.filteredDataChanged.emit(
                self._source_df.copy() if self._source_df is not None else pd.DataFrame()
            )
            return

        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)

        self._proxy.setRanges(self._filter_ranges)

        n_filtered = len(filtered)
        n_total = len(self._source_df)
        self._status_label.setText(f"{n_filtered} / {n_total} rows match")

        self._redraw_histogram(filtered=filtered)
        if self._show_filtered_cb.isChecked():
            self._update_sibling_layer(filtered)

        self.filteredDataChanged.emit(filtered)

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

    def _redraw_histogram(self, filtered: pd.DataFrame | None = None, *_):
        if self._source_df is None or self._hist_fig is None:
            return
        # Handle case where filtered might be a boolean (from checkbox toggled signal)
        if not isinstance(filtered, pd.DataFrame):
            filtered = None
        if filtered is None:
            mask = pd.Series(True, index=self._source_df.index)
            for col, (lo, hi) in self._filter_ranges.items():
                mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
            filtered = self._source_df[mask].reset_index(drop=True)

        self._hist_fig.clear()
        ax = self._hist_fig.add_subplot(111)
        if self._normalize_histogram():
            ax.set_xlabel("normalized [0, 1]")
        else:
            ax.set_xlabel("value")
        ax.set_ylabel("count")

        any_drawn = False
        normalize = self._normalize_histogram()
        for col, slider in self._sliders.items():
            cb = self._hist_checkboxes.get(col)
            if cb is None or not cb.isChecked():
                continue
            col_min = self._source_df[col].min()
            col_max = self._source_df[col].max()
            if col_max == col_min:
                continue
            values = filtered[col].to_numpy() if col in filtered.columns else np.array([])
            if values.size == 0:
                continue
            color = self._hist_colors.get(col, "tab:blue")
            if normalize:
                normed = (values - col_min) / (col_max - col_min)
                ax.hist(
                    normed, bins=30, range=(0, 1),
                    histtype="stepfilled", alpha=0.5, color=color,
                    label=f"{col}  [{col_min:.3g} - {col_max:.3g}]",
                )
            else:
                ax.hist(
                    values, bins=30, range=(float(col_min), float(col_max)),
                    histtype="stepfilled", alpha=0.5, color=color,
                    label=f"{col}",
                )
            any_drawn = True

        if any_drawn:
            ax.legend(loc="upper right", fontsize=8)
        style_figure(self._hist_fig)
        self._hist_canvas.draw_idle()

    def _on_show_filtered_toggled(self, checked: bool):
        if not checked:
            self._remove_sibling_layer()
            return
        if self._source_df is None:
            return
        mask = pd.Series(True, index=self._source_df.index)
        for col, (lo, hi) in self._filter_ranges.items():
            mask &= (self._source_df[col] >= lo) & (self._source_df[col] <= hi)
        filtered = self._source_df[mask].reset_index(drop=True)
        self._update_sibling_layer(filtered)

    def _sibling_name(self) -> str:
        if self._source_layer is None:
            return ""
        return f"{self._source_layer.name} (filtered)"

    def _remove_sibling_layer(self):
        name = self._sibling_name()
        if not name:
            return
        if name in self._viewer.layers:
            try:
                self._viewer.layers.remove(name)
            except (KeyError, ValueError):
                pass
        self._sibling_layer = None

    def _update_sibling_layer(self, filtered: pd.DataFrame):
        if self._source_layer is None:
            return
        name = self._sibling_name()
        if self._layer_kind == "Points":
            if "frame" in filtered.columns:
                data = filtered[["frame", "y", "x"]].to_numpy()
            else:
                data = filtered[["y", "x"]].to_numpy()
            if name in self._viewer.layers:
                self._viewer.layers[name].data = data
                self._sibling_layer = self._viewer.layers[name]
            else:
                self._sibling_layer = self._viewer.add_points(
                    data, name=name, size=3, face_color="cyan",
                )
        elif self._layer_kind == "Tracks":
            self._update_tracks_sibling(filtered, name)

    def _update_tracks_sibling(self, filtered: pd.DataFrame, name: str):
        tracked_df = self._source_layer.metadata.get("tracked_df")
        if not isinstance(tracked_df, pd.DataFrame):
            return
        kept_ids = filtered[self._id_column].astype(int).tolist()
        if not kept_ids:
            data = np.empty((0, 4))
        else:
            sub = tracked_df[tracked_df["particle"].isin(kept_ids)]
            sub = sub.sort_values(["particle", "frame"])
            data = sub[["particle", "frame", "y", "x"]].to_numpy()

        if name in self._viewer.layers:
            try:
                self._viewer.layers.remove(name)
            except (KeyError, ValueError):
                pass

        if len(data) == 0:
            self._sibling_layer = None
            return
        self._sibling_layer = self._viewer.add_tracks(data, name=name)
