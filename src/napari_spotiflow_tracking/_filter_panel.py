"""Reusable property-filter panel for Points and Tracks layers."""

from __future__ import annotations

from typing import Literal

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget


class FilterPanel(QWidget):
    """Reusable filter panel embedded in Detection and Tracking widgets."""

    selectionChanged = Signal(int)  # row's id_column value when user clicks

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
