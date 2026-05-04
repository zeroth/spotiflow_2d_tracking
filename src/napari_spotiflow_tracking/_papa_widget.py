from __future__ import annotations

import napari
from qtpy.QtWidgets import QWidget


class PAPAWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
