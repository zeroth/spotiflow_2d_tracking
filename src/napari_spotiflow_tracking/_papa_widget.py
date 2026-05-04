from __future__ import annotations

import napari
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PAPAWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_per_frame_df = None
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout()
        self.setLayout(outer)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        self._input_tab = self._build_input_tab()
        self._plots_tab = self._build_plots_tab()
        self._tabs.addTab(self._input_tab, "Input")
        self._tabs.addTab(self._plots_tab, "Plots")

    def _build_input_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Points layer:"))
        self._points_combo = QComboBox()
        row.addWidget(self._points_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setToolTip("Refresh layer list")
        row.addWidget(refresh_btn)
        input_layout.addLayout(row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Cycle Pattern group
        cycle_group = QGroupBox("Cycle Pattern")
        cycle_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Group 1 size:"))
        self._group1_size = QLineEdit("5")
        self._group1_size.setValidator(QIntValidator(1, 100000, self))
        row1.addWidget(self._group1_size)
        cycle_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Group 2 size:"))
        self._group2_size = QLineEdit("5")
        self._group2_size.setValidator(QIntValidator(1, 100000, self))
        row2.addWidget(self._group2_size)
        cycle_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Skip frames:"))
        self._skip_frames = QLineEdit("1")
        self._skip_frames.setValidator(QIntValidator(0, 100, self))
        row3.addWidget(self._skip_frames)
        cycle_layout.addLayout(row3)

        cycle_group.setLayout(cycle_layout)
        layout.addWidget(cycle_group)

        # Labels group
        labels_group = QGroupBox("Labels")
        labels_layout = QVBoxLayout()
        row_l1 = QHBoxLayout()
        row_l1.addWidget(QLabel("Group 1 label:"))
        self._label_group1 = QLineEdit("DR")
        row_l1.addWidget(self._label_group1)
        labels_layout.addLayout(row_l1)
        row_l2 = QHBoxLayout()
        row_l2.addWidget(QLabel("Group 2 label:"))
        self._label_group2 = QLineEdit("PAPA")
        row_l2.addWidget(self._label_group2)
        labels_layout.addLayout(row_l2)
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)

        # Buttons + status
        self._plot_btn = QPushButton("Plot")
        layout.addWidget(self._plot_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

        self._status_label = QLabel("Ready. Set parameters and click Plot.")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return page

    def _build_plots_tab(self) -> QWidget:
        # Plots tab built in Task 7. For now, return an empty scroll area
        # so the tab exists.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setLayout(QVBoxLayout())
        scroll.setWidget(inner)
        self._plots_inner = inner
        return scroll
