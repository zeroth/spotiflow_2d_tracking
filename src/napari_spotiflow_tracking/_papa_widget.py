from __future__ import annotations

import numpy as np
import napari
from napari.utils.notifications import show_error, show_info
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy.QtGui import QIntValidator

from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
)
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
        self._connect_events()
        self._refresh_points_combo()

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
        refresh_btn.clicked.connect(self._refresh_points_combo)
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
        self._plot_btn.clicked.connect(self._on_plot_clicked)
        layout.addWidget(self._plot_btn)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

        self._status_label = QLabel("Ready. Set parameters and click Plot.")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return page

    def _build_plots_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner.setLayout(inner_layout)

        self._fig_combined = Figure(figsize=(10, 4), tight_layout=True)
        self._canvas_combined = FigureCanvas(self._fig_combined)
        self._toolbar_combined = NavigationToolbar(self._canvas_combined, self)
        inner_layout.addWidget(self._toolbar_combined)
        inner_layout.addWidget(self._canvas_combined)

        self._fig_ratio = Figure(figsize=(10, 9), tight_layout=True)
        self._canvas_ratio = FigureCanvas(self._fig_ratio)
        self._toolbar_ratio = NavigationToolbar(self._canvas_ratio, self)
        inner_layout.addWidget(self._toolbar_ratio)
        inner_layout.addWidget(self._canvas_ratio)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        self._refresh_points_combo()

    def _refresh_points_combo(self):
        prev = self._points_combo.currentText()
        self._points_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self._points_combo.addItem(layer.name)
        idx = self._points_combo.findText(prev)
        if idx >= 0:
            self._points_combo.setCurrentIndex(idx)

    def _set_status_error(self, msg: str):
        self._status_label.setText(msg)
        show_error(msg)

    def _parse_int_field(self, field, name: str, min_value: int) -> int | None:
        text = field.text().strip()
        if not text:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        try:
            value = int(text)
        except ValueError:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        if value < min_value:
            self._set_status_error(
                f"{name} must be a {'positive' if min_value > 0 else 'non-negative'} integer."
            )
            return None
        return value

    def _validate_inputs(self):
        """Returns a tuple (points, g1, g2, skip, label1, label2) on success,
        or None if any guard failed (status already updated)."""
        layer_name = self._points_combo.currentText()
        if not layer_name:
            self._set_status_error("No Points layer selected.")
            return None

        if layer_name not in self.viewer.layers:
            self._set_status_error("Selected layer no longer exists.")
            return None

        layer = self.viewer.layers[layer_name]
        if not isinstance(layer, napari.layers.Points):
            self._set_status_error("Selected layer is not a Points layer.")
            return None

        data = np.asarray(layer.data)
        if data.ndim != 2 or data.shape[1] != 3:
            self._set_status_error(
                "Points layer must have 3 columns (frame, y, x)."
            )
            return None

        if len(data) == 0:
            self._set_status_error("Points layer is empty; nothing to plot.")
            return None

        g1 = self._parse_int_field(self._group1_size, "Group 1 size", 1)
        if g1 is None:
            return None
        g2 = self._parse_int_field(self._group2_size, "Group 2 size", 1)
        if g2 is None:
            return None
        skip = self._parse_int_field(self._skip_frames, "Skip frames", 0)
        if skip is None:
            return None

        label1 = self._label_group1.text() or "G1"
        label2 = self._label_group2.text() or "G2"
        return data, g1, g2, skip, label1, label2

    def _on_plot_clicked(self):
        validated = self._validate_inputs()
        if validated is None:
            return
        data, g1, g2, skip, label1, label2 = validated

        per_frame, n_neg = compute_particles_per_frame(data)
        if len(per_frame) == 0:
            self._set_status_error("Need at least 1 frame of data.")
            return

        max_time = int(per_frame["frame"].max())
        cycle_info = build_cycle_info(max_time, g1, g2, skip)
        per_frame = annotate_with_cycle_group(per_frame, cycle_info)

        self._render_combined_plot(
            self._fig_combined, per_frame, cycle_info,
            max_time, g1, g2, skip, label1, label2,
        )
        self._canvas_combined.draw_idle()

        self._render_ratio_plot(
            self._fig_ratio, per_frame, cycle_info, label1, label2,
        )
        self._canvas_ratio.draw_idle()

        self._last_per_frame_df = per_frame
        self._export_btn.setEnabled(True)

        n_cycles = max((c["cycle"] for c in cycle_info), default=0)
        msg = f"Plotted {n_cycles} cycles across {max_time} frames."
        if g1 + skip + g2 + skip > max_time and n_cycles > 0:
            msg += " (cycle pattern exceeds frame range; only partial cycles drawn.)"
        if n_neg > 0:
            msg += f" Negative frame indices found; clipped to 0 ({n_neg} rows)."
        self._status_label.setText(msg)
        show_info(msg)

        self._tabs.setCurrentWidget(self._plots_tab)

    NAPARI_BG = "#262930"
    NAPARI_FG = "#d6d6d6"
    LINE_COLOR = "#e0e0e0"

    @classmethod
    def _style_figure(cls, fig):
        fig.patch.set_facecolor(cls.NAPARI_BG)
        for ax in fig.axes:
            ax.set_facecolor(cls.NAPARI_BG)
            for spine in ax.spines.values():
                spine.set_color(cls.NAPARI_FG)
            ax.tick_params(colors=cls.NAPARI_FG)
            ax.xaxis.label.set_color(cls.NAPARI_FG)
            ax.yaxis.label.set_color(cls.NAPARI_FG)
            ax.title.set_color(cls.NAPARI_FG)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(cls.NAPARI_FG)

    def _render_combined_plot(
        self, fig, per_frame, cycle_info, max_time,
        g1, g2, skip, label1, label2,
    ):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(
            per_frame["frame"], per_frame["n_particles"],
            color=self.LINE_COLOR, linewidth=1.0,
        )

        sep_colors = ["tab:blue", "tab:orange"]
        sep1_labeled = False
        sep2_labeled = False
        for info in cycle_info:
            if info["end_frame"] < max_time:
                sep_color = sep_colors[(info["group"] - 1) % 2]
                if info["group"] == 1 and not sep1_labeled:
                    sep_label = label1
                    sep1_labeled = True
                elif info["group"] == 2 and not sep2_labeled:
                    sep_label = label2
                    sep2_labeled = True
                else:
                    sep_label = None
                ax.axvline(
                    x=info["end_frame"] + 0.5, color=sep_color,
                    linestyle="-", alpha=0.6, linewidth=1.5,
                    label=sep_label,
                )

        ax.set_title(
            f"All Cycles Combined — Particles per Frame  "
            f"(G1={g1}f + skip{skip}f + G2={g2}f + skip{skip}f)"
        )
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Number of Particles")
        if sep1_labeled or sep2_labeled:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self._style_figure(fig)

    def _render_ratio_plot(self, fig, per_frame, cycle_info, label1, label2):
        # Stub - filled in Task 10.
        fig.clear()
        self._style_figure(fig)
