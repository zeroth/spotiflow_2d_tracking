"""Tests for PAPAWidget."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def papa_widget(make_napari_viewer):
    from napari_spotiflow_tracking._papa_widget import PAPAWidget

    viewer = make_napari_viewer()
    widget = PAPAWidget(viewer)
    return widget, viewer


class TestPAPAWidgetConstruction:
    def test_creates_without_error(self, papa_widget):
        widget, _ = papa_widget
        assert widget is not None

    def test_has_input_and_plots_tabs(self, papa_widget):
        widget, _ = papa_widget
        assert widget._tabs.count() == 2
        assert widget._tabs.tabText(0) == "Input"
        assert widget._tabs.tabText(1) == "Plots"

    def test_default_field_values(self, papa_widget):
        widget, _ = papa_widget
        assert widget._group1_size.text() == "5"
        assert widget._group2_size.text() == "5"
        assert widget._skip_frames.text() == "1"
        assert widget._label_group1.text() == "DR"
        assert widget._label_group2.text() == "PAPA"

    def test_export_csv_starts_disabled(self, papa_widget):
        widget, _ = papa_widget
        assert widget._export_btn.isEnabled() is False

    def test_initial_status_message(self, papa_widget):
        widget, _ = papa_widget
        assert "Ready" in widget._status_label.text()


import napari
import napari.layers


class TestPAPAWidgetLayerCombo:
    def test_combo_starts_empty(self, papa_widget):
        widget, _ = papa_widget
        assert widget._points_combo.count() == 0

    def test_combo_lists_points_layers(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20], [1, 11, 21]]), name="spots")
        assert widget._points_combo.count() == 1
        assert widget._points_combo.itemText(0) == "spots"

    def test_combo_ignores_image_layers(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_image(np.random.rand(64, 64), name="img")
        assert widget._points_combo.count() == 0

    def test_combo_updates_on_layer_remove(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20]]), name="spots")
        assert widget._points_combo.count() == 1
        viewer.layers.remove("spots")
        assert widget._points_combo.count() == 0

    def test_manual_refresh_button_works(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10, 20]]), name="spots")
        widget._points_combo.clear()
        # Walk children to find the refresh button: it's the only push
        # button with text "↻".
        from qtpy.QtWidgets import QPushButton
        refresh_btns = [
            b for b in widget.findChildren(QPushButton) if b.text() == "↻"
        ]
        assert len(refresh_btns) == 1
        refresh_btns[0].click()
        assert widget._points_combo.count() == 1


class TestPAPAWidgetCanvases:
    def test_combined_figure_and_canvas_exist(self, papa_widget):
        widget, _ = papa_widget
        assert widget._fig_combined is not None
        assert widget._canvas_combined is not None
        assert widget._toolbar_combined is not None

    def test_ratio_figure_and_canvas_exist(self, papa_widget):
        widget, _ = papa_widget
        assert widget._fig_ratio is not None
        assert widget._canvas_ratio is not None
        assert widget._toolbar_ratio is not None

    def test_figures_start_blank(self, papa_widget):
        widget, _ = papa_widget
        assert len(widget._fig_combined.axes) == 0
        assert len(widget._fig_ratio.axes) == 0


class TestPAPAWidgetValidation:
    def test_no_layer_selected(self, papa_widget):
        widget, _ = papa_widget
        widget._plot_btn.click()
        assert "no points layer" in widget._status_label.text().lower()
        assert widget._export_btn.isEnabled() is False

    def test_wrong_shape_layer(self, papa_widget):
        widget, viewer = papa_widget
        # Create a Points layer with 2 columns instead of 3.
        viewer.add_points(np.array([[10.0, 20.0]]), name="bad")
        widget._plot_btn.click()
        assert "3 columns" in widget._status_label.text()

    def test_empty_layer(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.empty((0, 3)), name="empty", ndim=3)
        widget._plot_btn.click()
        assert "empty" in widget._status_label.text().lower()

    def test_empty_group_size_field(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10.0, 20.0]]), name="spots")
        widget._group1_size.setText("")
        widget._plot_btn.click()
        assert "group 1 size" in widget._status_label.text().lower()

    def test_zero_group_size_rejected(self, papa_widget):
        widget, viewer = papa_widget
        viewer.add_points(np.array([[0, 10.0, 20.0]]), name="spots")
        # Bypass the validator (which prevents typing 0) by setText.
        widget._group2_size.setText("0")
        widget._plot_btn.click()
        assert "group 2 size" in widget._status_label.text().lower()


class TestPAPAWidgetPlotPipeline:
    # The default 5/5/1 pattern needs 5 + 1 + 5 + 1 + 5 = 17 frames for one
    # complete cycle, because the first group-1 block is an excluded lead-in.
    def _make_synthetic_layer(self, viewer, n_frames=17, spots_per_frame=3):
        """Add a Points layer with `spots_per_frame` spots on each of `n_frames` frames."""
        rows = []
        rng = np.random.default_rng(0)
        for f in range(n_frames):
            for _ in range(spots_per_frame):
                rows.append([f, rng.uniform(0, 100), rng.uniform(0, 100)])
        viewer.add_points(np.array(rows), name="spots")

    def test_plot_renders_combined_axes(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._group1_size.setText("5")
        widget._group2_size.setText("5")
        widget._skip_frames.setText("1")
        widget._plot_btn.click()
        assert len(widget._fig_combined.axes) == 1

    def test_plot_caches_per_frame_df(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17, spots_per_frame=3)
        widget._plot_btn.click()
        df = widget._last_per_frame_df
        assert df is not None
        assert list(df.columns) == ["frame", "n_particles", "cycle", "group"]
        assert df["n_particles"].tolist() == [3] * 17

    def test_lead_in_frames_have_no_cycle(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        df = widget._last_per_frame_df
        # Frames 1-5 are the excluded lead-in; 7-11 are cycle 1 group 2 and
        # 13-17 are cycle 1 group 1.
        lead_in = df[df["frame"].between(1, 5)]
        assert lead_in["cycle"].isna().all()
        assert lead_in["group"].isna().all()
        assert df.loc[df["frame"] == 9, "group"].tolist() == [2]
        assert df.loc[df["frame"] == 15, "group"].tolist() == [1]

    def test_plot_status_message(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        assert "plotted" in widget._status_label.text().lower()
        assert "1 cycles" in widget._status_label.text() or \
               "1 cycle" in widget._status_label.text()

    def test_zero_complete_cycles_reports_requirement(self, papa_widget):
        widget, viewer = papa_widget
        # 12 frames is one block short of a complete cycle.
        self._make_synthetic_layer(viewer, n_frames=12)
        widget._plot_btn.click()
        text = widget._status_label.text()
        assert "no complete cycles" in text.lower()
        assert "17" in text
        # Both figures must still render rather than raising.
        assert len(widget._fig_combined.axes) == 1
        assert len(widget._fig_ratio.axes) == 1

    def test_export_button_enabled_after_plot(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        assert widget._export_btn.isEnabled() is True

    def test_active_tab_switches_to_plots(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        assert widget._tabs.currentIndex() == 0
        widget._plot_btn.click()
        assert widget._tabs.currentIndex() == 1


class TestPAPAWidgetRatioPlot:
    def _make_synthetic_layer(self, viewer, n_frames=24):
        rows = []
        for f in range(n_frames):
            for _ in range(3):
                rows.append([f, 10.0, 10.0])
        viewer.add_points(np.array(rows), name="spots")

    def test_ratio_figure_has_three_axes(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        assert len(widget._fig_ratio.axes) == 3

    def test_ratio_figure_titles_contain_labels(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._label_group1.setText("DR")
        widget._label_group2.setText("PAPA")
        widget._plot_btn.click()
        titles = [ax.get_title() for ax in widget._fig_ratio.axes]
        assert "DR / PAPA" in titles[1]
        assert "PAPA / DR" in titles[2]


class TestPAPAWidgetTheming:
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(17) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")

    def test_combined_figure_uses_dark_facecolor(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        from matplotlib.colors import to_rgba
        expected = to_rgba(widget.NAPARI_BG)
        assert widget._fig_combined.get_facecolor() == expected
        for ax in widget._fig_combined.axes:
            assert ax.get_facecolor() == expected

    def test_ratio_figure_uses_dark_facecolor(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()
        from matplotlib.colors import to_rgba
        expected = to_rgba(widget.NAPARI_BG)
        assert widget._fig_ratio.get_facecolor() == expected
        for ax in widget._fig_ratio.axes:
            assert ax.get_facecolor() == expected

    def test_zero_cycle_placeholder_uses_theme_foreground(self, papa_widget):
        # 12 frames is one block short of a complete cycle under the default
        # 5/5/1 pattern, so the ratio plot falls back to the placeholder
        # text. It must not stay matplotlib's default black on the dark
        # napari background.
        widget, viewer = papa_widget
        rows = [[f, 10.0, 10.0] for f in range(12) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")
        widget._plot_btn.click()

        from matplotlib.colors import to_rgba
        ax = widget._fig_ratio.axes[0]
        assert len(ax.texts) == 1
        assert to_rgba(ax.texts[0].get_color()) == to_rgba(widget.NAPARI_FG)


class TestPAPAWidgetExportCSV:
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(17) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")

    def test_export_writes_csv(self, papa_widget, tmp_path, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            lambda *a, **kw: str(tmp_path),
        )
        widget._export_btn.click()

        # Both per-frame and per-cycle CSVs should be written (both checkboxes default to checked).
        per_frame_path = tmp_path / "papa_per_frame.csv"
        per_cycle_path = tmp_path / "papa_per_cycle.csv"
        assert per_frame_path.exists()
        assert per_cycle_path.exists()

        # Verify per-frame CSV structure.
        df_frame = pd.read_csv(per_frame_path)
        assert list(df_frame.columns) == ["frame", "n_particles", "cycle", "group"]
        # Frames are 1-indexed.
        assert df_frame["frame"].min() == 1
        # Skip-gap frame (6) should be present with NaN cycle/group and
        # n_particles == 3 (the layer has 3 spots on every frame).
        skip_row = df_frame[df_frame["frame"] == 6].iloc[0]
        assert pd.isna(skip_row["cycle"])
        assert pd.isna(skip_row["group"])
        assert int(skip_row["n_particles"]) == 3

    def test_export_cancel_does_nothing(self, papa_widget, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            lambda *a, **kw: "",
        )
        # Should not raise.
        widget._export_btn.click()
