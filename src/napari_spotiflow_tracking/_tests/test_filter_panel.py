"""Tests for FilterPanel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def filter_panel_points(make_napari_viewer):
    from napari_spotiflow_tracking._filter_panel import FilterPanel
    viewer = make_napari_viewer()
    panel = FilterPanel(viewer, "Points", "spot_id")
    return panel, viewer


@pytest.fixture
def filter_panel_tracks(make_napari_viewer):
    from napari_spotiflow_tracking._filter_panel import FilterPanel
    viewer = make_napari_viewer()
    panel = FilterPanel(viewer, "Tracks", "track_id")
    return panel, viewer


class TestFilterPanelConstruction:
    def test_constructs_without_error_no_layers(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert panel is not None

    def test_status_when_no_layers(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert "no points layer" in panel._status_label.text().lower()

    def test_layer_combo_starts_empty(self, filter_panel_points):
        panel, _ = filter_panel_points
        assert panel._layer_combo.count() == 0

    def test_layer_combo_lists_points(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 1
        assert panel._layer_combo.itemText(0) == "spots"

    def test_layer_combo_ignores_tracks(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_tracks(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]), name="t")
        assert panel._layer_combo.count() == 0

    def test_tracks_panel_lists_tracks(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        viewer.add_tracks(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]), name="t")
        assert panel._layer_combo.count() == 1

    def test_tracks_panel_ignores_points(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 0

    def test_combo_updates_on_layer_remove(self, filter_panel_points):
        panel, viewer = filter_panel_points
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert panel._layer_combo.count() == 1
        viewer.layers.remove("spots")
        assert panel._layer_combo.count() == 0


def _add_points_with_props(viewer, name, df):
    layer = viewer.add_points(
        df[["frame", "y", "x"]].to_numpy(),
        name=name,
        metadata={"properties": df}
    )
    return layer


class TestSliderGeneration:
    def test_no_metadata_disables_sliders(self, filter_panel_points):
        panel, viewer = filter_panel_points
        # Layer with no metadata
        viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        assert "no properties metadata" in panel._status_label.text().lower()
        assert len(panel._sliders) == 0

    def test_metadata_not_dataframe(self, filter_panel_points):
        panel, viewer = filter_panel_points
        layer = viewer.add_points(np.array([[0, 1, 2]]), name="spots")
        layer.metadata["properties"] = {"not": "a df"}
        # Force re-read by reselecting the layer
        panel._on_layer_changed("spots")
        assert "not a dataframe" in panel._status_label.text().lower()

    def test_creates_slider_per_filterable_column(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "intensity": [10.0, 20.0, 30.0],
            "area": [100, 200, 300],
        })
        _add_points_with_props(viewer, "spots", df)
        # spot_id, frame, y, x are excluded -> intensity + area = 2 sliders
        assert set(panel._sliders.keys()) == {"intensity", "area"}

    def test_excludes_id_and_spatial_columns(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        df = pd.DataFrame({
            "track_id": [0, 1],
            "length": [10, 20],
        })
        # Use a real Tracks layer
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        assert set(panel._sliders.keys()) == {"length"}

    def test_constant_column_gets_static_label_not_slider(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "area": [1, 1, 1],  # constant, whitelisted
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert "area" not in panel._sliders
        assert "intensity" in panel._sliders
        assert "area" in panel._constant_labels

    def test_slider_range_matches_column_min_max(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        slider = panel._sliders["intensity"]
        lo, hi = slider.value()
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(30.0)

    def test_integer_column_uses_int_slider(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        from superqt import QLabeledRangeSlider
        df = pd.DataFrame({"track_id": [0, 1], "length": [5, 25]})
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        assert isinstance(panel._sliders["length"], QLabeledRangeSlider)

    def test_float_column_uses_float_slider(self, filter_panel_points):
        panel, viewer = filter_panel_points
        from superqt import QLabeledDoubleRangeSlider
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [1.0, 2.0],
            "x": [3.0, 4.0],
            "intensity": [1.5, 2.5],
        })
        _add_points_with_props(viewer, "spots", df)
        assert isinstance(panel._sliders["intensity"], QLabeledDoubleRangeSlider)


class TestFilterPipeline:
    def test_initial_table_shows_all_rows(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert panel._proxy.rowCount() == 5

    def test_slider_range_filters_rows(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Set intensity range to [25, 45] -> rows with 30 and 40 match -> 2 rows
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()  # bypass debounce
        assert panel._proxy.rowCount() == 2

    def test_filter_status_label(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(10)),
            "frame": list(range(10)),
            "y": [0.0] * 10,
            "x": [0.0] * 10,
            "intensity": list(np.linspace(0, 100, 10)),
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((50.0, 100.0))
        panel._refresh()
        # 5 of 10 rows have intensity >= 50
        assert "5 / 10" in panel._status_label.text()

    def test_table_columns_match_source_df(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        model = panel._table_model
        assert model.columnCount() == len(df.columns)

    def test_selection_changed_signal_emits_id(self, filter_panel_points, qtbot):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [10, 20, 30],
            "frame": [0, 1, 2],
            "y": [0.0] * 3,
            "x": [0.0] * 3,
            "intensity": [1.0, 2.0, 3.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Select the second row in the proxy
        with qtbot.waitSignal(panel.selectionChanged, timeout=500) as blocker:
            proxy_index = panel._proxy.index(1, 0)
            panel._table_view.selectionModel().select(
                proxy_index,
                panel._table_view.selectionModel().Select
                | panel._table_view.selectionModel().Rows,
            )
            panel._on_table_selection_changed(proxy_index, proxy_index)
        assert blocker.args == [20]


class TestResetAndExport:
    def test_reset_restores_full_range(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((25.0, 35.0))
        panel._refresh()
        assert panel._proxy.rowCount() == 1

        panel._reset_btn.click()
        # After reset, slider returns to full range and table shows all
        assert panel._proxy.rowCount() == 5
        lo, hi = panel._sliders["intensity"].value()
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(50.0)

    def test_export_writes_filtered_csv(self, filter_panel_points, tmp_path, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()

        out_path = tmp_path / "filtered.csv"
        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV files (*.csv)"),
        )
        panel._export_btn.click()

        assert out_path.exists()
        out_df = pd.read_csv(out_path)
        assert list(out_df.columns) == list(df.columns)
        assert len(out_df) == 2
        assert out_df["intensity"].tolist() == [30.0, 40.0]

    def test_export_zero_rows_writes_header_only(self, filter_panel_points, tmp_path, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Bypass the slider (its range is clamped to [10, 20]); set
        # _filter_ranges directly to a range that excludes all rows.
        panel._filter_ranges["intensity"] = (100.0, 200.0)
        panel._refresh()

        out_path = tmp_path / "empty.csv"
        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV files (*.csv)"),
        )
        panel._export_btn.click()

        out_df = pd.read_csv(out_path)
        assert len(out_df) == 0
        assert list(out_df.columns) == list(df.columns)

    def test_export_cancel_no_op(self, filter_panel_points, monkeypatch):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0],
            "frame": [0],
            "y": [0.0],
            "x": [0.0],
            "intensity": [10.0],
        })
        _add_points_with_props(viewer, "spots", df)

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        # Should not raise
        panel._export_btn.click()


class TestHistogram:
    def test_checkbox_per_property(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": list(range(5)),
            "frame": list(range(5)),
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
            "area": [100, 200, 300, 400, 500],
        })
        _add_points_with_props(viewer, "spots", df)
        assert set(panel._hist_checkboxes.keys()) == {"intensity", "area"}
        assert all(cb.isChecked() for cb in panel._hist_checkboxes.values())

    def test_histogram_has_one_axes(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert len(panel._hist_fig.axes) == 1

    def test_unchecking_property_removes_artist(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [0.0] * 3,
            "x": [0.0] * 3,
            "intensity": [10.0, 20.0, 30.0],
            "area": [100, 200, 300],
        })
        _add_points_with_props(viewer, "spots", df)
        # Force redraw
        panel._refresh()
        ax = panel._hist_fig.axes[0]
        n_patches_before = len(ax.patches)
        assert n_patches_before > 0

        # Uncheck intensity
        panel._hist_checkboxes["intensity"].setChecked(False)
        ax_after = panel._hist_fig.axes[0]
        n_patches_after = len(ax_after.patches)
        # At least the intensity patches are gone, area still there
        assert n_patches_after < n_patches_before
        assert n_patches_after > 0  # area still drawn


class TestSiblingLayerPoints:
    def test_checkbox_unchecked_no_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [10.0, 20.0],
            "x": [10.0, 20.0],
            "intensity": [1.0, 2.0],
        })
        _add_points_with_props(viewer, "spots", df)
        # Default unchecked
        assert panel._show_filtered_cb.isChecked() is False
        assert all("(filtered)" not in l.name for l in viewer.layers)

    def test_check_creates_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [10.0, 20.0, 30.0],
            "x": [10.0, 20.0, 30.0],
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        names = [l.name for l in viewer.layers]
        assert "spots (filtered)" in names

    def test_filter_shrinks_sibling_data(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2, 3, 4],
            "frame": [0, 1, 2, 3, 4],
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        panel._sliders["intensity"].setValue((25.0, 45.0))
        panel._refresh()
        sibling = viewer.layers["spots (filtered)"]
        assert len(sibling.data) == 2

    def test_uncheck_removes_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "intensity": [10.0, 20.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        assert "spots (filtered)" in [l.name for l in viewer.layers]
        panel._show_filtered_cb.setChecked(False)
        assert "spots (filtered)" not in [l.name for l in viewer.layers]

    def test_source_removed_removes_sibling(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0],
            "frame": [0],
            "y": [0.0],
            "x": [0.0],
            "intensity": [10.0],
        })
        _add_points_with_props(viewer, "spots", df)
        panel._show_filtered_cb.setChecked(True)
        assert "spots (filtered)" in [l.name for l in viewer.layers]
        viewer.layers.remove("spots")
        assert "spots (filtered)" not in [l.name for l in viewer.layers]


class TestSiblingLayerTracks:
    def _add_tracks_with_props(self, viewer, name, tracked_df, meta_df):
        # tracked_df: columns frame, y, x, particle (output of track_blobs)
        # Build (N, 4) napari Tracks data: (track_id, t, y, x)
        rows = tracked_df.sort_values(["particle", "frame"])[
            ["particle", "frame", "y", "x"]
        ].to_numpy()
        layer = viewer.add_tracks(rows, name=name)
        layer.metadata["properties"] = meta_df
        layer.metadata["tracked_df"] = tracked_df
        return layer

    def test_check_creates_tracks_sibling(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1, 2],
            "y": [0.0, 0.0, 5.0, 5.0, 5.0],
            "x": [0.0, 0.0, 5.0, 5.0, 5.0],
            "particle": [1, 1, 2, 2, 2],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2], "length": [2, 3]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._on_layer_changed("t")
        panel._show_filtered_cb.setChecked(True)
        assert "t (filtered)" in [l.name for l in viewer.layers]

    def test_filter_keeps_only_long_tracks(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1, 2],
            "y": [0.0, 0.0, 5.0, 5.0, 5.0],
            "x": [0.0, 0.0, 5.0, 5.0, 5.0],
            "particle": [1, 1, 2, 2, 2],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2], "length": [2, 3]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._on_layer_changed("t")
        panel._show_filtered_cb.setChecked(True)
        # Filter to length >= 3 -> only track 2 (3 vertices)
        panel._sliders["length"].setValue((3, 3))
        panel._refresh()
        sibling = viewer.layers["t (filtered)"]
        assert len(sibling.data) == 3
        assert all(row[0] == 2 for row in sibling.data)


class TestHistogramWhitelist:
    def test_points_panel_has_checkbox_for_each_whitelisted_property(self, filter_panel_points):
        panel, viewer = filter_panel_points
        df = pd.DataFrame({
            "spot_id": [0, 1, 2],
            "frame": [0, 1, 2],
            "y": [0.0] * 3,
            "x": [0.0] * 3,
            "intensity": [10.0, 20.0, 30.0],
            "area": [100, 200, 300],
        })
        _add_points_with_props(viewer, "spots", df)
        # Both whitelisted, both get histogram checkboxes
        assert set(panel._hist_checkboxes.keys()) == {"intensity", "area"}

    def test_tracks_panel_only_has_length_checkbox(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        df = pd.DataFrame({
            "track_id": [0, 1],
            "length": [10, 20],
            "mean_intensity": [50.0, 60.0],
            "max_intensity": [100.0, 200.0],
        })
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        # Tracks filter whitelist is narrowed to just `length`, so only
        # length gets a slider and a histogram checkbox.
        assert set(panel._sliders.keys()) == {"length"}
        assert set(panel._hist_checkboxes.keys()) == {"length"}

    def test_tracks_histogram_uses_raw_values(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        df = pd.DataFrame({"track_id": [0, 1, 2], "length": [10, 20, 30]})
        tracks_data = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        layer = viewer.add_tracks(tracks_data, name="t")
        layer.metadata["properties"] = df
        panel._on_layer_changed("t")
        ax = panel._hist_fig.axes[0]
        # x-axis label is "value" (raw), not "normalized [0, 1]"
        assert ax.get_xlabel() == "value"


class TestTracksFilterPersistence:
    def _add_tracks_with_props(self, viewer, name, tracked_df, meta_df):
        rows = tracked_df.sort_values(["particle", "frame"])[
            ["particle", "frame", "y", "x"]
        ].to_numpy()
        layer = viewer.add_tracks(rows, name=name)
        layer.metadata["properties"] = meta_df
        layer.metadata["tracked_df"] = tracked_df
        return layer

    def test_slider_value_survives_sibling_update(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1, 2, 0, 1, 2, 3],
            "y": [0.0] * 9,
            "x": [0.0] * 9,
            "particle": [1, 1, 2, 2, 2, 3, 3, 3, 3],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2, 3], "length": [2, 3, 4]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._on_layer_changed("t")

        # Enable sibling and apply a filter
        panel._show_filtered_cb.setChecked(True)
        slider = panel._sliders["length"]
        slider.setValue((3, 4))
        panel._refresh()

        # The slider value should persist after sibling has been recreated
        lo, hi = slider.value()
        assert lo == 3
        assert hi == 4

        # And the filter ranges still reflect the user's setting
        assert panel._filter_ranges["length"] == (3, 4)

    def test_sibling_layer_excluded_from_combo(self, filter_panel_tracks):
        panel, viewer = filter_panel_tracks
        tracked_df = pd.DataFrame({
            "frame": [0, 1, 0, 1],
            "y": [0.0] * 4,
            "x": [0.0] * 4,
            "particle": [1, 1, 2, 2],
        })
        meta_df = pd.DataFrame({"track_id": [1, 2], "length": [2, 2]})
        self._add_tracks_with_props(viewer, "t", tracked_df, meta_df)
        panel._on_layer_changed("t")
        panel._show_filtered_cb.setChecked(True)
        # The sibling layer "t (filtered)" exists in the viewer
        assert "t (filtered)" in [l.name for l in viewer.layers]
        # but is NOT shown in the panel's source layer combo
        items = [panel._layer_combo.itemText(i) for i in range(panel._layer_combo.count())]
        assert "t (filtered)" not in items
        assert "t" in items
