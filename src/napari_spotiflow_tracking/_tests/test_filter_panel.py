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
            "category": [1, 1, 1],  # constant
            "intensity": [10.0, 20.0, 30.0],
        })
        _add_points_with_props(viewer, "spots", df)
        assert "category" not in panel._sliders
        assert "intensity" in panel._sliders
        assert "category" in panel._constant_labels

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
