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
