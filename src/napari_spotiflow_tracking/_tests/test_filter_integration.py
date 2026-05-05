"""End-to-end tests for FilterPanel embedded in Detection and Tracking widgets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking._detection_widget import DetectionWidget
    viewer = make_napari_viewer()
    widget = DetectionWidget(viewer)
    return widget, viewer


class TestDetectionPointsMetadata:
    def test_detect_finished_populates_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Synthetic 2-frame stack: 1 spot per frame
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([100.0, 200.0])
        # Call _on_detection_finished directly with our synthetic data
        widget._on_detection_finished(points, None, intensities)

        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        assert len(layers) == 1
        props = layers[0].metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        assert list(props.columns) == ["spot_id", "frame", "y", "x", "intensity"]
        assert props["intensity"].tolist() == [100.0, 200.0]
        assert props["spot_id"].tolist() == [0, 1]

    def test_detect_finished_2d_points_no_frame(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[5.0, 5.0], [6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)
        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        props = layers[0].metadata.get("properties")
        assert list(props.columns) == ["spot_id", "y", "x", "intensity"]


class TestMaskGenMerge:
    def test_merge_into_existing_points_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Pre-populate a Points layer with detection-style metadata
        points_data = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        points_layer = viewer.add_points(points_data, name="Detected Spots")
        points_layer.metadata["properties"] = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "intensity": [10.0, 20.0],
        })
        widget._mask_target_points = points_layer

        # Simulate mask-gen finishing with regionprops including new columns
        mask = np.zeros((2, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "area": [10, 20],
            "mean_intensity": [50.0, 60.0],
        })
        widget._on_mask_finished(mask, props_df)

        merged = points_layer.metadata["properties"]
        # Old columns retained
        assert "spot_id" in merged.columns
        assert "intensity" in merged.columns
        # New columns added
        assert "area" in merged.columns
        assert "mean_intensity" in merged.columns
        # Values merged correctly
        assert merged.loc[merged["spot_id"] == 0, "area"].iloc[0] == 10
        assert merged.loc[merged["spot_id"] == 1, "mean_intensity"].iloc[0] == 60.0

    def test_merge_when_target_has_no_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Points layer without properties metadata (user-imported)
        points_data = np.array([[0, 5.0, 5.0]])
        points_layer = viewer.add_points(points_data, name="My Points")
        widget._mask_target_points = points_layer

        mask = np.zeros((1, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0],
            "y": [5.0],
            "x": [5.0],
            "area": [42],
        })
        widget._on_mask_finished(mask, props_df)

        out = points_layer.metadata["properties"]
        assert isinstance(out, pd.DataFrame)
        assert "spot_id" in out.columns
        assert "area" in out.columns
        assert out["area"].iloc[0] == 42


class TestDetectionFilterPanelEmbedded:
    def test_widget_has_filter_panel(self, detection_widget):
        widget, viewer = detection_widget
        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panels = widget.findChildren(FilterPanel)
        assert len(panels) == 1
        assert panels[0]._layer_kind == "Points"
        assert panels[0]._id_column == "spot_id"

    def test_filter_panel_picks_up_detected_layer(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)

        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panel = widget.findChildren(FilterPanel)[0]
        # Manually force layer combo refresh (in tests events may be eager)
        panel._refresh_layer_combo()
        panel._on_layer_changed("Detected Spots")
        assert "intensity" in panel._sliders
