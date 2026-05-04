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
