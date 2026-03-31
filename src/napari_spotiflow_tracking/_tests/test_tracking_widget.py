import napari
import napari.layers
import numpy as np
import pytest


@pytest.fixture
def tracking_widget(make_napari_viewer):
    from napari_spotiflow_tracking._tracking_widget import TrackingWidget

    viewer = make_napari_viewer()
    widget = TrackingWidget(viewer)
    return widget, viewer


class TestTrackingWidgetCreation:
    def test_creates_without_error(self, tracking_widget):
        widget, viewer = tracking_widget
        assert widget is not None

    def test_has_track_button(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._track_btn is not None
        assert widget._track_btn.text() == "Track"

    def test_has_search_range_spinbox(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._search_range.value() == 2.0

    def test_has_memory_spinbox(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._memory.value() == 3


class TestTrackingWidgetLayerSync:
    def test_points_combo_shows_points_layers(self, tracking_widget):
        widget, viewer = tracking_widget
        assert widget._points_combo.count() == 0

        viewer.add_points(np.array([[0, 10, 20], [1, 11, 21]]), name="spots")
        assert widget._points_combo.count() == 1

    def test_points_combo_ignores_image_layers(self, tracking_widget):
        widget, viewer = tracking_widget
        viewer.add_image(np.random.rand(64, 64), name="img")
        assert widget._points_combo.count() == 0


class TestTrackingWidgetValidation:
    def test_rejects_no_points_selected(self, tracking_widget):
        widget, viewer = tracking_widget
        widget._track_btn.click()
        assert "select" in widget._status_label.text().lower() or \
               "no" in widget._status_label.text().lower()

    def test_tracks_points_and_adds_tracks_layer(self, tracking_widget):
        widget, viewer = tracking_widget

        points_data = np.array([
            [0, 10, 10],
            [0, 40, 40],
            [1, 11, 11],
            [1, 41, 41],
            [2, 12, 12],
            [2, 42, 42],
        ], dtype=float)
        viewer.add_points(points_data, name="Detected Spots")

        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        tracks_layers = [
            l for l in viewer.layers if isinstance(l, napari.layers.Tracks)
        ]
        assert len(tracks_layers) == 1
        assert "found" in widget._status_label.text().lower()
