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


class TestTracksLayerCombo:
    def test_lists_a_tracks_layer_from_a_run(self, tracking_widget):
        widget, viewer = tracking_widget

        viewer.add_points(np.array([
            [0, 10, 10], [1, 11, 11], [2, 12, 12],
        ], dtype=float), name="Detected Spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        assert widget._tracks_combo.count() == 1
        assert widget._tracks_combo.currentText() == "Tracked Spots"

    def test_ignores_a_tracks_layer_without_tracked_df(self, tracking_widget):
        widget, viewer = tracking_widget

        viewer.add_tracks(np.array([
            [0, 0, 10.0, 10.0], [0, 1, 11.0, 11.0],
        ]), name="Bare Tracks")

        assert widget._tracks_combo.count() == 0

    def test_selects_the_second_run_not_the_first(self, tracking_widget):
        """A second run used to leave the charts bound to the first layer."""
        widget, viewer = tracking_widget

        viewer.add_points(np.array([
            [0, 10, 10], [1, 11, 11], [2, 12, 12],
        ], dtype=float), name="Detected Spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()
        widget._track_btn.click()

        assert widget._tracks_combo.count() == 2
        selected = widget._tracks_layer()
        assert selected is viewer.layers[-1]

    def test_analysis_reads_a_reader_produced_layer(self, tracking_widget,
                                                   tmp_path):
        import pandas as pd

        from napari_spotiflow_tracking._reader import read_tracks

        widget, viewer = tracking_widget
        path = tmp_path / "tracks.csv"
        pd.DataFrame({
            "frame": [0, 1, 2], "y": [10.0, 11.0, 12.0],
            "x": [10.0, 11.0, 12.0], "particle": [4, 4, 4],
            "intensity": [1.0, 2.0, 3.0],
        }).to_csv(path, index=False)

        data, attrs, _ = read_tracks(str(path))[0]
        viewer.add_tracks(data, **attrs)

        assert widget._tracks_combo.currentText() == tmp_path.name
        sub = widget._get_track_subdf(4)
        assert sub is not None and len(sub) == 3
        np.testing.assert_array_equal(
            widget._get_track_intensity(4), [1.0, 2.0, 3.0]
        )
