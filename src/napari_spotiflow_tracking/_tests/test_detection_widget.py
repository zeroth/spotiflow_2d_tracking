import numpy as np
import pytest


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking._detection_widget import DetectionWidget

    viewer = make_napari_viewer()
    widget = DetectionWidget(viewer)
    return widget, viewer


class TestDetectionWidgetCreation:
    def test_creates_without_error(self, detection_widget):
        widget, viewer = detection_widget
        assert widget is not None

    def test_has_detect_button(self, detection_widget):
        widget, _ = detection_widget
        assert widget._detect_btn is not None
        assert widget._detect_btn.text() == "Detect Spots"

    def test_has_model_combo(self, detection_widget):
        widget, _ = detection_widget
        items = [widget._model_combo.itemText(i) for i in range(widget._model_combo.count())]
        assert "general" in items
        assert "synth_complex" in items
        assert "hybiss" in items
        assert "fluo_live" in items


class TestDetectionWidgetLayerSync:
    def test_image_combo_updates_on_layer_add(self, detection_widget):
        widget, viewer = detection_widget
        assert widget._image_combo.count() == 0

        viewer.add_image(np.random.rand(64, 64))
        assert widget._image_combo.count() == 1

    def test_image_combo_filters_to_images_only(self, detection_widget):
        widget, viewer = detection_widget
        viewer.add_image(np.random.rand(64, 64), name="my_image")
        viewer.add_labels(np.zeros((64, 64), dtype=int), name="my_labels")

        assert widget._image_combo.count() == 1


class TestDetectionWidgetValidation:
    def test_rejects_no_image_selected(self, detection_widget, qtbot):
        widget, viewer = detection_widget
        widget._detect_btn.click()
        assert "select" in widget._status_label.text().lower() or \
               "no" in widget._status_label.text().lower()
