import numpy as np
import pytest


@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking._cleanup_widget import PreProcessingWidget

    viewer = make_napari_viewer()
    widget = PreProcessingWidget(viewer)
    return widget, viewer


class TestPreProcessingWidgetCreation:
    def test_creates_without_error(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget is not None

    def test_has_crop_controls(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget._shapes_combo is not None
        assert widget._crop_btn is not None
        assert widget._crop_btn.text() == "Apply Crop"


class TestCropWorkflow:
    def test_new_shapes_layer_button_creates_and_selects(self, preproc_widget):
        widget, viewer = preproc_widget

        widget._new_shapes_layer()

        # Exactly one Shapes layer named "Crop ROI" is present
        shapes_layers = [l for l in viewer.layers if l.__class__.__name__ == "Shapes"]
        assert len(shapes_layers) == 1
        assert shapes_layers[0].name == "Crop ROI"
        # Combo is populated and points at the new layer
        assert widget._shapes_combo.currentText() == "Crop ROI"

    def test_shapes_combo_refreshes_on_layer_add(self, preproc_widget):
        widget, viewer = preproc_widget

        viewer.add_shapes(name="my_rois")

        items = [widget._shapes_combo.itemText(i) for i in range(widget._shapes_combo.count())]
        assert "my_rois" in items

    def test_apply_crop_on_2d_image_adds_cropped_layer(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_rectangles(
            np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        cropped = viewer.layers["img (crop)"]
        assert cropped.data.shape == (50, 60)
        assert np.array_equal(cropped.data, image[10:60, 20:80])

    def test_apply_crop_on_3d_stack_preserves_time(self, preproc_widget):
        widget, viewer = preproc_widget

        stack = np.arange(5 * 100 * 100, dtype=np.float32).reshape(5, 100, 100)
        viewer.add_image(stack, name="stack")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("stack")

        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_rectangles(
            np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        cropped = viewer.layers["stack (crop)"]
        assert cropped.data.shape == (5, 50, 60)

    def test_apply_crop_without_rectangle_shows_no_output(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((100, 100), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        # Shapes layer with an ellipse only
        shapes_layer = viewer.add_shapes(name="rois")
        shapes_layer.add_ellipses(
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float),
        )
        widget._shapes_combo.setCurrentText("rois")

        widget._run_crop()

        # No "img (crop)" layer was added
        names = [l.name for l in viewer.layers]
        assert "img (crop)" not in names


class TestCalculationWorkflow:
    def test_section_exists_with_defaults(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget._calc_op_combo is not None
        assert widget._calc_value is not None
        assert widget._calc_apply_btn is not None
        items = [
            widget._calc_op_combo.itemText(i)
            for i in range(widget._calc_op_combo.count())
        ]
        assert items == ["Add", "Subtract"]
        assert widget._calc_op_combo.currentText() == "Add"
        assert widget._calc_apply_btn.text() == "Apply"

    def test_mean_button_populates_spinbox(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.full((10, 10), 4.0, dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._run_calc_mean()

        assert widget._calc_value.value() == pytest.approx(4.0)

    def test_apply_add_creates_layer_with_offset(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5.0)
        widget._run_calc_apply()

        assert "img (+5)" in [l.name for l in viewer.layers]
        assert np.array_equal(
            viewer.layers["img (+5)"].data, np.full((10, 10), 5.0),
        )

    def test_apply_subtract_creates_layer_with_negative(self, preproc_widget):
        widget, viewer = preproc_widget

        image = np.zeros((10, 10), dtype=np.float32)
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        widget._calc_op_combo.setCurrentText("Subtract")
        widget._calc_value.setValue(3.0)
        widget._run_calc_apply()

        assert "img (-3)" in [l.name for l in viewer.layers]
        assert np.array_equal(
            viewer.layers["img (-3)"].data, np.full((10, 10), -3.0),
        )

    def test_apply_with_no_image_does_nothing(self, preproc_widget):
        widget, viewer = preproc_widget

        before_names = [l.name for l in viewer.layers]
        # No image is in the combo (it's empty)
        widget._calc_op_combo.setCurrentText("Add")
        widget._calc_value.setValue(5.0)
        widget._run_calc_apply()

        after_names = [l.name for l in viewer.layers]
        assert after_names == before_names

    def test_mean_with_no_image_does_nothing(self, preproc_widget):
        widget, _ = preproc_widget

        before = widget._calc_value.value()
        widget._run_calc_mean()
        # Spinbox is unchanged (no image was selected so nothing happened)
        assert widget._calc_value.value() == before
