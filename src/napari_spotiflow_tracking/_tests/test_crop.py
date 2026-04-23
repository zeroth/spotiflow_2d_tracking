import numpy as np
import pytest


class TestFirstRectangleBbox:
    def test_simple_2d_rectangle(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[10, 20], [10, 80], [60, 80], [60, 20]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (10, 60, 20, 80)

    def test_clipping_to_image_bounds(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[-10, -10], [-10, 50], [200, 50], [200, -10]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (0, 100, 0, 50)

    def test_entirely_out_of_bounds_raises(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array([[200, 200], [200, 300], [400, 300], [400, 200]], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))

    def test_skips_non_rectangle_and_uses_next_rectangle(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        ellipse = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        rect = np.array([[20, 30], [20, 70], [60, 70], [60, 30]], dtype=float)
        bbox = first_rectangle_bbox(
            [ellipse, rect], ["ellipse", "rectangle"], image_shape=(100, 100),
        )
        assert bbox == (20, 60, 30, 70)

    def test_no_rectangle_raises(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        ellipse = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        with pytest.raises(ValueError, match="No rectangle"):
            first_rectangle_bbox([ellipse], ["ellipse"], image_shape=(100, 100))

    def test_rotated_rectangle_uses_aabb(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        # Square inscribed at 45 deg: corners at (50,30), (70,50), (50,70), (30,50).
        corners = np.array([[50, 30], [70, 50], [50, 70], [30, 50]], dtype=float)
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        assert bbox == (30, 70, 30, 70)

    def test_3d_corner_coords(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        # Rectangle stored with (T, Y, X) coords; T is constant, use last two dims.
        corners = np.array(
            [[4, 10, 20], [4, 10, 80], [4, 60, 80], [4, 60, 20]], dtype=float,
        )
        bbox = first_rectangle_bbox(
            [corners], ["rectangle"], image_shape=(8, 100, 100),
        )
        assert bbox == (10, 60, 20, 80)

    def test_rounds_fractional_coords(self):
        from napari_spotiflow_tracking._preprocessing import first_rectangle_bbox

        corners = np.array(
            [[10.4, 20.6], [10.4, 79.4], [59.6, 79.4], [59.6, 20.6]], dtype=float,
        )
        bbox = first_rectangle_bbox([corners], ["rectangle"], image_shape=(100, 100))
        # 10.4 -> 10, 59.6 -> 60, 20.6 -> 21, 79.4 -> 79
        assert bbox == (10, 60, 21, 79)


class TestCropImage:
    def test_2d_crop(self):
        from napari_spotiflow_tracking._preprocessing import crop_image

        image = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
        result = crop_image(image, (10, 60, 20, 80))
        assert result.shape == (50, 60)
        # Ensure it matches the manual slice
        assert np.array_equal(result, image[10:60, 20:80])

    def test_3d_stack_crop_preserves_time(self):
        from napari_spotiflow_tracking._preprocessing import crop_image

        stack = np.arange(5 * 100 * 100, dtype=np.float32).reshape(5, 100, 100)
        result = crop_image(stack, (10, 60, 20, 80))
        assert result.shape == (5, 50, 60)
        assert np.array_equal(result, stack[:, 10:60, 20:80])
