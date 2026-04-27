import numpy as np
import pytest


class TestApplyCalculation:
    def test_add_returns_sum(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((10, 10), dtype=np.uint16)
        result = apply_calculation(arr, "add", 5.0)
        assert np.array_equal(result, np.full((10, 10), 6.0))

    def test_subtract_returns_difference(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((10, 10), 10, dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 3.0)
        assert np.array_equal(result, np.full((10, 10), 7.0))

    @pytest.mark.parametrize(
        "in_dtype",
        [np.uint8, np.uint16, np.float32, np.float64, np.int32],
    )
    def test_output_dtype_is_float64(self, in_dtype):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=in_dtype)
        result = apply_calculation(arr, "add", 1.0)
        assert result.dtype == np.float64

    def test_3d_stack(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        stack = np.full((5, 10, 10), 2.0, dtype=np.float32)
        result = apply_calculation(stack, "add", 3.0)
        assert result.shape == (5, 10, 10)
        assert np.allclose(result, 5.0)

    def test_subtract_below_zero_no_underflow(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 5.0)
        assert np.array_equal(result, np.full((4, 4), -5.0))

    def test_add_negative_value(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((4, 4), 10.0, dtype=np.float32)
        result = apply_calculation(arr, "add", -3.0)
        assert np.array_equal(result, np.full((4, 4), 7.0))

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown op"):
            apply_calculation(arr, "multiply", 2.0)

    def test_result_is_independent_copy(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=np.float64)
        result = apply_calculation(arr, "add", 0.0)
        result[0, 0] = 999.0
        assert arr[0, 0] == 1.0
