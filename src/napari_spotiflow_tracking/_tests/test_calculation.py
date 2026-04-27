import numpy as np
import pytest


class TestApplyCalculation:
    def test_add_returns_sum_same_dtype(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((10, 10), dtype=np.uint16)
        result = apply_calculation(arr, "add", 5)
        assert np.array_equal(result, np.full((10, 10), 6, dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_subtract_returns_difference_same_dtype(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((10, 10), 10, dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 3)
        assert np.array_equal(result, np.full((10, 10), 7, dtype=np.uint16))
        assert result.dtype == np.uint16

    @pytest.mark.parametrize(
        "in_dtype",
        [np.uint8, np.uint16, np.float32, np.float64, np.int32],
    )
    def test_output_dtype_matches_input(self, in_dtype):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=in_dtype)
        result = apply_calculation(arr, "add", 1)
        assert result.dtype == in_dtype

    def test_3d_stack(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        stack = np.full((5, 10, 10), 2.0, dtype=np.float32)
        result = apply_calculation(stack, "add", 3)
        assert result.shape == (5, 10, 10)
        assert np.allclose(result, 5.0)
        assert result.dtype == np.float32

    def test_subtract_clips_at_zero_for_unsigned(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "subtract", 5)
        assert np.array_equal(result, np.zeros((4, 4), dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_add_clips_at_max_for_unsigned(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.full((4, 4), 65000, dtype=np.uint16)
        result = apply_calculation(arr, "add", 1000)
        assert np.array_equal(result, np.full((4, 4), 65535, dtype=np.uint16))
        assert result.dtype == np.uint16

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.zeros((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="Unknown op"):
            apply_calculation(arr, "multiply", 2)

    def test_result_is_independent_copy(self):
        from napari_spotiflow_tracking._preprocessing import apply_calculation

        arr = np.ones((4, 4), dtype=np.uint16)
        result = apply_calculation(arr, "add", 0)
        result[0, 0] = 999
        assert arr[0, 0] == 1
