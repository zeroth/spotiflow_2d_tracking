import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_model():
    """Create a mock Spotiflow model that returns predictable spots."""
    model = MagicMock()
    model.predict.return_value = (
        np.array([[10, 20], [30, 40]]),  # 2 spots at (y, x)
        MagicMock(),  # details object
    )
    return model


class TestDetectSpots:
    def test_returns_points_and_details(self):
        from napari_spotiflow_tracking._segmentation import detect_spots

        model = _make_mock_model()
        image = np.random.rand(64, 64).astype(np.float32)

        points, details = detect_spots(image, model, prob_thresh=0.5, min_distance=2)

        assert points.shape == (2, 2)
        assert points[0, 0] == 10
        assert points[0, 1] == 20

    def test_calls_model_predict_with_params(self):
        from napari_spotiflow_tracking._segmentation import detect_spots

        model = _make_mock_model()
        image = np.random.rand(64, 64).astype(np.float32)

        detect_spots(image, model, prob_thresh=0.3, min_distance=5)

        model.predict.assert_called_once_with(
            image,
            prob_thresh=0.3,
            min_distance=5,
            verbose=False,
        )

    def test_returns_empty_array_when_no_spots(self):
        from napari_spotiflow_tracking._segmentation import detect_spots

        model = MagicMock()
        model.predict.return_value = (np.empty((0, 2)), MagicMock())
        image = np.random.rand(64, 64).astype(np.float32)

        points, details = detect_spots(image, model)

        assert points.shape == (0, 2)


class TestLoadModel:
    @patch("napari_spotiflow_tracking._segmentation.Spotiflow")
    def test_loads_pretrained_model(self, MockSpotiflow):
        from napari_spotiflow_tracking._segmentation import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_pretrained.return_value = mock_model

        result = load_model("general", device="cpu")

        MockSpotiflow.from_pretrained.assert_called_once_with(
            "general", map_location="cpu"
        )
        assert result is mock_model

    @patch("napari_spotiflow_tracking._segmentation.Spotiflow")
    def test_loads_custom_model_from_path(self, MockSpotiflow):
        from napari_spotiflow_tracking._segmentation import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_folder.return_value = mock_model

        result = load_model("/path/to/custom_model", device="cuda")

        MockSpotiflow.from_folder.assert_called_once_with(
            "/path/to/custom_model", map_location="cuda"
        )
        assert result is mock_model
