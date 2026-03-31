import numpy as np
import pandas as pd
import pytest


class TestTrackBlobs:
    def test_assigns_particle_ids(self):
        from napari_spotiflow_tracking._tracking import track_blobs

        df = pd.DataFrame({
            "frame": [0, 0, 1, 1, 2, 2],
            "y": [10, 40, 11, 41, 12, 42],
            "x": [10, 40, 11, 41, 12, 42],
        })

        tracked = track_blobs(df, search_range=5, memory=0)

        assert "particle" in tracked.columns
        assert tracked["particle"].nunique() == 2

    def test_returns_dataframe(self):
        from napari_spotiflow_tracking._tracking import track_blobs

        df = pd.DataFrame({
            "frame": [0, 1],
            "y": [10, 11],
            "x": [10, 11],
        })

        result = track_blobs(df, search_range=5, memory=0)

        assert isinstance(result, pd.DataFrame)


class TestTracksToNapari:
    def test_output_shape(self):
        from napari_spotiflow_tracking._tracking import tracks_to_napari

        df = pd.DataFrame({
            "frame": [0, 1, 0, 1],
            "y": [10, 11, 40, 41],
            "x": [10, 11, 40, 41],
            "particle": [0, 0, 1, 1],
        })

        result = tracks_to_napari(df)

        assert result.shape == (4, 4)

    def test_column_order_is_track_frame_y_x(self):
        from napari_spotiflow_tracking._tracking import tracks_to_napari

        df = pd.DataFrame({
            "frame": [0, 1],
            "y": [10, 20],
            "x": [30, 40],
            "particle": [5, 5],
        })

        result = tracks_to_napari(df)

        assert result[0, 0] == 5
        assert result[0, 1] == 0
        assert result[0, 2] == 10
        assert result[0, 3] == 30
