import numpy as np
import pandas as pd


def _tracks_csv(tmp_path, name="tracks.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "frame": [0, 1, 0, 1],
        "y": [10.0, 11.0, 40.0, 41.0],
        "x": [10.0, 11.0, 40.0, 41.0],
        "particle": [0, 0, 1, 1],
    }).to_csv(path, index=False)
    return path


class TestGetReader:
    def test_returns_a_callable_for_a_tracks_csv(self, tmp_path):
        from napari_spotiflow_tracking._reader import get_reader

        assert callable(get_reader(str(_tracks_csv(tmp_path))))

    def test_declines_a_points_csv(self, tmp_path):
        from napari_spotiflow_tracking._reader import get_reader

        path = tmp_path / "points.csv"
        pd.DataFrame({"axis-0": [0], "axis-1": [1.0]}).to_csv(path, index=False)

        assert get_reader(str(path)) is None

    def test_declines_a_non_csv(self, tmp_path):
        from napari_spotiflow_tracking._reader import get_reader

        assert get_reader(str(tmp_path / "stack.tif")) is None

    def test_declines_a_list_of_paths(self, tmp_path):
        from napari_spotiflow_tracking._reader import get_reader

        assert get_reader([str(_tracks_csv(tmp_path))]) is None

    def test_declines_a_missing_file(self, tmp_path):
        from napari_spotiflow_tracking._reader import get_reader

        assert get_reader(str(tmp_path / "nope.csv")) is None


class TestReadTracks:
    def test_returns_one_tracks_layer_tuple(self, tmp_path):
        from napari_spotiflow_tracking._reader import read_tracks

        layers = read_tracks(str(_tracks_csv(tmp_path)))

        assert len(layers) == 1
        data, attrs, layer_type = layers[0]
        assert layer_type == "tracks"
        assert data.shape == (4, 4)
        np.testing.assert_array_equal(data[:, 0], [0, 0, 1, 1])

    def test_carries_the_widget_metadata_keys(self, tmp_path):
        from napari_spotiflow_tracking._reader import read_tracks

        _, attrs, _ = read_tracks(str(_tracks_csv(tmp_path)))[0]

        meta = attrs["metadata"]
        assert isinstance(meta["properties"], pd.DataFrame)
        assert isinstance(meta["tracked_df"], pd.DataFrame)
        assert meta["n_dropped_rows"] == 0

    def test_names_the_layer_after_the_folder(self, tmp_path):
        from napari_spotiflow_tracking._reader import read_tracks

        folder = tmp_path / "cell01"
        folder.mkdir()

        _, attrs, _ = read_tracks(str(_tracks_csv(folder)))[0]

        assert attrs["name"] == "cell01"


class TestManifest:
    def test_reader_contribution_is_registered(self):
        from pathlib import Path

        from npe2 import PluginManifest

        import napari_spotiflow_tracking

        manifest = PluginManifest.from_file(
            Path(napari_spotiflow_tracking.__file__).parent / "napari.yaml"
        )

        readers = manifest.contributions.readers
        assert len(readers) == 1
        assert readers[0].filename_patterns == ["*.csv"]
        assert readers[0].accepts_directories is False
