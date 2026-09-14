import numpy as np
import pandas as pd
import pytest


def _widget_export(tmp_path, name="exported.csv"):
    """A CSV shaped like the Spot Tracking widget's own export."""
    path = tmp_path / name
    pd.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "y": [10.0, 11.0, 12.0, 40.0, 41.0, 42.0],
        "x": [10.0, 11.0, 12.0, 40.0, 41.0, 42.0],
        "particle": [0, 0, 0, 1, 1, 1],
        "intensity": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }).to_csv(path, index=False)
    return path


def _bulk_tracks(tmp_path, name="tracks.csv"):
    """A CSV shaped like scripts/bulk_track_analysis.py's tracks.csv."""
    path = tmp_path / name
    pd.DataFrame({
        "frame": [0, 1, 0, 1],
        "y": [10.0, 11.0, 40.0, 41.0],
        "x": [10.0, 11.0, 40.0, 41.0],
        "particle": [0, 0, 1, 1],
        "amp": [7.7, 7.8, np.nan, 8.1],
        "sigma_x": [1.2, 1.3, np.nan, 1.4],
        "mean_intensity": [9.8, 9.9, 10.1, 10.2],
    }).to_csv(path, index=False)
    return path


class TestIsTracksCsv:
    def test_accepts_widget_export(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        assert is_tracks_csv(_widget_export(tmp_path)) is True

    def test_accepts_bulk_tracks_csv(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        assert is_tracks_csv(_bulk_tracks(tmp_path)) is True

    def test_accepts_track_id_alias(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        path = tmp_path / "alias.csv"
        pd.DataFrame({"frame": [0], "y": [1.0], "x": [2.0],
                      "track_id": [3]}).to_csv(path, index=False)

        assert is_tracks_csv(path) is True

    def test_declines_a_points_csv(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        path = tmp_path / "points.csv"
        pd.DataFrame({"axis-0": [0], "axis-1": [1.0],
                      "axis-2": [2.0]}).to_csv(path, index=False)

        assert is_tracks_csv(path) is False

    def test_declines_a_csv_without_a_track_id(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        path = tmp_path / "spots.csv"
        pd.DataFrame({"frame": [0], "y": [1.0],
                      "x": [2.0]}).to_csv(path, index=False)

        assert is_tracks_csv(path) is False

    def test_declines_an_unreadable_file(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import is_tracks_csv

        path = tmp_path / "junk.csv"
        path.write_bytes(b"\x00\x01\x02")

        assert is_tracks_csv(path) is False


class TestReadTracksCsv:
    def test_data_is_track_frame_y_x(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        result = read_tracks_csv(_widget_export(tmp_path))

        assert result.data.shape == (6, 4)
        np.testing.assert_array_equal(result.data[:, 0], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result.data[:, 1], [0, 1, 2, 0, 1, 2])
        assert result.n_tracks == 2

    def test_renames_track_id_to_particle(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = tmp_path / "alias.csv"
        pd.DataFrame({"frame": [0, 1], "y": [1.0, 2.0], "x": [1.0, 2.0],
                      "track_id": [7, 7]}).to_csv(path, index=False)

        result = read_tracks_csv(path)

        assert "particle" in result.tracked_df.columns
        assert result.tracked_df["particle"].tolist() == [7, 7]

    def test_sorts_by_track_then_frame(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = tmp_path / "shuffled.csv"
        pd.DataFrame({"frame": [2, 0, 1, 1], "y": [1.0, 2.0, 3.0, 4.0],
                      "x": [1.0, 2.0, 3.0, 4.0],
                      "particle": [1, 1, 0, 1]}).to_csv(path, index=False)

        result = read_tracks_csv(path)

        assert result.data[:, 0].tolist() == [0, 1, 1, 1]
        assert result.data[1:, 1].tolist() == [0, 1, 2]

    def test_drops_rows_missing_a_coordinate(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = tmp_path / "holes.csv"
        pd.DataFrame({"frame": [0, 1, 2], "y": [1.0, np.nan, 3.0],
                      "x": [1.0, 2.0, 3.0],
                      "particle": [0, 0, 0]}).to_csv(path, index=False)

        result = read_tracks_csv(path)

        assert result.n_dropped == 1
        assert len(result.data) == 2

    def test_keeps_nan_in_non_coordinate_columns(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        result = read_tracks_csv(_bulk_tracks(tmp_path))

        assert result.n_dropped == 0
        assert result.tracked_df["amp"].isna().sum() == 1

    def test_maps_mean_intensity_onto_intensity(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        result = read_tracks_csv(_bulk_tracks(tmp_path))

        assert "intensity" in result.tracked_df.columns
        assert result.tracked_df["intensity"].tolist() == [9.8, 9.9, 10.1, 10.2]

    def test_keeps_an_existing_intensity_column(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        result = read_tracks_csv(_widget_export(tmp_path))

        assert result.tracked_df["intensity"].tolist() == [5, 6, 7, 8, 9, 10]

    def test_computes_properties_without_a_sibling(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        result = read_tracks_csv(_widget_export(tmp_path))

        assert result.properties_from is None
        assert set(result.properties["track_id"]) == {0, 1}
        assert "mean_speed" in result.properties.columns

    def test_prefers_the_sibling_track_analysis_csv(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = _bulk_tracks(tmp_path)
        pd.DataFrame({"track_id": [0, 1], "length": [2, 2],
                      "mean_speed": [1.4, 1.4], "D": [0.5, 0.6],
                      "alpha": [1.9, 0.2],
                      "motility_class": ["directed", "constrained"]}
                     ).to_csv(tmp_path / "track_analysis.csv", index=False)

        result = read_tracks_csv(path)

        assert result.properties_from == "track_analysis.csv"
        assert result.properties["motility_class"].tolist() == [
            "directed", "constrained"]

    def test_falls_back_when_the_sibling_is_unusable(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = _bulk_tracks(tmp_path)
        (tmp_path / "track_analysis.csv").write_text("not,a,table\n")

        result = read_tracks_csv(path)

        assert result.properties_from is None
        assert set(result.properties["track_id"]) == {0, 1}

    def test_rejects_a_csv_without_track_columns(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = tmp_path / "points.csv"
        pd.DataFrame({"axis-0": [0], "axis-1": [1.0]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing column"):
            read_tracks_csv(path)

    def test_rejects_a_file_with_no_complete_row(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import read_tracks_csv

        path = tmp_path / "empty.csv"
        pd.DataFrame({"frame": [0], "y": [np.nan], "x": [1.0],
                      "particle": [0]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="no row"):
            read_tracks_csv(path)


class TestLayerNameFor:
    def test_uses_the_folder_for_a_generic_tracks_csv(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import layer_name_for

        folder = tmp_path / "DE 10FPS 01-1_blur"
        folder.mkdir()

        assert layer_name_for(folder / "tracks.csv") == "DE 10FPS 01-1_blur"

    def test_uses_the_stem_for_a_named_file(self, tmp_path):
        from napari_spotiflow_tracking._tracks_io import layer_name_for

        assert layer_name_for(tmp_path / "cell01_tracks.csv") == "cell01_tracks"
