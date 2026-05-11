"""End-to-end tests for FilterPanel embedded in Detection and Tracking widgets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking._detection_widget import DetectionWidget
    viewer = make_napari_viewer()
    widget = DetectionWidget(viewer)
    return widget, viewer


class TestDetectionPointsMetadata:
    def test_detect_finished_populates_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Synthetic 2-frame stack: 1 spot per frame
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([100.0, 200.0])
        # Call _on_detection_finished directly with our synthetic data
        widget._on_detection_finished(points, None, intensities)

        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        assert len(layers) == 1
        props = layers[0].metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        assert list(props.columns) == ["spot_id", "frame", "y", "x", "intensity"]
        assert props["intensity"].tolist() == [100.0, 200.0]
        assert props["spot_id"].tolist() == [0, 1]

    def test_detect_finished_2d_points_no_frame(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[5.0, 5.0], [6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)
        layers = [l for l in viewer.layers if l.name == "Detected Spots"]
        props = layers[0].metadata.get("properties")
        assert list(props.columns) == ["spot_id", "y", "x", "intensity"]


class TestMaskGenMerge:
    def test_merge_into_existing_points_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Pre-populate a Points layer with detection-style metadata
        points_data = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        points_layer = viewer.add_points(points_data, name="Detected Spots")
        points_layer.metadata["properties"] = pd.DataFrame({
            "spot_id": [0, 1],
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "intensity": [10.0, 20.0],
        })
        widget._mask_target_points = points_layer

        # Simulate mask-gen finishing with regionprops including new columns
        mask = np.zeros((2, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0, 1],
            "y": [5.0, 6.0],
            "x": [5.0, 6.0],
            "area": [10, 20],
            "mean_intensity": [50.0, 60.0],
        })
        widget._on_mask_finished(mask, props_df)

        merged = points_layer.metadata["properties"]
        # Old columns retained
        assert "spot_id" in merged.columns
        assert "intensity" in merged.columns
        # New columns added
        assert "area" in merged.columns
        assert "mean_intensity" in merged.columns
        # Values merged correctly
        assert merged.loc[merged["spot_id"] == 0, "area"].iloc[0] == 10
        assert merged.loc[merged["spot_id"] == 1, "mean_intensity"].iloc[0] == 60.0

    def test_merge_when_target_has_no_properties(self, detection_widget):
        widget, viewer = detection_widget
        # Points layer without properties metadata (user-imported)
        points_data = np.array([[0, 5.0, 5.0]])
        points_layer = viewer.add_points(points_data, name="My Points")
        widget._mask_target_points = points_layer

        mask = np.zeros((1, 16, 16), dtype=int)
        props_df = pd.DataFrame({
            "frame": [0],
            "y": [5.0],
            "x": [5.0],
            "area": [42],
        })
        widget._on_mask_finished(mask, props_df)

        out = points_layer.metadata["properties"]
        assert isinstance(out, pd.DataFrame)
        assert "spot_id" in out.columns
        assert "area" in out.columns
        assert out["area"].iloc[0] == 42


    def test_integrated_detect_with_mask_merges_regionprops(self, detection_widget):
        widget, viewer = detection_widget
        # Add an Image layer the widget can find via _image_combo
        image = np.ones((2, 16, 16), dtype=float) * 100
        viewer.add_image(image, name="img")
        widget._refresh_image_combo()
        widget._image_combo.setCurrentText("img")

        # Synthetic detection result with a non-trivial mask: each frame has
        # one labeled region of area 4 around the detected point.
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([100.0, 200.0])
        mask = np.zeros((2, 16, 16), dtype=int)
        mask[0, 4:6, 4:6] = 1
        mask[1, 5:7, 5:7] = 1
        widget._on_detection_finished(points, mask, intensities)

        points_layer = next(l for l in viewer.layers if l.name == "Detected Spots")
        props = points_layer.metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        # Original detect-time columns retained
        assert "spot_id" in props.columns
        assert "intensity" in props.columns
        # Regionprops columns merged in
        assert "area" in props.columns
        assert "mean_intensity" in props.columns


class TestDetectionFilterPanelEmbedded:
    def test_widget_has_filter_panel(self, detection_widget):
        widget, viewer = detection_widget
        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panels = widget.findChildren(FilterPanel)
        assert len(panels) == 1
        assert panels[0]._layer_kind == "Points"
        assert panels[0]._id_column == "spot_id"

    def test_filter_panel_picks_up_detected_layer(self, detection_widget):
        widget, viewer = detection_widget
        points = np.array([[0, 5.0, 5.0], [1, 6.0, 6.0]])
        intensities = np.array([10.0, 20.0])
        widget._on_detection_finished(points, None, intensities)

        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panel = widget.findChildren(FilterPanel)[0]
        # Manually force layer combo refresh (in tests events may be eager)
        panel._refresh_layer_combo()
        panel._on_layer_changed("Detected Spots")
        assert "intensity" in panel._sliders


@pytest.fixture
def tracking_widget(make_napari_viewer):
    from napari_spotiflow_tracking._tracking_widget import TrackingWidget
    viewer = make_napari_viewer()
    widget = TrackingWidget(viewer)
    return widget, viewer


class TestTrackingPropertiesAndFilter:
    def _add_points_layer(self, viewer):
        # 6 detections, 2 tracks (3 frames each)
        data = np.array([
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [2, 2.0, 0.0],
            [0, 10.0, 10.0],
            [1, 11.0, 10.0],
            [2, 12.0, 10.0],
        ])
        viewer.add_points(data, name="spots")

    def test_widget_has_image_combo(self, tracking_widget):
        widget, viewer = tracking_widget
        assert widget._image_combo is not None
        assert widget._image_combo.count() == 0  # no image layers initially

    def test_widget_has_filter_panel(self, tracking_widget):
        widget, viewer = tracking_widget
        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panels = widget.findChildren(FilterPanel)
        assert len(panels) == 1
        assert panels[0]._layer_kind == "Tracks"
        assert panels[0]._id_column == "track_id"

    def test_track_no_image_populates_geometric_properties(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        tracks_layers = [l for l in viewer.layers if l.name == "Tracked Spots"]
        assert len(tracks_layers) == 1
        props = tracks_layers[0].metadata.get("properties")
        assert isinstance(props, pd.DataFrame)
        assert set(props.columns) >= {
            "track_id", "length", "total_displacement",
            "net_displacement", "mean_speed",
        }
        # Intensity columns absent
        assert "mean_intensity" not in props.columns

    def test_track_with_image_includes_intensity(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        viewer.add_image(np.ones((3, 32, 32)), name="img")
        widget._points_combo.setCurrentText("spots")
        widget._image_combo.setCurrentText("img")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        tracks_layers = [l for l in viewer.layers if l.name == "Tracked Spots"]
        props = tracks_layers[0].metadata.get("properties")
        assert "mean_intensity" in props.columns

    def test_tracked_df_stored_on_layer(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_points_layer(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()
        tracks_layer = viewer.layers["Tracked Spots"]
        assert isinstance(tracks_layer.metadata.get("tracked_df"), pd.DataFrame)


class TestFilterEndToEndTracking:
    def test_slider_filters_sibling_tracks_layer(self, tracking_widget):
        widget, viewer = tracking_widget
        # 2 tracks: short (length 2) and long (length 4)
        data = np.array([
            [0, 0.0, 0.0],
            [1, 1.0, 0.0],
            [0, 10.0, 10.0],
            [1, 11.0, 10.0],
            [2, 12.0, 10.0],
            [3, 13.0, 10.0],
        ])
        viewer.add_points(data, name="spots")
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        from napari_spotiflow_tracking._filter_panel import FilterPanel
        panel = widget.findChildren(FilterPanel)[0]
        panel._refresh_layer_combo()
        panel._on_layer_changed("Tracked Spots")
        assert "length" in panel._sliders

        # Check sibling toggle and filter
        panel._show_filtered_cb.setChecked(True)
        # Filter to length >= 4 -> only the long track
        slider = panel._sliders["length"]
        slider.setValue((4, 4))
        panel._refresh()

        sibling = viewer.layers["Tracked Spots (filtered)"]
        # 4 vertices for the single long track
        assert len(sibling.data) == 4


class TestGpuFitColumnRenames:
    """Verify make_gpu_spot_mask emits regionprops-style column names.

    Does not exercise the actual GPU fitter — just the post-fit mask + props
    builder, which is pure-numpy and works on CPU.
    """

    def test_make_gpu_spot_mask_returns_regionprops_columns(self):
        from napari_spotiflow_tracking._fitting import make_gpu_spot_mask
        # 1 frame, 16x16 image, one fitted spot at (8, 8) with sigma 1.5
        df = pd.DataFrame({
            "time": [0],  # 0-indexed (matches frame convention)
            "x": [8.0],
            "y": [8.0],
            "x_fit": [8.0],
            "y_fit": [8.0],
            "sigma_x": [1.5],
            "sigma_y": [1.5],
        })
        intensity_stack = np.full((1, 16, 16), 100.0)
        mask, props = make_gpu_spot_mask(
            stack_shape=(1, 16, 16), df=df,
            intensity_stack=intensity_stack,
        )
        # Column-rename contract
        assert "area" in props.columns
        assert "mean_intensity" in props.columns
        assert "max_intensity" in props.columns
        assert "spot_radius" in props.columns
        # And the OLD column names are NOT present
        assert "spot_area" not in props.columns
        assert "spot_mean" not in props.columns
        assert "spot_max" not in props.columns
        # Mask is non-zero around the spot
        assert mask.sum() > 0
        # Intensity stats reflect the constant-100 image
        assert props["mean_intensity"].iloc[0] == 100.0
        assert props["max_intensity"].iloc[0] == 100.0


class TestDetectionAnalysisTab:
    def test_widget_has_two_tabs(self, detection_widget):
        widget, _ = detection_widget
        assert widget._tabs.count() == 2
        assert widget._tabs.tabText(0) == "Control"
        assert widget._tabs.tabText(1) == "Analysis"

    def test_analysis_tab_has_table_and_chart(self, detection_widget):
        widget, _ = detection_widget
        # Table view re-parented from FilterPanel.
        assert widget._filter_panel.table_view is not None
        # Spots-per-frame canvas exists.
        assert widget._spots_per_frame_fig is not None
        assert widget._spots_per_frame_canvas is not None

    def test_spots_per_frame_chart_redraws_on_filter_change(self, detection_widget):
        widget, viewer = detection_widget
        # Synthetic detection: 3 spots in frame 0, 5 in frame 1
        points = np.array([
            [0, 1.0, 1.0], [0, 2.0, 2.0], [0, 3.0, 3.0],
            [1, 4.0, 4.0], [1, 5.0, 5.0], [1, 6.0, 6.0], [1, 7.0, 7.0], [1, 8.0, 8.0],
        ])
        intensities = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        widget._on_detection_finished(points, None, intensities)
        # Force the FilterPanel to pick up the new layer
        widget._filter_panel._refresh_layer_combo()
        widget._filter_panel._on_layer_changed("Detected Spots")
        # The chart should have a single Axes with 2 bars (frames 0 and 1)
        ax = widget._spots_per_frame_fig.axes[0]
        bars = [p for p in ax.patches]
        assert len(bars) == 2


class TestTrackingAnalysisTab:
    def test_widget_has_two_tabs(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._tabs.count() == 2
        assert widget._tabs.tabText(0) == "Control"
        assert widget._tabs.tabText(1) == "Analysis"

    def test_analysis_tab_has_table_and_two_charts(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._filter_panel.table_view is not None
        assert widget._length_fig is not None
        assert widget._speed_fig is not None

    def test_charts_redraw_on_filter_change(self, tracking_widget):
        widget, viewer = tracking_widget
        # 2 tracks: short (length 2) and long (length 4) — distinct lengths
        # so the histogram should have non-zero bars.
        data = np.array([
            [0, 0.0, 0.0], [1, 1.0, 0.0],
            [0, 10.0, 10.0], [1, 11.0, 10.0], [2, 12.0, 10.0], [3, 13.0, 10.0],
        ])
        viewer.add_points(data, name="spots")
        widget._points_combo.setCurrentText("spots")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()
        widget._filter_panel._refresh_layer_combo()
        widget._filter_panel._on_layer_changed("Tracked Spots")
        # Both charts should have axes drawn
        assert len(widget._length_fig.axes) == 1
        assert len(widget._speed_fig.axes) == 1


class TestPerTrackAnalysis:
    def _add_synthetic_tracks(self, viewer):
        # Two tracks: a long one (8 frames, moving 1 px/frame) and a short
        # one (3 frames). Image is constant so intensity is uniform.
        data = np.array([
            # particle 1, 8 frames
            [0, 0.0, 0.0],
            [1, 0.0, 1.0],
            [2, 0.0, 2.0],
            [3, 0.0, 3.0],
            [4, 0.0, 4.0],
            [5, 0.0, 5.0],
            [6, 0.0, 6.0],
            [7, 0.0, 7.0],
            # particle 2, 3 frames
            [0, 10.0, 10.0],
            [1, 10.0, 10.0],
            [2, 10.0, 10.0],
        ])
        viewer.add_points(data, name="spots")
        image = np.full((8, 32, 32), 100.0)
        viewer.add_image(image, name="img")

    def _get_track_ids_by_length(self, viewer):
        """Return (long_id, short_id) sorted by vertex count descending."""
        layer = viewer.layers["Tracked Spots"]
        tracked_df = layer.metadata["tracked_df"]
        lengths = tracked_df.groupby("particle").size().sort_values(ascending=False)
        ids = lengths.index.tolist()
        return ids[0], ids[-1]  # (longest, shortest)

    def test_track_selection_redraws_charts_with_intensity(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_synthetic_tracks(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._image_combo.setCurrentText("img")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        long_id, _ = self._get_track_ids_by_length(viewer)
        widget._on_track_selected(long_id)

        # MSD chart has axes drawn (length > 5)
        assert len(widget._msd_fig.axes) == 1
        # Step chart has axes drawn (intensity available)
        assert len(widget._step_fig.axes) == 1
        # Intensity chart drawn
        assert len(widget._intensity_fig.axes) == 1
        # Status reflects vertex count (8 for the long track)
        assert "8 vertices" in widget._per_track_status.text()

    def test_short_track_msd_shows_placeholder(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_synthetic_tracks(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._image_combo.setCurrentText("img")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        _, short_id = self._get_track_ids_by_length(viewer)
        widget._on_track_selected(short_id)  # short track (3 frames)

        ax = widget._msd_fig.axes[0]
        # Look for the placeholder text
        texts = [t.get_text() for t in ax.texts]
        assert any("too short" in s for s in texts)

    def test_step_param_change_redraws_step_chart(self, tracking_widget):
        widget, viewer = tracking_widget
        self._add_synthetic_tracks(viewer)
        widget._points_combo.setCurrentText("spots")
        widget._image_combo.setCurrentText("img")
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        long_id, _ = self._get_track_ids_by_length(viewer)
        widget._on_track_selected(long_id)
        # Change a knob; the step chart should redraw without error
        widget._step_window.setValue(10)
        widget._step_threshold.setValue(0.7)
        # Trigger handler explicitly in case tests don't fire signal eagerly
        widget._on_step_params_changed()
        assert len(widget._step_fig.axes) == 1

    def test_no_image_skips_intensity_chart(self, tracking_widget):
        widget, viewer = tracking_widget
        # Add tracks WITHOUT an image layer
        data = np.array([
            [0, 0.0, 0.0], [1, 0.0, 1.0], [2, 0.0, 2.0], [3, 0.0, 3.0],
            [4, 0.0, 4.0], [5, 0.0, 5.0], [6, 0.0, 6.0],
        ])
        viewer.add_points(data, name="spots")
        widget._points_combo.setCurrentText("spots")
        # Leave _image_combo blank
        widget._search_range.setValue(5.0)
        widget._memory.setValue(0)
        widget._track_btn.click()

        # Look up the actual particle ID rather than assuming it's 1
        layer = viewer.layers["Tracked Spots"]
        tracked_df = layer.metadata["tracked_df"]
        any_id = int(tracked_df["particle"].iloc[0])
        widget._on_track_selected(any_id)

        ax = widget._intensity_fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("No intensity data" in s for s in texts)
