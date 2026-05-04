"""Tests for PAPAWidget."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def papa_widget(make_napari_viewer):
    from napari_spotiflow_tracking._papa_widget import PAPAWidget

    viewer = make_napari_viewer()
    widget = PAPAWidget(viewer)
    return widget, viewer


class TestPAPAWidgetConstruction:
    def test_creates_without_error(self, papa_widget):
        widget, _ = papa_widget
        assert widget is not None

    def test_has_input_and_plots_tabs(self, papa_widget):
        widget, _ = papa_widget
        assert widget._tabs.count() == 2
        assert widget._tabs.tabText(0) == "Input"
        assert widget._tabs.tabText(1) == "Plots"

    def test_default_field_values(self, papa_widget):
        widget, _ = papa_widget
        assert widget._group1_size.text() == "5"
        assert widget._group2_size.text() == "5"
        assert widget._skip_frames.text() == "1"
        assert widget._label_group1.text() == "DR"
        assert widget._label_group2.text() == "PAPA"

    def test_export_csv_starts_disabled(self, papa_widget):
        widget, _ = papa_widget
        assert widget._export_btn.isEnabled() is False

    def test_initial_status_message(self, papa_widget):
        widget, _ = papa_widget
        assert "Ready" in widget._status_label.text()
