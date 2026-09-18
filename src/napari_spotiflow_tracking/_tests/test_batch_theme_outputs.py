"""Theme palettes/style and output selection/saving."""
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from skimage.color import deltaE_ciede2000, rgb2lab

from napari_spotiflow_tracking.batch import outputs, theme
from napari_spotiflow_tracking.batch.config import Calibration, ConfigError, ConfigWarning
from napari_spotiflow_tracking.batch.outputs import OutputSelector, OutputSpec

# Machado et al. (2009) full-severity simulation matrices, applied in linear RGB.
_CVD = {
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def _lab(color, mode):
    rgb = np.array(to_rgb(color))
    if mode != "normal":
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        sim = np.clip(lin @ _CVD[mode].T, 0, 1)
        rgb = np.where(sim <= 0.0031308, 12.92 * sim, 1.055 * sim ** (1 / 2.4) - 0.055)
    return rgb2lab(rgb.reshape(1, 1, 3))[0, 0]


def _min_delta(pairs):
    return min(
        deltaE_ciede2000(_lab(a, m), _lab(b, m))
        for m in ("normal", "deutan", "protan", "tritan")
        for a, b in pairs
    )


def test_semantic_palette_separates_under_cvd():
    colors = [theme.STEP_COLORS[1], theme.STEP_COLORS[2], theme.STEP_COLORS[3], theme.NEUTRAL]
    assert _min_delta(itertools.combinations(colors, 2)) >= 20


def test_semantic_palettes_share_colors():
    assert theme.MOTILITY_COLORS["constrained"] == theme.STEP_COLORS[1]
    assert theme.MOTILITY_COLORS["diffusive"] == theme.STEP_COLORS[2]
    assert theme.MOTILITY_COLORS["directed"] == theme.STEP_COLORS[3]
    assert theme.PAPA_COLORS["group1"] == theme.STEP_COLORS[1]
    assert theme.PAPA_COLORS["group2"] == theme.STEP_COLORS[2]


def test_group_palette_neighbours_and_first_four_separate():
    g = theme.GROUP_COLORS
    assert len(g) == 8
    assert _min_delta(zip(g, g[1:])) >= 15
    assert _min_delta(itertools.combinations(g[:4], 2)) >= 12


def test_group_colors_follow_input_order_and_pins():
    colors = theme.group_colors(["A", "B", "C"], pinned={"B": "#000000", "C": None})
    assert colors == {"A": theme.GROUP_COLORS[0], "B": "#000000", "C": theme.GROUP_COLORS[2]}


def test_style_does_not_leak():
    before = dict(matplotlib.rcParams)
    with theme.style():
        assert matplotlib.rcParams["axes.spines.top"] is False
    assert dict(matplotlib.rcParams) == before


def test_axis_labels_follow_calibration():
    px = Calibration()
    um = Calibration(pixel_size_nm=100, frame_interval_s=0.1)
    assert theme.axis_label("time", px) == "Time (frames)"
    assert theme.axis_label("time", um) == "Time (s)"
    assert theme.axis_label("length", um) == "Distance (µm)"
    assert theme.axis_label("msd", px) == "MSD (px²)"
    assert theme.axis_label("intensity", um) == "Intensity (a.u.)"
    with pytest.raises(ValueError):
        theme.axis_label("colour", px)


CATALOGUE = [
    OutputSpec("traces", "movie", "figure", "Traces"),
    OutputSpec("sheet", "movie", "figure", "Sheet"),
    OutputSpec("steps", "movie", "table", "Steps"),
]


def test_selector_all_none_and_lists():
    sel = OutputSelector({}, CATALOGUE, "demo")
    assert sel.figure("traces") and sel.figure("sheet") and sel.table("steps")
    sel = OutputSelector({"figures": "none", "tables": ["steps"]}, CATALOGUE, "demo")
    assert not sel.figure("traces") and sel.table("steps")
    sel = OutputSelector({"figures": "sheet"}, CATALOGUE, "demo")
    assert sel.figure("sheet") and not sel.figure("traces")


def test_selector_warns_on_unknown_name():
    with pytest.warns(ConfigWarning, match="analyses.demo.figures: unknown output 'trace'"):
        sel = OutputSelector({"figures": ["trace", "sheet"]}, CATALOGUE, "demo")
    assert sel.figure("sheet") and not sel.figure("traces")


def test_selector_rejects_bad_type():
    with pytest.raises(ConfigError, match="analyses.demo.tables"):
        OutputSelector({"tables": 3}, CATALOGUE, "demo")


def test_save_figure_writes_each_format(tmp_path):
    fig = Figure()
    fig.subplots().plot([0, 1], [0, 1])
    written = outputs.save_figure(fig, tmp_path / "sub" / "plot", ["png", "svg"], dpi=50)
    assert [p.name for p in written] == ["plot.png", "plot.svg"]
    assert all(p.exists() for p in written)
    assert not list((tmp_path / "sub").glob("*.tmp.*"))


def test_save_figure_falls_back_when_target_locked(tmp_path, monkeypatch):
    real_replace = os.replace

    def locked(src, dst):
        if str(dst).endswith("plot.png"):
            raise OSError(22, "Invalid argument")
        return real_replace(src, dst)

    monkeypatch.setattr(outputs.os, "replace", locked)
    written = outputs.save_figure(Figure(), tmp_path / "plot", ["png"], dpi=50)
    assert [p.name for p in written] == ["plot_new.png"]


def test_save_table_creates_parent(tmp_path):
    path = outputs.save_table(pd.DataFrame({"a": [1, 2]}), tmp_path / "x" / "t.csv")
    assert pd.read_csv(path)["a"].tolist() == [1, 2]
