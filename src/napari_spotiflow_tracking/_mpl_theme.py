"""Shared napari dark-theme styling for embedded matplotlib figures."""

from __future__ import annotations

from matplotlib.figure import Figure

NAPARI_BG = "#262930"
NAPARI_FG = "#d6d6d6"
LINE_COLOR = "#e0e0e0"


def style_figure(fig: Figure) -> None:
    """Apply napari dark-theme colors to a matplotlib Figure and its axes."""
    fig.patch.set_facecolor(NAPARI_BG)
    for ax in fig.axes:
        ax.set_facecolor(NAPARI_BG)
        for spine in ax.spines.values():
            spine.set_color(NAPARI_FG)
        ax.tick_params(colors=NAPARI_FG)
        ax.xaxis.label.set_color(NAPARI_FG)
        ax.yaxis.label.set_color(NAPARI_FG)
        ax.title.set_color(NAPARI_FG)
        for text in ax.texts:
            text.set_color(NAPARI_FG)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(NAPARI_BG)
            legend.get_frame().set_edgecolor(NAPARI_FG)
            for text in legend.get_texts():
                text.set_color(NAPARI_FG)
