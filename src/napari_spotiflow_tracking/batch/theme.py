"""One look for every batch figure: rc style, palettes and axis labels.

Palettes were checked with CIEDE2000 under Machado (2009) deutan, protan and
tritan simulation (see test_batch_theme_outputs). matplotlib's tab:orange and
tab:green collapse to ΔE 1.3 under protanopia, so they are not used.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional, Sequence

import matplotlib

from .config import Calibration

#: Categorical hues for groups, assigned in input order. Ordered so every
#: neighbouring pair and the first four stay apart under colour-vision deficiency.
GROUP_COLORS = ("#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6",
                "#e87ba4", "#008300", "#eda100", "#e34948")

_SKY, _AMBER, _INDIGO = "#56b4e9", "#eda100", "#4a3aa7"
NEUTRAL = "#7f7f7f"
#: Monomer / dimer / trimer.
STEP_COLORS = {1: _SKY, 2: _AMBER, 3: _INDIGO}
MOTILITY_COLORS = {"constrained": _SKY, "diffusive": _AMBER,
                   "directed": _INDIGO, "unclassified": NEUTRAL}
PAPA_COLORS = {"group1": _SKY, "group2": _AMBER, "ratio": _INDIGO}
#: Raw data drawn behind a fit.
RAW_TRACE = "#9a9a9a"

# Type sized for a figure printed at column width (bulk_track_analysis.py).
STYLE = {
    "font.size": 17,
    "axes.labelsize": 19,
    "axes.titlesize": 15,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "lines.linewidth": 2.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}


@contextmanager
def style():
    """Apply :data:`STYLE` for the duration of the block only."""
    with matplotlib.rc_context(STYLE):
        yield


def group_colors(
    labels: Sequence[str],
    pinned: Optional[Mapping[str, Optional[str]]] = None,
) -> dict:
    """Colour per group label: the input-order slot, unless pinned in the config."""
    pinned = pinned or {}
    return {
        label: pinned.get(label) or GROUP_COLORS[i % len(GROUP_COLORS)]
        for i, label in enumerate(labels)
    }


def axis_label(kind: str, calibration: Calibration) -> str:
    physical = calibration.is_physical
    labels = {
        "time": "Time (s)" if physical else "Time (frames)",
        "length": "Distance (µm)" if physical else "Distance (px)",
        "msd": "MSD (µm²)" if physical else "MSD (px²)",
        "intensity": "Intensity (a.u.)",
        "speed": "Speed (µm/s)" if physical else "Speed (px/frame)",
        "diffusion": "D (µm²/s^α)" if physical else "D (px²/frame^α)",
        "duration": "Duration (s)" if physical else "Duration (frames)",
        "path_length": "Path length (µm)" if physical else "Path length (px)",
        "displacement": "Net displacement (µm)" if physical else "Net displacement (px)",
        "alpha": "Scaling exponent α",
        "lag": "Lag (s)" if physical else "Lag (frames)",
    }
    if kind not in labels:
        raise ValueError(f"unknown axis kind '{kind}'; choose one of: {', '.join(labels)}")
    return labels[kind]
