"""Read a tracks CSV back into the arrays a napari Tracks layer wants.

Pure logic — no Qt, no napari. The npe2 reader in ``_reader.py`` is a thin
shell over this, and the tests exercise it directly.

Two flavours of file are accepted, because the plugin writes both:

* the Spot Tracking widget's "Export Tracks to CSV" (``frame,y,x,particle``,
  plus ``intensity`` when an image layer was selected), and
* ``tracks.csv`` from ``scripts/bulk_track_analysis.py``, which carries the
  Gaussian-fit and intensity columns as well.

``track_id`` is accepted in place of ``particle`` so a table exported from the
analysis side round-trips too.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from napari_spotiflow_tracking._track_properties import compute_track_properties

#: Columns every tracks CSV must have, whatever wrote it.
COORD_COLUMNS = ("frame", "y", "x")
#: Accepted spellings of the track identifier, in preference order.
TRACK_ID_ALIASES = ("particle", "track_id")
#: Per-spot columns usable as the per-vertex intensity trace, in preference
#: order. The widget's own export writes ``intensity``; the bulk script writes
#: ``mean_intensity`` (the fitted spot's mean) instead.
INTENSITY_ALIASES = ("intensity", "mean_intensity", "max_intensity")
#: Written beside ``tracks.csv`` by the bulk script — one row per track, with
#: D / alpha / motility_class already computed.
PROPERTIES_SIBLING = "track_analysis.csv"


@dataclass
class TracksFile:
    """Everything a Tracks layer needs, plus what had to be discarded."""

    data: np.ndarray            # (N, 4) [track_id, frame, y, x]
    tracked_df: pd.DataFrame    # per-vertex rows, 'particle' normalized
    properties: pd.DataFrame    # one row per track
    n_tracks: int
    n_dropped: int              # rows discarded for a missing coordinate
    properties_from: str | None  # sibling CSV the properties came from, if any


def track_id_column(columns) -> str | None:
    """Return which of :data:`TRACK_ID_ALIASES` ``columns`` uses, if any."""
    for name in TRACK_ID_ALIASES:
        if name in columns:
            return name
    return None


def read_header(path) -> list[str]:
    """Column names of ``path``, or [] if it cannot be read as a CSV.

    Reads a single row, so this stays cheap enough to run on every CSV the
    reader is offered — including multi-million-row track tables.
    """
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def is_tracks_csv(path) -> bool:
    """True when ``path`` has the columns needed to build a Tracks layer.

    The reader declines everything else, so napari falls through to its own
    CSV reader for points and shapes tables.
    """
    cols = read_header(path)
    if not cols:
        return False
    return (all(c in cols for c in COORD_COLUMNS)
            and track_id_column(cols) is not None)


def layer_name_for(path) -> str:
    """Name the layer after its folder when the file is a generic tracks.csv.

    The bulk script writes one ``tracks.csv`` per stack into a folder named
    after that stack, so the folder is the informative half of the path.
    """
    path = Path(path)
    if path.stem.lower() == "tracks" and path.parent.name:
        return path.parent.name
    return path.stem


def read_tracks_csv(path) -> TracksFile:
    """Load ``path`` into Tracks-layer form.

    Raises ``ValueError`` when a required column is missing or no row survives
    coordinate cleaning — the reader turns that into a napari error rather
    than adding an empty layer.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in COORD_COLUMNS if c not in df.columns]
    id_col = track_id_column(df.columns)
    if id_col is None:
        missing.append("particle (or track_id)")
    if missing:
        raise ValueError(
            f"{path.name} is not a tracks CSV — missing column(s): "
            f"{', '.join(missing)}"
        )

    if id_col != "particle":
        df = df.rename(columns={id_col: "particle"})

    key = ["particle", *COORD_COLUMNS]
    for col in key:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    # A row missing any coordinate cannot be placed. The bulk script's
    # tracks.csv can carry NaNs in the fit columns, which stay untouched.
    df = df.dropna(subset=key)
    n_dropped = n_before - len(df)
    if df.empty:
        raise ValueError(f"{path.name} has no row with a complete "
                         f"{'/'.join(key)}")

    df["particle"] = df["particle"].astype(int)
    df["frame"] = df["frame"].astype(int)
    # napari wants each track's vertices in time order, tracks grouped.
    df = df.sort_values(["particle", "frame"]).reset_index(drop=True)

    if "intensity" not in df.columns:
        for alias in INTENSITY_ALIASES:
            if alias in df.columns:
                # The per-track intensity chart reads 'intensity'; point it at
                # whichever per-spot column this file happens to carry.
                df["intensity"] = pd.to_numeric(df[alias], errors="coerce")
                break

    properties, properties_from = _load_properties(path, df)
    return TracksFile(
        data=df[["particle", "frame", "y", "x"]].to_numpy(dtype=float),
        tracked_df=df,
        properties=properties,
        n_tracks=int(df["particle"].nunique()),
        n_dropped=int(n_dropped),
        properties_from=properties_from,
    )


def _load_properties(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Per-track properties: the sibling table if there is one, else computed.

    ``track_analysis.csv`` already holds D, alpha and motility_class, and
    reusing it avoids a per-track pass over a table that can run to millions
    of rows.
    """
    sibling = path.parent / PROPERTIES_SIBLING
    if sibling.is_file():
        try:
            props = pd.read_csv(sibling)
            if "track_id" in props.columns:
                return props, sibling.name
        except Exception:
            pass  # unreadable or not the table we expect — compute instead
    return compute_track_properties(df, image=None), None
