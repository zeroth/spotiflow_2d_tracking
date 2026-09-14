"""npe2 reader contribution: open a tracks CSV as a napari Tracks layer.

Registered in ``napari.yaml`` against ``*.csv``, so File → Open, drag-and-drop
and ``napari path/to/tracks.csv`` all work. ``get_reader`` declines any CSV
without the track columns, leaving napari's builtin reader to handle points
and shapes tables as before.

All parsing lives in :mod:`napari_spotiflow_tracking._tracks_io`; this module
only shapes the result into LayerData tuples.
"""
from __future__ import annotations

from napari_spotiflow_tracking._tracks_io import (
    is_tracks_csv,
    layer_name_for,
    read_tracks_csv,
)


def get_reader(path):
    """Return :func:`read_tracks` if ``path`` is a tracks CSV, else None."""
    # A list means napari is offering several paths at once; this reader
    # handles one file at a time and lets napari fan out.
    if not isinstance(path, str):
        return None
    if not path.lower().endswith(".csv"):
        return None
    if not is_tracks_csv(path):
        return None
    return read_tracks


def read_tracks(path):
    """Read ``path`` into a single Tracks LayerData tuple."""
    result = read_tracks_csv(path)
    # The same metadata keys the Spot Tracking widget attaches after a live
    # run, so its Analysis tab and filter panel work on an opened file with no
    # special-casing.
    attrs = {
        "name": layer_name_for(path),
        "metadata": {
            "properties": result.properties,
            "tracked_df": result.tracked_df,
            "source_csv": str(path),
            "n_dropped_rows": result.n_dropped,
        },
    }
    return [(result.data, attrs, "tracks")]
