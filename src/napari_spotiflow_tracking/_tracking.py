import numpy as np
import pandas as pd
import trackpy as tp


def track_blobs(
    blobs_df: pd.DataFrame,
    search_range: float = 2,
    memory: int = 3,
) -> pd.DataFrame:
    """Link detected spots across frames using trackpy.

    Args:
        blobs_df: DataFrame with columns 'frame', 'y', 'x'.
        search_range: max distance a particle can move between frames.
        memory: frames a particle can vanish before being lost.

    Returns:
        DataFrame with added 'particle' column containing integer track IDs.
    """
    tp.quiet()
    return tp.link_df(blobs_df, search_range=search_range, memory=memory)


def tracks_to_napari(tracked_df: pd.DataFrame) -> np.ndarray:
    """Convert a tracked DataFrame to the array format for napari Tracks layer.

    Returns (N, 4) array with columns [track_id, frame, y, x].
    """
    return tracked_df[["particle", "frame", "y", "x"]].to_numpy()
