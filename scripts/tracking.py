import numpy as np
import trackpy as tp
import pandas as pd
from pathlib import Path

def track_blobs(blobs_df, search_range=2, memory=3):
    """
    Track blobs over time using trackpy.

    Parameters:
    - blobs_df: DataFrame with columns ['time', 'y', 'x', 'sigma'] representing detected blobs.
    - search_range: Maximum distance a particle can move between frames.
    - memory: Number of frames a particle can vanish for before being considered lost.

    Returns:
    - tracked_df: DataFrame with tracking results including particle IDs.
    """
    tp.quiet()  # Suppress trackpy output messages
    # Rename columns to match trackpy expectations
    blobs_df = blobs_df.rename(columns={'y': 'y', 'x': 'x', 'time': 'frame'})
    
    # Perform tracking
    tracked_df = tp.link_df(blobs_df, search_range=search_range, memory=memory)
    
    # Filter out short-lived trajectories
    # tracked_df = tp.filter_stubs(tracked_df, threshold=2)
    
    return tracked_df

def cycle_tracking(blobs_df, group_size, search_range=2, memory=3):
    """
    Track particles in cycles. Each cycle consists of a group of
    consecutive frames to track, followed by one event frame to skip.

    Parameters:
    - blobs_df: DataFrame with columns ['time', 'y', 'x', ...].
    - group_size: Number of frames in each tracking group.
    - search_range: Maximum distance a particle can move between frames.
    - memory: Number of frames a particle can vanish for.

    Returns:
    - summary_df: DataFrame with group number, frame range, and particle count.
    - tracked_groups: list of tracked DataFrames per group.
    """
    max_time = int(blobs_df['time'].max())

    summary = []
    tracked_groups = []
    group_num = 0
    start = 1  # 1-indexed

    while start <= max_time:
        group_num += 1
        end = min(start + group_size - 1, max_time)

        group_df = blobs_df[(blobs_df['time'] >= start) & (blobs_df['time'] <= end)].copy()

        n_particles = 0
        tracked = pd.DataFrame()
        if len(group_df) > 0:
            tracked = track_blobs(group_df, search_range=search_range, memory=memory)
            n_particles = tracked['particle'].nunique()

        summary.append({
            'group': group_num,
            'start_frame': start,
            'end_frame': end,
            'n_particles': n_particles,
        })
        tracked_groups.append(tracked)

        # advance past the event frame
        start = end + 2

    summary_df = pd.DataFrame(summary)
    return summary_df, tracked_groups

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Track blobs in a DataFrame.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file with detected blobs.")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file with tracked blobs.")
    parser.add_argument("--search_range", type=float, default=2, help="Maximum distance a particle can move between frames.")
    parser.add_argument("--memory", type=int, default=3, help="Number of frames a particle can vanish for before being considered lost.")
    parser.add_argument("--group_size", type=int, default=None, help="Enable cycle tracking: track groups of this many frames, skipping 1 event frame between groups.")
    args = parser.parse_args()

    # Load detected blobs
    blobs_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(blobs_df)} blobs from {args.input_csv}")

    if args.group_size is not None:
        # Cycle tracking mode
        summary_df, tracked_groups = cycle_tracking(
            blobs_df, group_size=args.group_size,
            search_range=args.search_range, memory=args.memory,
        )
        print(summary_df.to_string(index=False))

        summary_path = str(Path(args.output_csv).with_suffix('')) + "_cycle_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved cycle summary to {summary_path}")
    else:
        # Normal tracking
        tracked_df = track_blobs(blobs_df, search_range=args.search_range, memory=args.memory)
        tracked_df.sort_values(by=['particle'], inplace=True)
        print(f"Tracked blobs. Resulting DataFrame has {len(tracked_df)} entries.")

        tracked_df.to_csv(args.output_csv, index=False)
        print(f"Saved tracked blobs to {args.output_csv}")

        # tracks_max_intensity = tracked_df.groupby('particle')['mean_intensity'].max().reset_index()
        # tracks_max_intensity.columns = ['particle', 'max_intensity']
        # tracks_max_intensity.to_csv(Path(args.output_csv).parent.joinpath(Path(args.output_csv).stem + "_tracks_max_intensity.csv"), index=False)
        # print(f"Saved tracks maximum intensity to {Path(args.output_csv).stem}_tracks_max_intensity.csv")
