"""Pure-Python helpers for the PAPA widget.

No Qt imports; functions are unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_particles_per_frame(points: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Convert a Points layer (N, 3) array into a per-frame count DataFrame.

    Input frames are 0-indexed (napari convention); output frames are 1-indexed
    to match the canonical analysis in papa_analysis.py.

    Returns
    -------
    per_frame : DataFrame with columns ``frame`` (1-indexed int) and
        ``n_particles`` (int). Reindexed so every frame in [1, max_frame] is
        present (zero-detection frames included with ``n_particles=0``).
    n_negative_clipped : number of input rows that had negative frame indices
        (clipped to 0 before the +1 shift). Caller may surface this as a
        warning.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with columns (frame, y, x)")

    raw_frames = np.rint(points[:, 0]).astype(int)
    n_negative_clipped = int((raw_frames < 0).sum())
    frames_0idx = np.clip(raw_frames, 0, None)
    frames_1idx = frames_0idx + 1

    if frames_1idx.size == 0:
        per_frame = pd.DataFrame({"frame": [], "n_particles": []}).astype(int)
        return per_frame, n_negative_clipped

    counts = (
        pd.Series(frames_1idx)
        .value_counts()
        .sort_index()
    )
    max_frame = int(frames_1idx.max())
    full_index = pd.Index(range(1, max_frame + 1), name="frame")
    counts = counts.reindex(full_index, fill_value=0).astype(int)

    per_frame = counts.reset_index()
    per_frame.columns = ["frame", "n_particles"]
    per_frame["frame"] = per_frame["frame"].astype(int)
    return per_frame, n_negative_clipped


def _iter_blocks(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Walk the raw acquisition blocks from frame 1.

    Block *k* (0-indexed) holds ``group1_size`` frames when *k* is even and
    ``group2_size`` when *k* is odd; its group number is ``(k % 2) + 1``. The
    next block starts ``skip_frames`` frames after this one ends.

    Only blocks that fit entirely within ``max_time`` are returned — a block
    that would overrun ends the walk, since nothing after it can fit either.
    """
    sizes = (group1_size, group2_size)
    blocks: list[dict] = []
    current = 1
    k = 0
    while current <= max_time:
        end = current + sizes[k % 2] - 1
        if end > max_time:
            break
        blocks.append({
            "group": (k % 2) + 1,
            "start_frame": current,
            "end_frame": end,
        })
        current = end + 1 + skip_frames
        k += 1
    return blocks


def build_cycle_info(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Generate cycle/group boundary information. Cycles and frames are 1-indexed.

    The first ``group1_size`` frames are the *lead-in*: readout that precedes
    the first reactivation pulse, so it belongs to no cycle (see
    :func:`lead_in_span`). Cycle 1 starts at the block after the lead-in, which
    makes a cycle ``(group 2 block, group 1 block)`` in frame order — each
    group-2 readout is paired with the group-1 readout that *follows* it.

    Group identity follows the block alternation from frame 1 and never moves:
    a ``group1_size``-sized block is always group 1, a ``group2_size``-sized
    block always group 2.

    Only complete cycles are returned — both blocks must fit entirely within
    ``max_time``, so the stack must be at least
    ``group1_size + skip_frames + group2_size + skip_frames + group1_size``
    frames long: the excluded lead-in plus one cycle. Nothing is clipped and
    no group is ever truncated.

    Entries are in ascending frame order: the group-2 entry then the group-1
    entry for each cycle.
    """
    if group1_size < 1 or group2_size < 1:
        raise ValueError("group1_size and group2_size must be >= 1")
    if skip_frames < 0:
        raise ValueError("skip_frames must be >= 0")

    # Drop block 0 — the lead-in — then take what remains two blocks at a time.
    # An odd trailing block is an incomplete cycle and is discarded.
    after_lead_in = _iter_blocks(
        max_time, group1_size, group2_size, skip_frames,
    )[1:]

    cycle_info: list[dict] = []
    for cycle_num, i in enumerate(range(0, len(after_lead_in) - 1, 2), start=1):
        for block in (after_lead_in[i], after_lead_in[i + 1]):
            cycle_info.append({
                "cycle": cycle_num,
                "group": block["group"],
                "start_frame": block["start_frame"],
                "end_frame": block["end_frame"],
            })
    return cycle_info


def lead_in_span(max_time: int, group1_size: int) -> tuple[int, int] | None:
    """Inclusive 1-indexed frame range of the excluded lead-in block.

    Returns ``None`` when the stack has no frames. The end is clamped to
    ``max_time`` so a stack shorter than ``group1_size`` still shades correctly.
    """
    if group1_size < 1:
        raise ValueError("group1_size must be >= 1")
    if max_time < 1:
        return None
    return (1, min(group1_size, max_time))


def annotate_with_cycle_group(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
) -> pd.DataFrame:
    """Add ``cycle`` and ``group`` columns (nullable Int64) to ``per_frame``.

    Frames falling in skip-gap regions receive ``NaN`` in both columns, as do
    frames in the excluded lead-in (see :func:`lead_in_span`), since neither
    belongs to any cycle. Returns a new DataFrame; does not mutate the input.
    """
    out = per_frame.copy()
    cycle_col = pd.array([pd.NA] * len(out), dtype="Int64")
    group_col = pd.array([pd.NA] * len(out), dtype="Int64")

    frame_to_idx = {int(f): i for i, f in enumerate(out["frame"].tolist())}
    for info in cycle_info:
        for f in range(info["start_frame"], info["end_frame"] + 1):
            i = frame_to_idx.get(f)
            if i is not None:
                cycle_col[i] = info["cycle"]
                group_col[i] = info["group"]

    out["cycle"] = cycle_col
    out["group"] = group_col
    return out


def compute_per_cycle_summary(
    per_frame: pd.DataFrame,
    cycle_info: list[dict],
    label1: str = "G1",
    label2: str = "G2",
) -> pd.DataFrame:
    """Aggregate ``per_frame`` counts to per-cycle totals and ratios.

    Returns a DataFrame with one row per cycle and columns:
      ``cycle``, ``<label1>_count``, ``<label2>_count``,
      ``<label1>_over_<label2>``, ``<label2>_over_<label1>``.

    Ratios use NaN when the denominator is zero. This is the data behind the
    bar/ratio subplots in :class:`PAPAWidget`'s plots tab — exposed here so it
    can be exported as well as rendered.
    """
    cycles = sorted({info["cycle"] for info in cycle_info})
    columns = [
        "cycle",
        f"{label1}_count",
        f"{label2}_count",
        f"{label1}_over_{label2}",
        f"{label2}_over_{label1}",
    ]
    rows = []
    for cyc in cycles:
        g1_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 1]
        g2_info = [i for i in cycle_info if i["cycle"] == cyc and i["group"] == 2]

        g1_total = 0
        if g1_info:
            mask = (
                (per_frame["frame"] >= g1_info[0]["start_frame"])
                & (per_frame["frame"] <= g1_info[0]["end_frame"])
            )
            g1_total = int(per_frame.loc[mask, "n_particles"].sum())

        g2_total = 0
        if g2_info:
            mask = (
                (per_frame["frame"] >= g2_info[0]["start_frame"])
                & (per_frame["frame"] <= g2_info[0]["end_frame"])
            )
            g2_total = int(per_frame.loc[mask, "n_particles"].sum())

        rows.append({
            "cycle": int(cyc),
            f"{label1}_count": g1_total,
            f"{label2}_count": g2_total,
            f"{label1}_over_{label2}":
                g1_total / g2_total if g2_total > 0 else float("nan"),
            f"{label2}_over_{label1}":
                g2_total / g1_total if g1_total > 0 else float("nan"),
        })

    # Pass `columns` explicitly so a zero-cycle result is still an empty frame
    # with the right columns rather than a shapeless one the plots can't index.
    return pd.DataFrame(rows, columns=columns)
