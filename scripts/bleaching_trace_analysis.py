"""Photobleaching step analysis from raw movies: pre-process, track, pick examples.

End-to-end pipeline for the GoodDataset movies:

1. Pre-process each raw movie as the plugin does: subtract the scalar stack
   mean (``apply_calculation``), then a per-frame Gaussian blur (sigma 1).
2. Detect spots on the pre-processed movie (Spotiflow "general", prob 0.5,
   min distance 2) and link them with trackpy (search range 3, memory 10).
3. For every track build an intensity trace: the mean of a 3x3-pixel ROI on
   the *raw* movie at the tracked position (the ROI of Dey et al., J. Phys.
   Chem. Lett. 2024, 15, 6292, SI 3), extended by ``POST_FRAMES`` frames at the
   track's last position; the first LEAD_TRIM samples are dropped because a
   track's first frame is often dim. Raw rather than pre-processed intensities because the
   uint16 mean subtraction clips everything below the stack mean to 0, which
   would flatten the post-bleach background. The extension matters: a track
   ends when its spot goes dark, so without it the final bleaching step to
   background falls outside the trace.
4. Fit steps with ``_track_properties.find_steps`` using ONE window and ONE
   threshold for every track of the dataset (``--window`` / ``--threshold``).
   Holding the fit parameters constant across an experiment keeps step counts
   comparable and unbiased: tuning them per trace would let the fit find
   whatever step count looks cleanest. A fit is kept when it looks like clean
   photobleaching: every step downward, the last level at background, the
   last step at the track end, each dwell >= MIN_DWELL frames, each step >=
   MIN_STEP_SNR x background noise. Score = smallest step in noise units
   divided by the largest/smallest step ratio.
5. Draw two figures per movie from the top-ranked monomers, dimers and trimers
   (shared intensity axis within a figure), plus contact sheets of the ranked
   candidates for manual review, and for each picked track a 2 x 5 montage
   of raw-movie crops at 10 evenly spaced trace times.

Usage:
    python scripts/bleaching_trace_analysis.py [DATA_ROOT] [OUT_DIR] [--retrack]
        [--window 2] [--threshold 0.4] [--frame-interval-s 0.02]

Time axes are in seconds when a frame interval is known: ``--frame-interval-s``
if given, otherwise read from a bulk_track_analysis ``track_analysis.csv`` for
the movie (``OUT_DIR/<movie stem>/`` or ``OUT_DIR/``), whose ``duration_s`` is
``(length - 1) * frame_interval_s``. Without either, axes stay in frames. The
step fit itself (window, MIN_DWELL, END_TOL) always works in frames.

Use the same --window/--threshold for every movie of one experiment; they are
recorded in ``analysis_params.json`` in OUT_DIR. DATA_ROOT defaults to D:/Data/Sudipta/Arpan/GoodDataset; OUT_DIR defaults to
DATA_ROOT/bleaching_analysis_preprocessed. Tracks are cached as
``tracks_<movie>.csv`` in OUT_DIR; pass --retrack to redo detection/tracking.
To override the automatic picks, set particle ids in PICKS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tifffile

from napari_spotiflow_tracking._preprocessing import apply_calculation, gaussian_blur
from napari_spotiflow_tracking._track_properties import find_steps
from napari_spotiflow_tracking._tracking import track_blobs

MOVIES = {"field_1": "bleach_fixed_TPD54_JFX650_CELL12_50p_54.6_posXY0_channels_t0_posZ0_center_crop.tif"}

GAUSS_SIGMA = 1.0
PROB_THRESH = 0.5
MIN_DISTANCE = 2
SEARCH_RANGE = 3.0
MEMORY = 10

ROI_HALF = 1  # 3x3 ROI
POST_FRAMES = 40
LEAD_TRIM = 2  # drop a trace's first samples; a track's first frame is often dim
MIN_TRACK_LEN = 10
# find_steps parameters, fixed for the whole dataset (override on the CLI):
# Gaussian-derivative sigma in frames, and the step-detection threshold
# relative to the trace's largest derivative.
WINDOW = 2
THRESHOLD = 0.4
MIN_DWELL = 5
MIN_STEP_SNR = 2.0
FINAL_LEVEL_TOL = 1.0  # last level within this many noise SDs of the background
END_TOL = 10  # last step within this many frames of the track end
N_EXAMPLES = 2
# A picked trace's every step must lie within these multiples of the movie's
# unit step (median step height of its 2- and 3-step candidates), per step
# count. Stops a large single drop -- several dyes bleaching unresolved --
# passing as a monomer; multi-step traces get more room for uneven dyes.
UNIT_RANGE = {1: (0.5, 1.6), 2: (0.4, 2.5), 3: (0.4, 2.5)}

# Image montage of each picked track: MONTAGE_GRABS frames evenly spaced along
# the trace (post-track frames included), in MONTAGE_ROWS rows of
# (2 * MONTAGE_HALF + 1)-pixel crops of the raw movie.
MONTAGE_GRABS = 10
MONTAGE_ROWS = 2
MONTAGE_HALF = 7

KINDS = [("Monomer", 1, "tab:blue"), ("Dimer", 2, "tab:orange"), ("Trimer", 3, "tab:green")]

# Manual override: {(movie, n_steps): [particle, particle]}; empty = top-ranked.
PICKS: dict = {}


# ── detection + tracking ────────────────────────────────────────────


def preprocess(raw: np.ndarray) -> np.ndarray:
    stack_mean = int(round(float(np.mean(raw))))
    return gaussian_blur(apply_calculation(raw, "subtract", stack_mean), GAUSS_SIGMA)


def detect_and_track(raw: np.ndarray, model) -> pd.DataFrame:
    from napari_spotiflow_tracking._segmentation import detect_spots

    pre = preprocess(raw)
    rows = []
    for t in range(pre.shape[0]):
        points, _ = detect_spots(pre[t], model, prob_thresh=PROB_THRESH,
                                 min_distance=MIN_DISTANCE)
        rows.extend((t, y, x) for y, x in np.asarray(points)[:, :2])
    spots = pd.DataFrame(rows, columns=["frame", "y", "x"])
    return track_blobs(spots, search_range=SEARCH_RANGE, memory=MEMORY)


# ── traces + step screening ─────────────────────────────────────────


def trace_positions(g: pd.DataFrame, shape):
    """Movie frame and ROI centre of every trace sample, plus the post-track count.

    The track's frames minus the first LEAD_TRIM, then POST_FRAMES frames at its
    last position.
    """
    r = ROI_HALF
    frames = g["frame"].astype(int).to_numpy()
    ys = np.clip(np.rint(g["y"].to_numpy()).astype(int), r, shape[1] - 1 - r)
    xs = np.clip(np.rint(g["x"].to_numpy()).astype(int), r, shape[2] - 1 - r)
    after = np.arange(frames[-1] + 1, min(frames[-1] + 1 + POST_FRAMES, shape[0]))
    return (np.r_[frames[LEAD_TRIM:], after],
            np.r_[ys[LEAD_TRIM:], np.full(after.size, ys[-1])],
            np.r_[xs[LEAD_TRIM:], np.full(after.size, xs[-1])],
            after.size)


def roi_trace(g: pd.DataFrame, image: np.ndarray):
    """3x3-ROI mean along a track, then POST_FRAMES frames at its last position."""
    r = ROI_HALF
    frames, ys, xs, n_post = trace_positions(g, image.shape)
    trace = np.asarray([image[f, y - r:y + r + 1, x - r:x + r + 1].mean()
                        for f, y, x in zip(frames, ys, xs)], dtype=float)
    return trace, trace[trace.size - n_post:]


def fit_trace(trace: np.ndarray, post: np.ndarray, track_len: int,
              window: int, threshold: float) -> dict | None:
    """Step fit at the dataset's fixed window/threshold; None if not clean bleaching."""
    background, noise = float(np.median(post)), float(post.std())
    table, fit, _ = find_steps(trace, window=window, threshold=threshold)
    if not table or len(table) > 3:
        return None
    heights = -np.array([s[3] for s in table])
    dwells = [s[4] for s in table] + [table[-1][5]]
    if ((heights <= 0).any()
            or abs(table[-1][2] - background) > FINAL_LEVEL_TOL * noise
            or abs(table[-1][0] - track_len) > END_TOL
            or min(dwells) < MIN_DWELL
            or heights.min() < MIN_STEP_SNR * noise):
        return None
    score = heights.min() / noise / (heights.max() / heights.min())
    return {"score": float(score), "fit": fit, "table": table}


def screen(tracks: pd.DataFrame, raw: np.ndarray, movie: str,
           window: int, threshold: float):
    candidates, traces = [], {}
    for pid, g in tracks.groupby("particle"):
        if len(g) < MIN_TRACK_LEN:
            continue
        g = g.sort_values("frame")
        trace, post = roi_trace(g, raw)
        if post.size < 20:
            continue
        track_len = len(g) - LEAD_TRIM
        b = fit_trace(trace, post, track_len, window, threshold)
        if b is None:
            continue
        n = len(b["table"])
        traces[(movie, int(pid), n)] = (trace, b["fit"], b["table"], track_len)
        candidates.append({
            "movie": movie, "particle": int(pid), "n_steps": n, "score": b["score"],
            "window": window, "threshold": threshold, "track_frames": len(g),
            "first_frame": int(g["frame"].iloc[0]) + LEAD_TRIM,
            "y": float(g["y"].iloc[-1]), "x": float(g["x"].iloc[-1]),
            "step_heights": [round(s[3], 1) for s in b["table"]],
        })
    return pd.DataFrame(candidates), traces


# ── frame interval ──────────────────────────────────────────────────


def frame_interval_from_tracking(out: Path, tif: str) -> tuple[float | None, str]:
    """Frame interval recorded by a bulk_track_analysis run, and where it came from.

    That script writes ``duration_s = (length - 1) * frame_interval_s`` into
    ``track_analysis.csv`` when run with a calibration; the interval is that
    ratio. Returns (None, reason) when no calibrated table is found.
    """
    for path in (out / Path(tif).stem / "track_analysis.csv", out / "track_analysis.csv"):
        if not path.exists():
            continue
        props = pd.read_csv(path)
        if "duration_s" not in props:
            continue
        props = props[props["length"] > 1]
        ratios = props["duration_s"] / (props["length"] - 1)
        if ratios.empty:
            continue
        dt = float(ratios.median())
        if not np.allclose(ratios, dt, rtol=1e-6):
            raise ValueError(f"{path}: inconsistent frame interval "
                             f"({ratios.min():g}-{ratios.max():g} s)")
        return dt, str(path)
    return None, "none found (axes in frames)"


def time_label(dt: float | None) -> str:
    return "Time (s)" if dt else "Time (frames)"


def fmt_time(i: int, dt: float | None) -> str:
    return f"{i * dt:.4g} s" if dt else str(i)


# ── plotting ────────────────────────────────────────────────────────


def save_figure(fig, stem: Path, formats=("png", "pdf", "svg"), dpi=300) -> None:
    """Write via a temp file + rename.

    Overwriting in place fails (Errno 22) while Explorer's preview pane or an
    image viewer still maps the old file; a rename replaces it regardless.
    """
    for ext in formats:
        target = stem.with_name(f"{stem.name}.{ext}")
        tmp = stem.with_name(f"{stem.name}.tmp.{ext}")
        fig.savefig(tmp, dpi=dpi)
        try:
            os.replace(tmp, target)
        except OSError:
            fallback = stem.with_name(f"{stem.name}_new.{ext}")
            os.replace(tmp, fallback)
            print(f"  {target.name} is locked; wrote {fallback.name} instead")


def draw_panel(ax, label, k, colour, trace, fit, table, dt=None):
    scale = dt or 1
    t = np.arange(trace.size) * scale
    ax.plot(t, trace, color="0.6", lw=0.8, label="raw intensity", zorder=1)
    ax.plot(t, fit, color=colour, lw=2, label="step fit", zorder=2)
    for step in table:
        ax.axvline(step[0] * scale, color=colour, ls=":", lw=1, alpha=0.8)
    ax.set_title(f"{label} ({k} step{'s' if k > 1 else ''})", style="italic",
                 fontsize=10, color="0.3")
    ax.set_xlabel(time_label(dt), style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def track_montage(raw: np.ndarray, g: pd.DataFrame, colour: str, title: str,
                  stem: Path, dt: float | None = None) -> None:
    """Raw-movie crops around a track at MONTAGE_GRABS evenly spaced trace times.

    Panel titles are trace times, matching the trace figure's x axis. All crops
    share one grey scale so the bleaching steps show as dimming. The ROI box is
    solid while the spot is tracked and dashed after the track ends.
    """
    frames, ys, xs, n_post = trace_positions(g, raw.shape)
    track_len = frames.size - n_post
    idx = np.unique(np.linspace(0, frames.size - 1, MONTAGE_GRABS).round().astype(int))
    h, r = MONTAGE_HALF, ROI_HALF
    crops, centres = [], []
    for i in idx:
        cy = int(np.clip(ys[i], h, raw.shape[1] - 1 - h))
        cx = int(np.clip(xs[i], h, raw.shape[2] - 1 - h))
        crops.append(raw[frames[i], cy - h:cy + h + 1, cx - h:cx + h + 1])
        centres.append((ys[i] - cy + h, xs[i] - cx + h))
    vmin, vmax = np.percentile(np.stack(crops), [1, 99.5])

    ncols = int(np.ceil(MONTAGE_GRABS / MONTAGE_ROWS))
    fig, axs = plt.subplots(MONTAGE_ROWS, ncols, figsize=(1.6 * ncols, 1.9 * MONTAGE_ROWS + 0.5),
                            squeeze=False)
    for ax, i, crop, (ry, rx) in zip(axs.flat, idx, crops, centres):
        ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        tracked = i < track_len
        ax.add_patch(Rectangle((rx - r - 0.5, ry - r - 0.5), 2 * r + 1, 2 * r + 1,
                                   fill=False, ec=colour, lw=1.2,
                                   ls="-" if tracked else "--"))
        ax.set_title(f"t = {fmt_time(i, dt)}", fontsize=9, style="italic",
                     color="0.2" if tracked else "0.5")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.flat[len(idx):]:
        ax.set_axis_off()
    fig.suptitle(f"{title}\nt = 0 is movie frame {frames[0]}; "
                 f"dashed box: after track end", fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def contact_sheet(cands: pd.DataFrame, traces: dict, movie: str, n: int, out: Path) -> None:
    sel = cands[(cands.movie == movie) & (cands.n_steps == n)].head(20)
    if sel.empty:
        return
    nrows = int(np.ceil(len(sel) / 5))
    fig, axs = plt.subplots(nrows, 5, figsize=(22, 3 * nrows), squeeze=False)
    for ax, (rank, r) in zip(axs.flat, enumerate(sel.itertuples())):
        trace, fit, _, track_len = traces[(movie, r.particle, n)]
        ax.plot(trace, color="0.6", lw=0.7)
        ax.plot(fit, lw=2)
        ax.axvline(track_len, color="k", ls=":", lw=0.8)
        ax.set_title(f"#{rank + 1} particle {r.particle} "
                     f"h{r.step_heights} s{r.score:.1f}", fontsize=8)
    for ax in axs.flat[len(sel):]:
        ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, out / f"candidates_{movie}_{n}step", formats=("png",), dpi=60)
    plt.close(fig)


def unit_step(cands: pd.DataFrame) -> float:
    """Median step height of a movie's multi-step candidates."""
    multi = cands[cands.n_steps >= 2].step_heights
    return float(np.median([-x for h in multi for x in h]))


def choose(cands: pd.DataFrame, movie: str) -> dict:
    """Top N_EXAMPLES per step count; a particle is used for one class only.

    Higher step counts pick first: a trace that passes as a clean trimer also
    tends to pass as a dimer by merging two of its steps.
    """
    picks, used = {}, set()
    mine = cands[cands.movie == movie]
    unit = unit_step(mine)
    print(f"{movie}: unit step {unit:.0f}")
    for k in (3, 2, 1):
        ranked = mine[mine.n_steps == k]
        wanted = PICKS.get((movie, k))
        if wanted:
            ranked = ranked.set_index("particle").loc[wanted].reset_index()
        else:
            lo, hi = UNIT_RANGE[k][0] * unit, UNIT_RANGE[k][1] * unit
            ranked = ranked[np.array([all(lo <= -x <= hi for x in h)
                                      for h in ranked.step_heights], dtype=bool)]
        ranked = ranked[~ranked.particle.isin(used)]
        chosen = [ranked.iloc[i] if len(ranked) > i else None for i in range(N_EXAMPLES)]
        used.update(int(c.particle) for c in chosen if c is not None)
        picks[k] = chosen
    return picks


def main(root: Path, out: Path, retrack: bool,
         window: int = WINDOW, threshold: float = THRESHOLD,
         frame_interval_s: float | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Per-movie frame interval: the CLI value for all, else each movie's tracking run.
    intervals = {}
    for movie, tif in MOVIES.items():
        if frame_interval_s is not None:
            intervals[movie] = (frame_interval_s, "--frame-interval-s")
        else:
            intervals[movie] = frame_interval_from_tracking(out, tif)
        print(f"{movie}: frame interval {intervals[movie][0]} s ({intervals[movie][1]})")
    (out / "analysis_params.json").write_text(json.dumps(
        {"window": window, "threshold": threshold, "min_dwell": MIN_DWELL,
         "min_step_snr": MIN_STEP_SNR, "final_level_tol": FINAL_LEVEL_TOL,
         "end_tol": END_TOL, "roi_half": ROI_HALF, "post_frames": POST_FRAMES,
         "lead_trim": LEAD_TRIM, "min_track_len": MIN_TRACK_LEN,
         "frame_interval_s": {m: {"value": v, "source": s}
                              for m, (v, s) in intervals.items()}}, indent=2))
    print(f"find_steps: window {window}, threshold {threshold} (fixed for all tracks)")
    model = None
    all_cands, all_traces, movie_data = [], {}, {}
    for movie, tif in MOVIES.items():
        raw = tifffile.imread(root / tif)
        tracks_csv = out / f"tracks_{movie}.csv"
        if retrack or not tracks_csv.exists():
            if model is None:
                from napari_spotiflow_tracking._segmentation import load_model
                model = load_model("general", device="cuda")
            detect_and_track(raw, model).to_csv(tracks_csv, index=False)
        tracks = pd.read_csv(tracks_csv)
        print(f"{tif}: {raw.shape[0]} frames, {len(tracks)} spots, "
              f"{tracks['particle'].nunique()} tracks")
        cands, traces = screen(tracks, raw, movie, window, threshold)
        all_cands.append(cands)
        all_traces.update(traces)
        movie_data[movie] = (raw, tracks)

    cands = (pd.concat(all_cands, ignore_index=True)
             .sort_values(["movie", "n_steps", "score"], ascending=[True, True, False]))
    cands.to_csv(out / "candidates.csv", index=False)
    print(cands.groupby(["movie", "n_steps"]).size().to_string())
    for movie in MOVIES:
        for n in (1, 2, 3):
            contact_sheet(cands, all_traces, movie, n, out)

    step_rows = []
    for movie, tif in MOVIES.items():
        picks = choose(cands, movie)
        raw, tracks = movie_data[movie]
        dt = intervals[movie][0]
        for i in range(N_EXAMPLES):
            name = f"{movie}_{'AB'[i]}"
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
            for ax, (label, k, colour) in zip(axes, KINDS):
                pick = picks[k][i]
                if pick is None:
                    ax.set_title(f"{label}: no candidate", style="italic", fontsize=10)
                    ax.spines[["top", "right"]].set_visible(False)
                    print(f"  {name}: no {label.lower()} candidate #{i + 1}")
                    continue
                trace, fit, table, track_len = all_traces[(movie, int(pick.particle), k)]
                draw_panel(ax, label, k, colour, trace, fit, table, dt)
                frame_idx = np.arange(trace.size)
                trace_df = pd.DataFrame({"frame_in_trace": frame_idx, "intensity": trace,
                                         "step_fit": fit,
                                         "in_track": frame_idx < track_len})
                if dt:
                    trace_df.insert(1, "time_s", frame_idx * dt)
                trace_df.to_csv(out / f"trace_{name}_{label.lower()}.csv", index=False)
                g = tracks[tracks.particle == pick.particle].sort_values("frame")
                track_montage(raw, g, colour,
                              f"{label} ({k} step{'s' if k > 1 else ''}), "
                              f"particle {int(pick.particle)} - {name}",
                              out / f"montage_{name}_{label.lower()}", dt)
                for j, s in enumerate(table):
                    row = {
                        "figure": name, "panel": label, "movie": tif,
                        "particle": int(pick.particle), "track_frames": track_len,
                        "first_frame": int(pick.first_frame), "window": int(pick.window),
                        "threshold": float(pick.threshold), "score": float(pick.score),
                        "step_index": j, "frame_in_trace": int(s[0]),
                        "level_before": s[1], "level_after": s[2], "step_height": s[3],
                    }
                    if dt:
                        row["time_s"] = int(s[0]) * dt
                        row["frame_interval_s"] = dt
                    step_rows.append(row)
            axes[0].set_ylabel("Intensity (a.u.)", style="italic")
            fig.suptitle(f"Representative single-complex bleaching traces ({tif}; "
                         f"window {window}, threshold {threshold})", fontweight="bold")
            fig.tight_layout()
            save_figure(fig, out / f"bleaching_traces_{name}")
            plt.close(fig)

    pd.DataFrame(step_rows).to_csv(out / "bleaching_traces_steps.csv", index=False)
    print(f"Wrote figures and tables to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photobleaching step analysis.")
    parser.add_argument("data_root", nargs="?", type=Path,
                        default=Path("D:/Data/Sudipta/Arpan/GoodDataset"))
    parser.add_argument("out_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--retrack", action="store_true",
                        help="redo detection/tracking instead of using cached tracks")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"find_steps window, same for every track (default {WINDOW})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"find_steps threshold, same for every track (default {THRESHOLD})")
    parser.add_argument("--frame-interval-s", type=float, default=None,
                        help="seconds per frame for time axes; default: read from the "
                             "movie's track_analysis.csv if present, else plot in frames")
    a = parser.parse_args()
    if a.frame_interval_s is not None and a.frame_interval_s <= 0:
        parser.error("--frame-interval-s must be positive")
    out_dir = a.out_dir or a.data_root / "bleaching_analysis_preprocessed"
    main(a.data_root, out_dir, a.retrack, window=a.window, threshold=a.threshold,
         frame_interval_s=a.frame_interval_s)
