"""Run a validated config over every movie, then group and comparison hooks."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure  # no pyplot: never switch the backend napari uses

from .._preprocessing import _get_device
from . import theme
from .analyses import enabled_analyses
from .config import PipelineConfig, write_resolved_config
from .context import MovieContext
from .detect import detect_stack, fit_spots, load_detector, track_spots
from .inputs import MovieJob, color_pins, expand_inputs
from .outputs import OutputSelector, save_figure, save_table
from .preprocess import run_steps
from .validation import validate

log = logging.getLogger("napari_spotiflow_tracking.batch")

STATUS_FILE = "movie_status.json"
SUMMARY_COLUMNS = ["group", "movie", "path", "status", "n_frames", "n_spots",
                   "n_tracks", "runtime_s", "error"]


@dataclass
class RunResult:
    rows: list = field(default_factory=list)
    failed_hooks: list = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        failed = self.failed_hooks or any(r["status"] == "FAIL" for r in self.rows)
        return 1 if failed else 0


def movie_dir(output_dir: Path, job: MovieJob) -> Path:
    return Path(output_dir) / job.group / job.movie_id


def _resolve_device(requested: str) -> str:
    device = _get_device(use_gpu=requested == "cuda")
    if device != requested:
        log.warning("device '%s' is not available; using '%s'", requested, device)
    return device


def _build_context(job: MovieJob, cfg: PipelineConfig, out_dir: Path, device: str,
                   detector: dict) -> MovieContext:
    raw = tifffile.imread(str(job.path))
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) movie, got shape {raw.shape}")
    pre = run_steps(raw, cfg.preprocessing.steps, device)
    if cfg.outputs.save_preprocessed:
        tifffile.imwrite(str(out_dir / "preprocessed.tif"), pre, bigtiff=True)

    spots_csv, tracks_csv = out_dir / "spots.csv", out_dir / "tracks.csv"
    reusable = spots_csv.exists() and (not cfg.tracking.enabled or tracks_csv.exists())
    mask = None
    if cfg.skip_existing and reusable:
        log.info("%s: reusing spots/tracks on disk", job.key)
        spots = pd.read_csv(spots_csv)
        tracks = pd.read_csv(tracks_csv) if cfg.tracking.enabled else None
    else:
        if "model" not in detector:
            detector["model"] = load_detector(cfg.detection, device)
        spots = detect_stack(pre, cfg.detection, detector["model"])
        if cfg.detection.fit.enabled:
            spots, mask = fit_spots(pre, spots, cfg.detection.fit, device)
            if cfg.outputs.save_masks:
                tifffile.imwrite(str(out_dir / "mask.tif"), mask, bigtiff=True)
        save_table(spots, spots_csv)
        tracks = None
        if cfg.tracking.enabled:
            tracks = track_spots(spots, cfg.tracking)
            save_table(tracks, tracks_csv)
    return MovieContext(job=job, out_dir=out_dir, config=cfg, raw=raw, preprocessed=pre,
                        spots=spots, tracks=tracks, mask=mask)


def _write_status(status_path: Path, row: dict) -> None:
    """Write ``row`` as JSON atomically (temp file in the same folder + ``os.replace``)."""
    tmp = status_path.with_name(f"{status_path.stem}.tmp{status_path.suffix}")
    tmp.write_text(json.dumps(row, indent=2), encoding="utf-8")
    os.replace(tmp, status_path)


def _run_movie(job: MovieJob, cfg: PipelineConfig, analyses: list, device: str,
               detector: dict) -> dict:
    out_dir = movie_dir(Path(cfg.output_dir), job)
    status_path = out_dir / STATUS_FILE
    if cfg.skip_existing and status_path.exists():
        try:
            row = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("%s: could not read %s (%s: %s); reprocessing", job.key,
                       STATUS_FILE, type(exc).__name__, exc)
        else:
            row["status"] = "SKIP"
            log.info("%s: already finished, skipped", job.key)
            return row

    start = time.perf_counter()
    row = {"group": job.group, "movie": job.movie_id, "path": str(job.path),
           "status": "OK", "n_frames": 0, "n_spots": 0, "n_tracks": 0,
           "runtime_s": 0.0, "error": ""}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = _build_context(job, cfg, out_dir, device, detector)
        row["n_frames"] = int(ctx.raw.shape[0])
        row["n_spots"] = int(len(ctx.spots))
        if ctx.tracks is not None:
            row["n_tracks"] = int(ctx.tracks["particle"].nunique())
        for analysis, settings in analyses:
            select = OutputSelector(settings, analysis.catalogue, analysis.name)
            record = analysis.run_movie(ctx, settings, select) or {}
            for key, value in record.items():
                row[f"{analysis.name}.{key}"] = value
    except Exception as exc:
        log.error("%s failed:\n%s", job.key, traceback.format_exc())
        row["status"] = "FAIL"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_s"] = round(time.perf_counter() - start, 2)
    if row["status"] == "OK":
        try:
            _write_status(status_path, row)
        except Exception as exc:
            log.error("%s: failed to write %s:\n%s", job.key, STATUS_FILE,
                     traceback.format_exc())
            row["status"] = "FAIL"
            row["error"] = f"{type(exc).__name__}: {exc}"
    log.info("%s: %s (%d spots, %d tracks, %.1f s)", job.key, row["status"],
             row["n_spots"], row["n_tracks"], row["runtime_s"])
    return row


def _run_hooks(cfg: PipelineConfig, analyses: list, jobs: list, rows: list) -> list:
    out = Path(cfg.output_dir)
    done = {(r["group"], r["movie"]) for r in rows if r["status"] in ("OK", "SKIP")}
    groups: dict = {}
    for job in jobs:
        if (job.group, job.movie_id) in done:
            groups.setdefault(job.group, []).append(movie_dir(out, job))
    failed = []
    for analysis, settings in analyses:
        select = OutputSelector(settings, analysis.catalogue, analysis.name)
        for group, dirs in groups.items():
            try:
                analysis.summarize_group(group, dirs, settings, select,
                                         out / group / "_summary" / analysis.name)
            except Exception:
                label = f"{analysis.name}: group summary for '{group}'"
                log.error("%s failed:\n%s", label, traceback.format_exc())
                failed.append(label)
        if len(groups) > 1:
            try:
                analysis.compare_groups(groups, settings, select,
                                        out / "_comparison" / analysis.name)
            except Exception:
                label = f"{analysis.name}: group comparison"
                log.error("%s failed:\n%s", label, traceback.format_exc())
                failed.append(label)
    return failed


def _write_summary(cfg: PipelineConfig, rows: list) -> None:
    out = Path(cfg.output_dir)
    table = pd.DataFrame(rows)
    extra = [c for c in table.columns if c not in SUMMARY_COLUMNS]
    save_table(table.reindex(columns=SUMMARY_COLUMNS + extra), out / "run_summary.csv")

    labels = list(dict.fromkeys(r["group"] for r in rows))
    colors = theme.group_colors(labels, color_pins(cfg))
    metrics = ["n_spots", "n_tracks"] if cfg.tracking.enabled else ["n_spots"]
    names = [f"{r['group']}/{r['movie']}" for r in rows]
    with theme.style():
        fig = Figure(figsize=(max(6.0, 0.6 * len(rows) + 2), 3.5 * len(metrics)))
        axes = fig.subplots(len(metrics), 1, squeeze=False, sharex=True)
        for ax, metric in zip(axes[:, 0], metrics):
            ax.bar(np.arange(len(rows)), [r[metric] for r in rows],
                   color=[colors[r["group"]] for r in rows])
            ax.set_ylabel("Spots" if metric == "n_spots" else "Tracks")
        axes[-1, 0].set_xticks(np.arange(len(rows)))
        axes[-1, 0].set_xticklabels(names, rotation=45, ha="right")
        fig.tight_layout()
        save_figure(fig, out / "run_overview", cfg.outputs.formats, cfg.outputs.dpi)


def run(cfg: PipelineConfig) -> RunResult:
    validate(cfg)
    jobs = expand_inputs(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        write_resolved_config(cfg, out / "resolved_config.yaml")
        analyses = enabled_analyses(cfg)
        device = _resolve_device(cfg.device)
        log.info("%d movies in %d groups; analyses: %s", len(jobs),
                 len({j.group for j in jobs}), ", ".join(a.name for a, _ in analyses) or "none")
        detector: dict = {}
        rows = [_run_movie(job, cfg, analyses, device, detector) for job in jobs]
        failed = _run_hooks(cfg, analyses, jobs, rows)
        _write_summary(cfg, rows)
        return RunResult(rows=rows, failed_hooks=failed)
    finally:
        log.removeHandler(handler)
        handler.close()
