"""The msd analysis: per-movie tables and figures, group summaries, comparison.

Ported from scripts/bulk_track_analysis.py: run_movie from analyze_folder,
summarize_group from write_run_summary, compare_groups from
write_group_comparison.
"""

from __future__ import annotations

from ...outputs import OutputSpec, save_table
from ..base import Analysis, measure_on_for
from . import compute, figures_movie
from .units import Units

CATALOGUE = (
    OutputSpec("msd_curves", "movie", "table", "Per-track MSD per lag, with SEM and pair count"),
    OutputSpec("msd_fits", "movie", "table", "D, alpha and motility class per track"),
    OutputSpec("track_analysis", "movie", "table", "Per-track geometry, intensity, D, alpha, class"),
    OutputSpec("msd_combined", "movie", "figure", "Per-track MSD curves by class, with ensemble fit"),
    OutputSpec("msd_regimes", "movie", "figure", "Normalized MSD curves over the alpha regime wedges"),
    OutputSpec("motility_classes", "movie", "figure", "Track counts per motility class"),
    OutputSpec("d_vs_alpha", "movie", "figure", "Joint distribution of D and alpha, with marginals"),
)


class MsdAnalysis(Analysis):
    name = "msd"
    requires_tracks = True
    default_measure_on = "preprocessed"
    catalogue = CATALOGUE

    def default_settings(self) -> dict:
        return {"max_lag": 25, "sigma_nm": 35.0, "min_length": 5,
                "max_tracks_plot": 200, "figures": "all", "tables": "all"}

    def validate(self, settings: dict) -> list:
        problems = []
        if settings["max_lag"] < 1:
            problems.append("max_lag must be at least 1")
        if settings["min_length"] < 3:
            problems.append("min_length must be at least 3")
        if settings["max_tracks_plot"] < 1:
            problems.append("max_tracks_plot must be at least 1")
        if settings["sigma_nm"] < 0:
            problems.append("sigma_nm must not be negative")
        return problems

    def run_movie(self, ctx, settings: dict, select) -> dict:
        units = Units(ctx.config.calibration, settings["sigma_nm"])
        tracked = ctx.tracks
        curves, fits = compute.compute_msd_analysis(
            tracked, units, max_lag=settings["max_lag"],
            min_length=settings["min_length"],
        )
        counts = compute.class_counts(fits)
        # Fixed keys: class_counts adds "unclassified" only when non-empty, and
        # run_summary.csv columns must not come and go between movies.
        record = {"n_fitted": int(len(fits)),
                  "constrained": counts.get("constrained", 0),
                  "diffusive": counts.get("diffusive", 0),
                  "directed": counts.get("directed", 0),
                  "unclassified": counts.get("unclassified", 0)}

        # ctx.analysis_dir creates the folder, so only touch it when something
        # will actually be written (a tables:none, figures:none run must leave
        # no empty msd/ folder).
        wanted = any(
            select.table(spec.name) if spec.kind == "table" else select.figure(spec.name)
            for spec in self.catalogue
        )
        if not wanted:
            return record

        out_dir = ctx.analysis_dir(self.name)
        for name, frame in (("msd_curves", curves), ("msd_fits", fits)):
            if select.table(name):
                save_table(frame, out_dir / f"{name}.csv")

        # track_analysis is the only output that needs the measured image and
        # the per-track geometry table — skip building both on a figures-only
        # (or msd_curves/msd_fits-only) run.
        if select.table("track_analysis"):
            measure_on = measure_on_for(self, settings, ctx.config)
            image = ctx.image(measure_on) if measure_on else None
            table = compute.track_table(tracked, image, fits, units)
            save_table(table, out_dir / "track_analysis.csv")

        formats, dpi = ctx.config.outputs.formats, ctx.config.outputs.dpi
        if select.figure("msd_combined"):
            figures_movie.plot_msd_combined(
                curves, fits, units, out_dir / "msd_combined", formats, dpi)
        if select.figure("msd_regimes"):
            figures_movie.plot_msd_regimes(
                curves, fits, units, out_dir / "msd_regimes", formats, dpi)
        if select.figure("motility_classes"):
            figures_movie.plot_motility_classes(
                fits, out_dir / "motility_classes", formats, dpi)
        if select.figure("d_vs_alpha"):
            figures_movie.plot_d_vs_alpha(
                fits, units, out_dir / "d_vs_alpha", formats, dpi)

        return record
