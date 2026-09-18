"""The msd analysis: per-movie tables and figures, group summaries, comparison.

Ported from scripts/bulk_track_analysis.py: run_movie from analyze_folder,
summarize_group from write_run_summary, compare_groups from
write_group_comparison.
"""

from __future__ import annotations

from ... import theme
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
    OutputSpec("length_hist", "movie", "figure", "Track duration (length) distribution"),
    OutputSpec("path_length_hist", "movie", "figure", "Distribution of total path length walked per track"),
    OutputSpec("speed_hist", "movie", "figure", "Distribution of per-track mean speed"),
    OutputSpec("length_per_track", "movie", "figure", "Per-track cumulative displacement vs elapsed time"),
    OutputSpec("speed_per_track", "movie", "figure", "Per-track frame-to-frame speed vs elapsed time"),
    OutputSpec("directed_displacement", "movie", "figure", "Run-length distribution of directed tracks"),
    OutputSpec("directed_velocity", "movie", "figure", "Velocity distribution of directed tracks"),
    OutputSpec("directed_displacement_vs_velocity", "movie", "figure",
               "Run length vs velocity, directed tracks only, with marginals"),
)

_TABLE_FIGURES = (
    "length_hist", "path_length_hist", "speed_hist",
    "length_per_track", "speed_per_track",
    "directed_displacement", "directed_velocity",
    "directed_displacement_vs_velocity",
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

        # track_analysis is the only table output that needs the measured
        # image and the per-track geometry table, but the distribution and
        # directed-motion figures need that same table too — so build it
        # whenever the table itself or any figure that reads it is selected,
        # and skip building it entirely on a run that wants neither (a
        # figures-only run of msd_combined/msd_regimes/etc. must not build it).
        needs_table = select.table("track_analysis") or any(
            select.figure(name) for name in _TABLE_FIGURES
        )
        table = None
        if needs_table:
            measure_on = measure_on_for(self, settings, ctx.config)
            image = ctx.image(measure_on) if measure_on else None
            table = compute.track_table(tracked, image, fits, units)
            if select.table("track_analysis"):
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

        if table is not None:
            plot_table = table[table["length"] >= settings["min_length"]]
            if select.figure("length_hist"):
                figures_movie.plot_length_hist(
                    plot_table, units, settings["min_length"],
                    out_dir / "length_hist", formats, dpi)
            if select.figure("path_length_hist"):
                figures_movie.plot_path_length_hist(
                    plot_table, units, out_dir / "path_length_hist", formats, dpi)
            if select.figure("speed_hist"):
                figures_movie.plot_speed_hist(
                    plot_table, units, out_dir / "speed_hist", formats, dpi)

            if select.figure("length_per_track") or select.figure("speed_per_track"):
                plot_tracked = tracked[tracked["particle"].isin(plot_table["track_id"])]
                if select.figure("length_per_track"):
                    figures_movie.plot_length_per_track(
                        plot_tracked, units, settings["max_tracks_plot"],
                        out_dir / "length_per_track", formats, dpi)
                if select.figure("speed_per_track"):
                    figures_movie.plot_speed_per_track(
                        plot_tracked, units, settings["max_tracks_plot"],
                        out_dir / "speed_per_track", formats, dpi)

            directed = compute.directed_metrics(table, units)
            if not directed.empty:
                if select.figure("directed_displacement"):
                    figures_movie.plot_directed_hist(
                        directed, units, "_disp",
                        theme.axis_label("displacement", units.calibration),
                        out_dir / "directed_displacement", formats, dpi)
                if select.figure("directed_velocity"):
                    figures_movie.plot_directed_hist(
                        directed, units, "_vel",
                        theme.axis_label("speed", units.calibration),
                        out_dir / "directed_velocity", formats, dpi)
                if select.figure("directed_displacement_vs_velocity"):
                    figures_movie.plot_directed_joint(
                        directed, units,
                        out_dir / "directed_displacement_vs_velocity", formats, dpi)

        return record
