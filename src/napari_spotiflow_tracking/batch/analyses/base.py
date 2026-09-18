"""Analysis plugin base class and registry.

An analysis declares what it needs and what it can write, then implements up
to three hooks. The runner calls ``run_movie`` once per movie, then
``summarize_group`` per group and ``compare_groups`` when there is more than
one group. Group-level hooks read the per-movie outputs from disk, so they
also work for movies skipped with ``skip_existing``. Hooks create their
``out_dir`` only when they write something.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import PipelineConfig, deep_merge


class Analysis:
    name: str = ""
    requires_tracks: bool = False
    default_measure_on: str = "raw"
    catalogue: tuple = ()

    def default_settings(self) -> dict:
        return {}

    def validate(self, settings: dict) -> list:
        """Problems with ``settings`` (already merged over defaults), as strings."""
        return []

    def run_movie(self, ctx, settings: dict, select) -> dict:
        """Write this movie's outputs; return a flat record for run_summary.csv."""
        raise NotImplementedError

    def summarize_group(self, group: str, movie_dirs: list, settings: dict, select,
                        out_dir: Path, cfg: PipelineConfig) -> None:
        """Write group-level outputs from the per-movie outputs in ``movie_dirs``.

        ``cfg`` carries ``outputs.formats``/``dpi``, ``calibration`` and the inputs.
        Only groups with at least one successful movie are summarised.
        """
        return None

    def compare_groups(self, groups: dict, settings: dict, select, out_dir: Path,
                       cfg: PipelineConfig) -> None:
        """Write comparison outputs; ``groups`` maps group label to its movie dirs.

        Groups whose movies all failed are left out of ``groups``. Take colours
        from ``theme.group_colors([input_label(s) for s in cfg.inputs],
        color_pins(cfg))`` so a dropped group does not shift palette slots.
        """
        return None


REGISTRY: dict = {}


def register(analysis: Analysis) -> Analysis:
    if analysis.name in REGISTRY:
        raise ValueError(f"analysis '{analysis.name}' is already registered")
    REGISTRY[analysis.name] = analysis
    return analysis


def get_analysis(name: str) -> Optional[Analysis]:
    return REGISTRY.get(name)


def settings_for(analysis: Analysis, section) -> dict:
    return deep_merge(analysis.default_settings(), section or {})


def measure_on_for(analysis: Analysis, settings: dict, cfg: PipelineConfig) -> Optional[str]:
    """The movie an analysis reads intensities from: ``"raw"``, ``"preprocessed"`` or None.

    Resolution order:

    1. ``None`` when ``analysis.default_measure_on == "none"`` (the analysis reads
       no intensities, so any setting is ignored);
    2. ``settings["measure_on"]`` (the ``analyses.<name>.measure_on`` key) when set;
    3. the top-level ``cfg.measure_on`` when set;
    4. otherwise ``analysis.default_measure_on`` — the top-level key is unset by
       default precisely so each analysis keeps the stack its own science needs
       (msd samples the preprocessed movie); a top-level value is a deliberate
       run-wide override, as in the ``bleaching`` preset.
    """
    if analysis.default_measure_on == "none":
        return None
    value = settings.get("measure_on")
    if value is not None:
        return value
    return cfg.measure_on if cfg.measure_on is not None else analysis.default_measure_on


def enabled_analyses(cfg: PipelineConfig) -> list:
    """``(analysis, merged settings)`` for every known, enabled section, in config order."""
    enabled = []
    for name, section in cfg.analyses.items():
        analysis = get_analysis(name)
        if analysis is None or not isinstance(section, (dict, type(None))):
            continue
        settings = settings_for(analysis, section)
        if settings.pop("enabled", True):
            enabled.append((analysis, settings))
    return enabled
