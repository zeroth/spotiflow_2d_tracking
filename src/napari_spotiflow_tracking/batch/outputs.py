"""Output catalogues, user selection, and lock-safe saving."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .config import ConfigError, ConfigWarning

log = logging.getLogger("napari_spotiflow_tracking.batch")


@dataclass(frozen=True)
class OutputSpec:
    """One figure or table an analysis can write."""

    name: str
    level: str  # "movie" | "group" | "comparison"
    kind: str  # "figure" | "table"
    description: str


class OutputSelector:
    """Which catalogue outputs the user asked for (``figures``/``tables`` keys)."""

    def __init__(self, settings: Mapping, catalogue: Sequence[OutputSpec], analysis: str):
        self._chosen = {
            kind: self._resolve(settings.get(f"{kind}s", "all"), kind, catalogue, analysis)
            for kind in ("figure", "table")
        }

    @staticmethod
    def _resolve(value, kind: str, catalogue, analysis: str) -> set:
        names = {spec.name for spec in catalogue if spec.kind == kind}
        key = f"analyses.{analysis}.{kind}s"
        if value is None or value == "none":
            return set()
        if value == "all":
            return names
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be 'all', 'none' or a list of names")
        for name in value:
            if name not in names:
                warnings.warn(
                    f"{key}: unknown output '{name}' is ignored; "
                    f"available: {', '.join(sorted(names)) or 'none'}",
                    ConfigWarning,
                    stacklevel=3,
                )
        return {name for name in value if name in names}

    def figure(self, name: str) -> bool:
        return name in self._chosen["figure"]

    def table(self, name: str) -> bool:
        return name in self._chosen["table"]


def _replace(tmp: Path, target: Path) -> Path:
    """Move ``tmp`` onto ``target``; if the target is locked, beside it as ``*_new``."""
    try:
        os.replace(tmp, target)
        return target
    except OSError:
        fallback = target.with_name(f"{target.stem}_new{target.suffix}")
        os.replace(tmp, fallback)
        log.warning("%s is locked (open in a viewer?); wrote %s instead",
                    target, fallback.name)
        return fallback


def save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> list:
    """Save ``fig`` as ``<stem>.<ext>`` for every format."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = stem.with_name(f"{stem.name}.{ext}")
        tmp = stem.with_name(f"{stem.name}.tmp.{ext}")
        fig.savefig(tmp, dpi=dpi)
        written.append(_replace(tmp, target))
    return written


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(tmp, index=False)
    return _replace(tmp, path)
