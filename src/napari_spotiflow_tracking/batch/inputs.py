"""Expand config inputs (files and folders) into per-movie jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, InputSpec, PipelineConfig

#: Labels become folder names, so they must be valid on Windows too.
_FORBIDDEN = set('<>:"/\\|?*')


@dataclass(frozen=True)
class MovieJob:
    group: str
    movie_id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.group}/{self.movie_id}"


def input_label(spec: InputSpec) -> str:
    if spec.label:
        return spec.label
    path = Path(spec.path)
    # is_file, not suffix: folders like "field_1_20_7.5_l_20" contain dots.
    return path.stem if path.is_file() else path.name


def expand_inputs(cfg: PipelineConfig) -> list:
    jobs, labels, errors = [], set(), []
    for index, spec in enumerate(cfg.inputs):
        where = f"inputs[{index}]"
        path = Path(spec.path)
        label = input_label(spec)
        if _FORBIDDEN & set(label):
            errors.append(f"{where}: label '{label}' cannot contain any of {''.join(sorted(_FORBIDDEN))}")
            continue
        if label in labels:
            errors.append(f"{where}: label '{label}' is used by an earlier input; "
                          "give each input a unique label")
            continue
        labels.add(label)
        if path.is_file():
            movies, base = [path], path.parent
        elif path.is_dir():
            found = path.rglob(spec.pattern) if spec.recursive else path.glob(spec.pattern)
            movies, base = sorted(p for p in found if p.is_file()), path
            if not movies:
                errors.append(f"{where}: no files match '{spec.pattern}' in {path}")
        else:
            errors.append(f"{where}: {path} does not exist")
            continue
        for movie in movies:
            movie_id = "__".join(movie.relative_to(base).with_suffix("").parts)
            jobs.append(MovieJob(group=label, movie_id=movie_id, path=movie))
    if errors:
        raise ConfigError("invalid inputs:\n  - " + "\n  - ".join(errors))
    return jobs


def color_pins(cfg: PipelineConfig) -> dict:
    return {input_label(spec): spec.color for spec in cfg.inputs if spec.color}
