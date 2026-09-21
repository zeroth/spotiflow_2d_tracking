"""Per-group and per-file parameter overrides for an ``InputSpec``.

``InputSpec.settings`` applies to every movie an input entry expands to;
``InputSpec.per_file`` applies exceptions to individual movies within it,
keyed by file name, movie id, or a glob against either. Both are restricted
to the sections a movie may reasonably own -- not the run-wide concerns
(``device``, ``outputs``, ``output_dir``, ``skip_existing``, ``inputs``
itself).

Resolution order, later winning, all merged with ``deep_merge``:

    dataclass defaults < modality preset < global YAML < --set
        < group ``settings`` < matching ``per_file``

``config_for`` resolves this by re-building the whole config from
``config_to_dict`` merged with the movie's override, so type-checking and
unknown-key warnings apply exactly as they do to the global config.

Two entry points, deliberately split by what they need to know:

- ``config_for(job, cfg)`` resolves *one* movie's effective config. Whether
  two ``per_file`` globs are ambiguous for a movie is a property of that
  movie's own matches, so it needs nothing about the rest of the entry.
- ``matches_for(spec, jobs)`` (and ``validate_all``, which calls it for every
  input entry) additionally checks that every ``per_file`` key matched *some*
  movie in the entry -- a check that inherently needs the whole entry's job
  list, which a single ``config_for`` call does not have. Keep it that way
  rather than threading ``jobs`` through ``config_for``: the runner already
  calls ``validate_all`` once, up front, so every movie's ``config_for`` call
  can assume unmatched-key problems were already caught.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional

from .config import ConfigError, InputSpec, PipelineConfig, _build, config_to_dict, deep_merge
from .inputs import MovieJob, input_label

#: The only sections a movie/group may set for itself; everything else
#: describes the run, not a movie (see module docstring).
ALLOWED_SECTIONS = ("preprocessing", "detection", "tracking", "calibration",
                    "measure_on", "analyses")


def validate_override_sections(section: Any, where: str) -> List[str]:
    """Problems with a ``settings``/``per_file`` value's top-level keys, as strings."""
    if not isinstance(section, dict):
        return [f"{where} must be a mapping, got {type(section).__name__}"]
    problems = []
    for key in section:
        if key not in ALLOWED_SECTIONS:
            problems.append(
                f"{where}.{key} is not allowed here; movie/group overrides may only set: "
                f"{', '.join(ALLOWED_SECTIONS)}"
            )
    return problems


def _match_kind(key: str, job: MovieJob) -> Optional[str]:
    """``"exact"``, ``"glob"`` or ``None`` for how (or whether) ``key`` names ``job``."""
    filename = job.path.name
    if key == filename or key == job.movie_id:
        return "exact"
    if fnmatch.fnmatch(filename, key) or fnmatch.fnmatch(job.movie_id, key):
        return "glob"
    return None


def _override_for_job(spec: InputSpec, job: MovieJob) -> dict:
    """The merged ``settings`` + matching ``per_file`` override for one movie.

    Only needs ``job`` itself (not its siblings): whether two ``per_file``
    keys are ambiguous for a movie depends only on that movie's matches.
    Whether a key matches *no* movie needs the whole entry -- see
    :func:`matches_for`.
    """
    merged = dict(spec.settings or {})
    per_file = spec.per_file or {}
    exact = [key for key in per_file if _match_kind(key, job) == "exact"]
    globs = [key for key in per_file if _match_kind(key, job) == "glob"]
    if exact:
        chosen = exact
    elif len(globs) > 1:
        raise ConfigError(
            f"movie '{job.movie_id}' in input '{job.group}' matches multiple per_file "
            f"patterns: {', '.join(sorted(globs))}"
        )
    else:
        chosen = globs
    for key in chosen:
        merged = deep_merge(merged, per_file[key])
    return merged


def matches_for(spec: InputSpec, jobs: List[MovieJob]) -> Dict[str, dict]:
    """The per-movie override dict for one input entry, keyed by ``movie_id``.

    Applies the matching rules (exact file name or movie id beats a glob;
    two globs matching one movie is ambiguous) and raises :class:`ConfigError`
    when a ``per_file`` key matches no movie in this entry.
    """
    label = input_label(spec)
    entry_jobs = [job for job in jobs if job.group == label]
    result = {}
    matched_keys = set()
    for job in entry_jobs:
        result[job.movie_id] = _override_for_job(spec, job)
        for key in (spec.per_file or {}):
            if _match_kind(key, job) is not None:
                matched_keys.add(key)
    unmatched = [key for key in (spec.per_file or {}) if key not in matched_keys]
    if unmatched:
        raise ConfigError(
            f"input '{label}': per_file key(s) match no movie: {', '.join(unmatched)}"
        )
    return result


def validate_all(cfg: PipelineConfig, jobs: List[MovieJob]) -> None:
    """Matching-rule problems (needs the expanded ``jobs``), one report for the run.

    Section-name restrictions are checked in ``validation.validate`` (no jobs
    needed there); this covers what that cannot: ambiguous globs and
    ``per_file`` keys that match nothing.
    """
    errors = []
    for spec in cfg.inputs:
        try:
            matches_for(spec, jobs)
        except ConfigError as exc:
            errors.append(str(exc))
    if errors:
        raise ConfigError("invalid per_file overrides:\n  - " + "\n  - ".join(errors))


def config_for(job: MovieJob, cfg: PipelineConfig) -> PipelineConfig:
    """The movie's effective config: ``cfg`` with its input entry's ``settings``
    and matching ``per_file`` override deep-merged in.

    Rebuilds from the raw dict so type-checking and unknown-key warnings
    apply to the merged result exactly as they do to the global config.
    """
    spec = next((s for s in cfg.inputs if input_label(s) == job.group), None)
    if spec is None:
        return cfg
    override = _override_for_job(spec, job)
    if not override:
        return cfg
    merged = deep_merge(config_to_dict(cfg), override)
    return _build(PipelineConfig, merged, "")
