# Unified batch pipeline — design

**Date:** 2026-09-17
**Status:** approved in brainstorming, awaiting spec review

## Problem

Batch processing is split across three independent scripts:

- `scripts/bulk_track_analysis.py` (3121 lines) — preprocessing, detection,
  tracking, MSD, multi-folder summary, group comparison.
- `scripts/papa_analysis.py` (734 lines) — one file at a time, positional
  group sizes, its own detection settings, no preprocessing, no tracking.
- `scripts/bleaching_trace_analysis.py` (329 lines) — its own preprocessing,
  tracking and photobleaching-step screening.

Each script re-implements loading, preprocessing and detection, and each styles
its own figures. Consequences: a user cannot enable tracking for PAPA or pick
GUI preprocessing operations in batch; one input layout per script; figure
styles and colours differ between techniques.

## Goal

One batch pipeline, inside the package, driven by a YAML config:

- the modality (tracking/MSD, PAPA, photobleaching) is a preset of defaults,
  not a separate program; analyses can be combined in one run;
- every Pre-Processing widget operation is available as an ordered step list;
- detection and tracking run once per movie and are shared by all analyses;
  tracking can be enabled for any modality;
- inputs are single files or folders, several at once, each entry a group,
  with per-group summaries and cross-group comparisons;
- the user chooses which figures and tables are written;
- one plotting theme and palette across every output.

## Non-goals

- Replacing or modifying the existing scripts. `bulk_track_analysis.py`,
  `papa_analysis.py`, `bleaching_trace_analysis.py`, the `*_complete.py`
  copies and `_build_bulk_complete.py` stay untouched and keep working.
- A standalone (plugin-less) build of the new pipeline.
- Parallel / distributed execution.
- Per-track PAPA analysis (tracks are only made available to other analyses).

## Decisions

| Topic | Decision |
|---|---|
| Location | Subpackage `napari_spotiflow_tracking.batch`, console command `spotiflow-batch` |
| Config format | YAML, plus `--set key.path=value` CLI overrides |
| Preprocessing | Ordered list of steps mirroring the GUI chain |
| Intensity sampling | `measure_on: raw \| preprocessed`, global default with per-analysis override |
| Modality | Preset of defaults; analyses are enable-able modules, combinable |
| Inputs | List of entries (file or folder); each entry is a group |
| Output selection | Per-analysis catalogue of named figures/tables; user lists names |
| Unknown config keys / output names | Warning, not error |
| Existing scripts | Left untouched |
| Internals | Analysis plugins + dataclass config (no new framework) |

## Architecture

```
src/napari_spotiflow_tracking/batch/
  __init__.py
  cli.py          spotiflow-batch run | validate | list-outputs | init-config <modality>
  config.py       dataclasses, YAML load, preset merge, --set overrides,
                  validation, resolved-config dump
  presets/        tracking.yaml, papa.yaml, bleaching.yaml
  inputs.py       expand inputs → list[MovieJob(group, label, path)]
  stages.py       load → preprocess → detect → track (calls existing logic modules)
  context.py      MovieContext: raw, preprocessed, spots, tracks, out_dir; disk cache
  runner.py       per-movie loop, error isolation, skip_existing, run summary,
                  then group and comparison hooks
  theme.py        matplotlib style context + palettes
  outputs.py      save figures/tables (formats, dpi, lock-safe rename), catalogue checks
  analyses/
    __init__.py   registry
    base.py       Analysis protocol
    bleaching.py  from scripts/bleaching_trace_analysis.py
    papa.py       from scripts/papa_analysis.py + _papa_analysis.py
    msd.py        from scripts/bulk_track_analysis.py
```

Logic modules (`_preprocessing`, `_segmentation`, `_fitting`, `_tracking`,
`_track_properties`, `_papa_analysis`) are reused unchanged; they stay Qt-free.
Plotting code that today lives only in the scripts moves into the analysis
modules (the scripts keep their own copies — accepted duplication).

`pyproject.toml` gains `pyyaml` as an explicit dependency (napari brings it,
but the CLI must not rely on napari) and a `[project.scripts]` entry
`spotiflow-batch = "napari_spotiflow_tracking.batch.cli:main"`. Code must run on
Python ≥ 3.9 (`from __future__ import annotations`, no `match`).

### Analysis protocol

```python
class Analysis(Protocol):
    name: str                                  # "msd", "papa", "bleaching"
    def default_settings(self) -> dict: ...        # merged under the user's section; "enabled: false" disables
    requires_tracks: bool
    default_measure_on: str                    # "raw" | "preprocessed" | "none"
    catalogue: list[OutputSpec]                # name, level, kind, description

    def validate(self, settings: dict) -> list[str]: ...          # problems, as strings
    def run_movie(self, ctx: MovieContext, settings: dict, select: OutputSelector) -> dict: ...  # summary record
    def summarize_group(self, group: str, movie_dirs: list[Path], settings: dict,
                        select: OutputSelector, out_dir: Path, cfg: PipelineConfig) -> None: ...
    def compare_groups(self, groups: dict[str, list[Path]], settings: dict,
                       select: OutputSelector, out_dir: Path, cfg: PipelineConfig) -> None: ...
```

`settings` is the analysis section merged over `default_settings()` (with
`enabled` removed); `select` answers which catalogue figures/tables the user
asked for. Group and comparison hooks also get the whole `cfg` for
`outputs.formats`/`dpi`, `calibration` and group colours; groups whose movies
all failed are left out (with a logged warning), so hooks take colours from
`theme.group_colors([input_label(s) for s in cfg.inputs], color_pins(cfg))`
to keep palette slots stable.

`measure_on_for(analysis, settings, cfg)` resolves the movie an analysis
samples: `None` when `default_measure_on == "none"`, else
`analyses.<name>.measure_on` when set, else the top-level `measure_on`.
Validation rejects an `analyses.<name>.measure_on` other than raw/preprocessed.

`OutputSpec(name, level: movie|group|comparison, kind: figure|table, description)`.
Group and comparison hooks read the per-movie tables from disk, so they also
work after `skip_existing` reuse.

### Data flow

1. Load config (defaults < preset < user YAML < `--set`), validate fully —
   including dependencies and output names — before reading any movie.
2. Expand inputs into movie jobs.
3. Per movie: load → preprocessing steps → detection → tracking (if enabled).
   `spots.csv` / `tracks.csv` written to the movie folder and reused by
   `skip_existing`.
4. Per movie, per enabled analysis: `run_movie` writes requested outputs and
   returns a summary record.
5. After all movies: `summarize_group` per group; `compare_groups` when there
   is more than one group.
6. Write `run_summary.csv`, run overview figure, `resolved_config.yaml`.

## Config schema

Precedence: built-in defaults < `presets/<modality>.yaml` < user YAML <
`--set` overrides. Unknown keys produce a warning naming the key path.

```yaml
modality: tracking            # tracking | papa | bleaching

inputs:
  - {path: D:/Data/control, label: Control, pattern: "*.tif", recursive: false, color: null}
  - {path: D:/Data/one_movie.tif}          # label defaults to the file/folder name
output_dir: D:/Results/run1
skip_existing: false
device: cuda                  # cuda | cpu; cuda unavailable → cpu with a warning

calibration:                  # optional; both set → physical units
  pixel_size_nm: null
  frame_interval_s: null

preprocessing:
  steps:                      # applied in order; [] = raw
    - crop: {y: [0, 263], x: [0, 263]}
    - walking_average: {window: 4}
    - subtract: {value: mean}          # add | subtract; number or "mean"
    - background: {method: high_pass, sigma: 10}   # or {method: erosion, disk_size: 5}
    - n2v: {n_epochs: 10, patch_size: 64}
    - gaussian_blur: {sigma: 1.0}

detection:
  method: spotiflow           # spotiflow | log
  spotiflow: {model: general, prob_thresh: 0.5, min_distance: 2}   # model: name or folder
  log: {min_sigma: 2, max_sigma: 10, num_sigma: 10, threshold: 0.09}
  fit: {enabled: false, window: 2, max_radius: 2.0}   # needs torchimize

tracking:
  enabled: true
  search_range: 3
  memory: 10
  min_track_length: 5

measure_on: raw

outputs:
  formats: [png, svg]
  dpi: 300
  save_preprocessed: false
  save_masks: false

analyses:
  msd: {...}
  papa: {...}
  bleaching: {...}
```

Each preprocessing step name maps to one existing function in
`_preprocessing.py` (`crop_image`, `walking_average`, `apply_calculation`,
`remove_background_high_pass_stack` / `remove_background_erosion`,
`denoise_n2v`, `gaussian_blur`). Step parameters mirror those functions'
arguments; the exact parameter names are fixed in the phase-1 plan by reading
the function signatures and the Pre-Processing widget.

### Presets

| Preset | Tracking | Analyses enabled | Other defaults |
|---|---|---|---|
| `tracking` | on | `msd` | bulk script defaults: walking average 4 → subtract mean → blur σ 1; search 5, memory 3 |
| `papa` | off | `papa` | no preprocessing; `papa.group1_size` / `group2_size` required (no default) |
| `bleaching` | on | `bleaching` | subtract mean → blur σ 1; search 3, memory 10 |

### Validation

- `msd` or `bleaching` enabled with `tracking.enabled: false` → error at start-up.
- `papa` enabled without `group1_size` / `group2_size` → error.
- Input path missing, or a folder matching no files → error for that entry.
- Unknown keys, unknown output names → warning.
- `--set` values parsed as YAML scalars/lists (`true`, `3`, `0.4`, `[png, svg]`).

`resolved_config.yaml` in the output folder holds every value actually used.
`spotiflow-batch init-config <modality>` prints a commented starter config.

### Analysis: msd

Ported from `bulk_track_analysis.py`. Config keys: `max_lag` (25), `sigma_nm`
(35), `figures`, `tables`. Its catalogue lists the figures that script writes
today (`msd_combined`, `d_vs_alpha`, `motility_classes`, `msd_regimes`,
`msd_archetypes`, …) and tables (`msd_fits`, `track_properties`, …); the full
list is taken from the script when the phase-4 plan is written. The script's
current defaults become the `tracking` preset so MSD figures match.

### Analysis: papa

Ported from `papa_analysis.py` using `_papa_analysis.py`. Config keys:
`group1_size`, `group2_size` (required), `skip_frames` (1),
`ratio_frame_fraction` (1.0), `labels` ([DR, PAPA]), `spot_radius` (1),
`figures`, `tables`. Counts come from detected spots; tracks, when enabled, are
not used by PAPA.

### Analysis: bleaching

Ported from `scripts/bleaching_trace_analysis.py`.

**Building the trace**

| Param | Default | Meaning |
|---|---|---|
| `roi_size` | 3 | Side (px) of the square around the spot whose mean is the trace (Dey et al. 2024, SI 3). Must be odd. |
| `measure_on` | raw | Movie the intensities come from. Raw because uint16 mean subtraction clips background to 0. |
| `min_track_length` | 10 | Shorter tracks are skipped. |
| `post_frames` | 40 | Frames read at the spot's last position after the track ends; brings the final bleach into the trace and gives the background level and noise. |
| `lead_trim` | 2 | Samples dropped from the start of each trace (a track's first frame is often dim). |

**Fitting steps** (`_track_properties.find_steps`, every combination tried; per
track and step count the best-scoring passing fit is kept)

| Param | Default | Meaning |
|---|---|---|
| `windows` | [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20] | Gaussian smoothing widths (frames). Small: close steps; large: small steps in noisy traces. |
| `thresholds` | [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75] | Minimum jump relative to the trace's largest jump. |

**Acceptance** (all must hold)

| Param | Default | Rule |
|---|---|---|
| — | — | Every step is downward. |
| `final_level_tol` | 1.0 | Last level within this many background-noise SDs of the background. |
| `end_tol` | 10 | Last step within this many frames of the track end. |
| `min_dwell` | 5 | Every level lasts at least this many frames. |
| `min_step_snr` | 2.0 | Every step ≥ this × background noise. |

**Ranking and picking**

| Param | Default | Meaning |
|---|---|---|
| (score) | — | (smallest step / noise) / (largest step / smallest step). |
| `unit_range` | {1: [0.5, 1.6], 2: [0.4, 2.5], 3: [0.4, 2.5]} | Allowed step size as multiples of the movie's unit step (median step of its 2- and 3-step candidates); automatic picks only. |
| `n_examples` | 2 | Examples per step count per movie = number of example figures. Higher step counts pick first; a particle is used for one class only. |
| `picks` | {} | Manual overrides `{"<group>/<movie>": {<n_steps>: [particle, ...]}}`; bypass `unit_range`. |

Catalogue: `representative_traces` (figure, movie), `candidate_sheets`
(figure, movie), `candidates` (table, movie), `steps` (table, movie),
`traces` (table, movie), `step_count_distribution` (figure, group and
comparison), `candidates_pooled` (table, group).

Tuning guide (for docs): too few candidates → lower `min_step_snr` /
`min_dwell` or raise `final_level_tol`; noisy picks → raise `min_step_snr` or
tighten `unit_range`; monomers that look like multi-dye drops → lower the upper
bound of `unit_range[1]`.

## Outputs

```
<output_dir>/
  resolved_config.yaml
  run_summary.csv        group, file, status, n_spots, n_tracks, runtime_s, error
  run_overview.png       spots/tracks per movie, coloured by group
  run.log
  <group>/<movie>/
    spots.csv, tracks.csv
    preprocessed.tif, mask.tif      (only if requested)
    <analysis>/                     tables + figures
  <group>/_summary/<analysis>/
  _comparison/<analysis>/           only when > 1 group
```

- Per analysis: `figures` and `tables` accept `all`, `none` or a list of names.
- `spotiflow-batch list-outputs [analysis]` prints each catalogue with
  descriptions.
- Figures are saved in every format in `outputs.formats` at `outputs.dpi`,
  through a temp file + rename; if the target is locked the file is written as
  `<name>_new.<ext>` with a warning.

## Theme

`theme.py` provides:

- `style()` — a context manager applying one rcParams set (font sizes, spines,
  ticks, line widths); used around every figure so napari's own plots are
  unaffected.
- Group palette — fixed categorical order, assigned in `inputs` order;
  `inputs[].color` pins a colour.
- Semantic palettes — step counts, motility classes and PAPA groups share
  #56b4e9 / #eda100 / #4a3aa7 (+ #7f7f7f unclassified); tab:orange/tab:green
  are not used because they collapse under protanopia.
- Helpers — axis labels with units from `calibration`, shared-y figure grids.

A unit test checks the palettes for colour-vision-deficiency separation.

## Error handling

- A failing movie: traceback to `run.log`, `FAIL` + message in
  `run_summary.csv`, excluded from summaries; other movies continue.
- A failing group/comparison hook: logged, run continues.
- Exit code non-zero if any movie or hook failed; 2 for config errors.
- `validate` subcommand runs steps 1–2 of the data flow only.
- `skip_existing`: a movie with `movie_status.json` is skipped; otherwise
  existing `spots.csv`/`tracks.csv` are reused and analyses re-run.

## Testing

All fast tests run without GPU or real data (synthetic stacks, fake model).

- `config`: precedence; `--set` parsing; unknown-key warnings; dependency
  errors; resolved config round-trips.
- `inputs`: files, folders, patterns, recursion, empty folder, duplicate labels.
- `stages`: each preprocessing step equals the direct logic-function call;
  detection with a fake model; tracking on hand-built spots.
- analyses: synthetic movies with planted answers — bleaching recovers planted
  1/2/3-step traces; PAPA counts match planted cycles; MSD recovers planted D;
  only requested outputs are written.
- `runner`: failure isolation, `skip_existing`, exit codes, summary/comparison
  only when applicable.
- `theme`: palette check; no global rcParams leakage.
- `@pytest.mark.slow` regression: old `bulk_track_analysis.py` vs new `msd` on
  one small movie, fit tables compared numerically.

## Build order

Each phase gets its own implementation plan and ends usable.

1. **Core** — config, presets, inputs, stages (all preprocessing steps, both
   detectors, optional fitting/masks, tracking), runner, outputs, theme; CLI
   `run`, `validate`, `list-outputs`, `init-config`. Usable for detect + track.
2. **bleaching** analysis.
3. **papa** analysis.
4. **msd** analysis + slow regression test.
5. Group summaries and cross-group comparisons for all analyses; MkDocs page
   (config reference, presets, output catalogues, bleaching parameter guide).
