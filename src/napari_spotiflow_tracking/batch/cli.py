"""``spotiflow-batch``: run, validate, list-outputs, init-config."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .analyses import base
from .config import MODALITIES, ConfigError, config_to_dict, load_config
from .inputs import expand_inputs
from .overrides import validate_all
from .runner import run
from .validation import validate

SECTION_HELP = {
    "skip_existing": "Skip movies that already finished in output_dir (resume an interrupted run).",
    "device": "cuda or cpu; cuda falls back to cpu when no GPU is available.",
    "calibration": "Set both to report lengths in µm and times in s.",
    "preprocessing": "Steps run top to bottom. Available: crop, add, subtract, "
                     "walking_average, gaussian_blur, background, n2v.",
    "detection": "method: spotiflow or log. fit.enabled (on by default) adds a 2D "
                 "Gaussian fit and spot masks: torchimize when available, else the "
                 "scipy fallback.",
    "tracking": "trackpy linking; every linked track is written, and each analysis "
                "applies its own minimum length.",
    "measure_on": "raw or preprocessed: the movie analyses read intensities from. "
                  "Unset (null) means each analysis decides for itself; a value "
                  "here overrides every analysis that sets no measure_on of its own.",
    "outputs": "formats: any of png, pdf, svg.",
    "analyses": "Enabled analyses and their settings; `spotiflow-batch list-outputs` "
                "lists figure and table names.",
}


def starter_config(modality: str) -> str:
    resolved = config_to_dict(load_config(data={"modality": modality}))
    lines = [
        f"# spotiflow-batch config — {modality} preset",
        "# Check: spotiflow-batch validate this.yaml   Run: spotiflow-batch run this.yaml",
        "",
        f"modality: {modality}",
        "",
        "# One entry per group: a folder of movies or a single movie.",
        "inputs:",
        "  - {path: 'D:/path/to/movies', label: Control, pattern: '*.tif', recursive: false}",
        "output_dir: 'D:/path/to/results'",
        "",
    ]
    for key, value in resolved.items():
        if key in ("modality", "inputs", "output_dir"):
            continue
        if key in SECTION_HELP:
            lines.append(f"# {SECTION_HELP[key]}")
        lines.append(yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True).rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def format_catalogues(name: Optional[str] = None) -> str:
    if name is not None:
        analysis = base.get_analysis(name)
        if analysis is None:
            raise ConfigError(f"unknown analysis '{name}'; known: "
                              f"{', '.join(sorted(base.REGISTRY)) or 'none'}")
        chosen = [analysis]
    else:
        chosen = [base.REGISTRY[key] for key in sorted(base.REGISTRY)]
    if not chosen:
        return "No analyses are registered yet."
    blocks = []
    for analysis in chosen:
        lines = [f"{analysis.name}:"]
        for spec in analysis.catalogue:
            lines.append(f"  {spec.name:<28} {spec.kind:<6} {spec.level:<10} {spec.description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spotiflow-batch",
        description="Config-driven batch detection, tracking and analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "run a config"), ("validate", "check a config and its inputs")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("config", type=Path)
        cmd.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="KEY=VALUE",
                         help="override a config value, e.g. --set tracking.memory=5")
    cmd = sub.add_parser("list-outputs", help="list the figures and tables analyses can write")
    cmd.add_argument("analysis", nargs="?")
    cmd = sub.add_parser("init-config", help="print a commented starter config")
    cmd.add_argument("modality", choices=MODALITIES)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.formatwarning = lambda message, *a, **k: f"WARNING: {message}\n"
    try:
        if args.command == "init-config":
            sys.stdout.write(starter_config(args.modality))
            return 0
        if args.command == "list-outputs":
            print(format_catalogues(args.analysis))
            return 0
        cfg = load_config(args.config, args.overrides)
        if args.command == "validate":
            validate(cfg)
            jobs = expand_inputs(cfg)
            validate_all(cfg, jobs)
            groups = {job.group for job in jobs}
            print(f"config OK: {len(jobs)} movie(s) in {len(groups)} group(s)")
            return 0
        return run(cfg).exit_code
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
