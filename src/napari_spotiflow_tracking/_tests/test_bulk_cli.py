"""Tests for the bulk CLI's argument parsing.

``scripts/bulk_track_analysis.py`` is imported by path (see
``test_group_compare`` for why), so these exercise ``parse_args`` exactly as a
user invoking the script would hit it.

Both positionals are optional, because four of the five modes take fewer than
two: argparse then binds a lone positional to ``input_dir``, which is the wrong
slot for every mode that takes no input directory.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bulk_track_analysis.py"


def _load_script():
    if "bulk_track_analysis" in sys.modules:
        return sys.modules["bulk_track_analysis"]
    spec = importlib.util.spec_from_file_location("bulk_track_analysis", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bulk_track_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


bulk = _load_script()


@pytest.mark.parametrize("argv", [
    ["--in", "stack.tif", "out"],
    ["--summary-only", "out"],
    ["--analysis-only", "out"],
])
def test_lone_positional_is_the_output_dir(argv):
    """Modes that take no input directory still accept an output directory."""
    args = bulk.parse_args(argv)
    assert args.output_dir == Path("out")
    assert args.input_dir is None


def test_directory_mode_keeps_both_positionals():
    args = bulk.parse_args(["in", "out"])
    assert (args.input_dir, args.output_dir) == (Path("in"), Path("out"))
    assert args.input_file is None


def test_single_file_mode_keeps_the_file():
    args = bulk.parse_args(["--in", "stack.tif", "out"])
    assert args.input_file == Path("stack.tif")


@pytest.mark.parametrize("argv", [
    ["--in", "stack.tif"],            # no output_dir
    ["in"],                           # no output_dir
    ["in", "out", "--in", "s.tif"],   # input_dir and --in together
    ["--summary-only", "in", "out"],  # --summary-only takes no input
    ["--compare", "parent", "out"],   # --compare takes no positionals
])
def test_rejected_invocations(argv):
    with pytest.raises(SystemExit):
        bulk.parse_args(argv)
