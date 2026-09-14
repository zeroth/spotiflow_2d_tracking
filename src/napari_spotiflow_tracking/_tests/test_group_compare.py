"""Tests for the ``--compare`` group-comparison mode of the bulk CLI.

The code under test lives in ``scripts/bulk_track_analysis.py`` rather than in
the plugin package: it is summary plotting for the CLI, and it reuses that
file's private helpers (``_pct_limits``, ``_experiment_values``, ``_save``)
which have no meaning inside napari. So the script is imported by path here,
the way a user would run it.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bulk_track_analysis.py"


def _load_script():
    """Import scripts/bulk_track_analysis.py as a module, once per session."""
    if "bulk_track_analysis" in sys.modules:
        return sys.modules["bulk_track_analysis"]
    spec = importlib.util.spec_from_file_location("bulk_track_analysis", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bulk_track_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


bulk = _load_script()


# ── fixtures ────────────────────────────────────────────────────────


def _fits(n, alpha_lo, alpha_hi, seed):
    """A synthetic msd_fits.csv frame with classes spanning the alpha cuts."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(alpha_lo, alpha_hi, n)
    cls = np.where(alpha < bulk.ALPHA_CONSTRAINED_MAX, "constrained",
                   np.where(alpha > bulk.ALPHA_DIRECTED_MIN,
                            "directed", "diffusive"))
    return pd.DataFrame({
        "track_id": np.arange(n),
        "length": rng.integers(5, 40, n),
        "n_lags": rng.integers(4, 25, n),
        "D": rng.uniform(0.01, 2.0, n),
        "alpha": alpha,
        "motility_class": cls,
    })


def _props(fits, seed):
    """A track_analysis.csv frame matching a fits frame track for track."""
    rng = np.random.default_rng(seed)
    n = len(fits)
    net = rng.uniform(0.5, 20.0, n)
    return pd.DataFrame({
        "track_id": fits["track_id"],
        "length": fits["length"],
        "total_displacement": net * rng.uniform(1.0, 3.0, n),
        "net_displacement": net,
        "mean_speed": rng.uniform(0.2, 5.0, n),
        "mean_intensity": rng.uniform(100, 200, n),
        "max_intensity": rng.uniform(200, 300, n),
        "intensity_std": rng.uniform(1, 10, n),
        "D": fits["D"],
        "alpha": fits["alpha"],
        "motility_class": fits["motility_class"],
    })


def _write_stack(folder: Path, n, alpha_lo, alpha_hi, seed):
    folder.mkdir(parents=True, exist_ok=True)
    fits = _fits(n, alpha_lo, alpha_hi, seed)
    fits.to_csv(folder / "msd_fits.csv", index=False)
    _props(fits, seed).to_csv(folder / "track_analysis.csv", index=False)


@pytest.fixture
def parent(tmp_path):
    """A parent of four groups, mixing the nested and flat run layouts.

    ``AA``/``BB`` nest a stack under an experiment folder (the DE/EK/WT
    layout); ``CC``/``DD`` hold the stack directly (the PL layout). Both must
    be discovered, and each group's experiments counted the same way a
    single-root run counts them.
    """
    root = tmp_path / "Root_Test"
    _write_stack(root / "AA" / "AA01" / "stack", 120, 0.0, 2.0, 1)
    _write_stack(root / "AA" / "AA02" / "stack", 90, 0.0, 2.0, 2)
    _write_stack(root / "BB" / "BB01" / "stack", 150, 0.2, 1.8, 3)
    _write_stack(root / "CC" / "stack_one", 80, 0.1, 2.5, 4)
    _write_stack(root / "CC" / "stack_two", 60, 0.1, 2.5, 5)
    _write_stack(root / "DD" / "stack_only", 200, 0.0, 3.0, 6)
    return root


# ── group discovery ─────────────────────────────────────────────────


def test_discover_groups_finds_every_child_sorted(parent):
    assert [name for name, _ in bulk.discover_groups(parent)] == [
        "AA", "BB", "CC", "DD"]


def test_discover_groups_returns_the_child_directory(parent):
    groups = dict(bulk.discover_groups(parent))
    assert groups["AA"] == parent / "AA"
    assert groups["DD"] == parent / "DD"


def test_discover_groups_handles_both_layouts(parent):
    """Each group's experiments come back via the existing discovery."""
    groups = dict(bulk.discover_groups(parent))
    nested = bulk.discover_processed(groups["AA"])
    flat = bulk.discover_processed(groups["CC"])
    assert [label for label, _ in nested] == ["AA01/stack", "AA02/stack"]
    assert [label for label, _ in flat] == ["stack_one", "stack_two"]


def test_discover_groups_skips_children_without_results(parent):
    (parent / "notes").mkdir()
    (parent / "notes" / "readme.txt").write_text("nothing to summarise")
    assert [name for name, _ in bulk.discover_groups(parent)] == [
        "AA", "BB", "CC", "DD"]


def test_discover_groups_ignores_loose_files(parent):
    (parent / "Root_Test_motility_summary.csv").write_text("stem,n_tracks\n")
    assert [name for name, _ in bulk.discover_groups(parent)] == [
        "AA", "BB", "CC", "DD"]


def test_discover_groups_skips_its_own_output(parent):
    """A second --compare run must not pick up the first run's folder."""
    out = parent / "comparison"
    out.mkdir()
    _write_stack(out / "leftover", 10, 0.0, 2.0, 9)
    assert [name for name, _ in bulk.discover_groups(parent)] == [
        "AA", "BB", "CC", "DD"]


def test_discover_groups_empty_parent(tmp_path):
    assert bulk.discover_groups(tmp_path) == []


# ── shared axis limits ──────────────────────────────────────────────


def test_union_limits_takes_the_widest_span():
    assert bulk._union_limits([(1.0, 4.0), (0.0, 3.0), (2.0, 9.0)]) == (0.0, 9.0)


def test_union_limits_single_group_is_that_group():
    assert bulk._union_limits([(2.5, 7.5)]) == (2.5, 7.5)


def test_union_limits_empty_falls_back():
    assert bulk._union_limits([]) == (0.0, 1.0)


def test_shared_hist_axes_span_covers_every_group():
    values = {"a": np.linspace(0.0, 10.0, 500),
              "b": np.linspace(5.0, 30.0, 500)}
    lo, hi, edges, _, _ = bulk._shared_hist_axes(values)
    assert lo <= 0.5 and hi >= 29.0
    assert edges[0] == pytest.approx(lo)
    assert edges[-1] == pytest.approx(hi)
    assert len(edges) == bulk.GROUP_BINS


def test_shared_hist_axes_bins_are_identical_for_every_group():
    """Comparable counts require one set of bin edges, not one per group."""
    values = {"a": np.linspace(0.0, 10.0, 500),
              "b": np.linspace(5.0, 30.0, 500)}
    _, _, edges, _, _ = bulk._shared_hist_axes(values)
    widths = np.diff(edges)
    assert np.allclose(widths, widths[0])


def test_shared_hist_axes_count_max_is_the_tallest_bin_of_any_group():
    values = {"small": np.zeros(10), "big": np.zeros(400)}
    _, _, edges, y_counts, _ = bulk._shared_hist_axes(values)
    tallest = max(np.histogram(v, bins=edges)[0].max() for v in values.values())
    assert y_counts >= tallest
    assert y_counts == pytest.approx(tallest * 1.05)


def test_shared_hist_axes_density_max_ignores_group_size():
    """Two groups of the same shape but different n share a density ceiling."""
    rng = np.random.default_rng(0)
    values = {"small": rng.normal(0, 1, 2_000),
              "big": rng.normal(0, 1, 40_000)}
    _, _, edges, _, y_density = bulk._shared_hist_axes(values)
    fractions = [np.histogram(v, bins=edges)[0].max() / len(v)
                 for v in values.values()]
    assert y_density == pytest.approx(max(fractions) * 1.05, rel=0.02)
    assert y_density < 1.0


def test_shared_hist_axes_skips_empty_groups():
    values = {"a": np.linspace(0.0, 10.0, 100), "empty": np.array([])}
    lo, hi, _, y_counts, _ = bulk._shared_hist_axes(values)
    assert hi > lo
    assert y_counts > 0


def test_shared_hist_axes_all_groups_empty():
    lo, hi, edges, y_counts, y_density = bulk._shared_hist_axes(
        {"a": np.array([]), "b": np.array([])})
    assert (lo, hi) == (0.0, 1.0)
    assert len(edges) == bulk.GROUP_BINS
    assert y_counts > 0 and y_density > 0


def test_shared_hist_axes_is_wider_than_any_single_group():
    """The whole point: no group's own percentile cut clips another group."""
    values = {"a": np.linspace(0.0, 10.0, 500),
              "b": np.linspace(20.0, 40.0, 500)}
    lo, hi, _, _, _ = bulk._shared_hist_axes(values)
    for v in values.values():
        v_lo, v_hi = bulk._pct_limits(v)
        assert lo <= v_lo + 1e-9
        assert hi >= v_hi - 1e-9


# ── colour assignment ───────────────────────────────────────────────


def test_group_colors_assigned_in_fixed_order():
    assert bulk._group_colors(["AA", "BB"]) == {
        "AA": bulk.GROUP_COLORS[0], "BB": bulk.GROUP_COLORS[1]}


def test_group_colors_follow_the_group_not_its_rank():
    """Dropping a group must not repaint the survivors."""
    full = bulk._group_colors(["AA", "BB", "CC", "DD"])
    subset = bulk._group_colors(["AA", "BB", "CC", "DD"], order=["AA", "CC"])
    assert subset["AA"] == full["AA"]
    assert subset["CC"] == full["CC"]


def test_group_colors_cycle_past_the_palette():
    names = [f"g{i}" for i in range(len(bulk.GROUP_COLORS) + 2)]
    colors = bulk._group_colors(names)
    assert len(colors) == len(names)


# ── end-to-end ──────────────────────────────────────────────────────


@pytest.fixture
def compare_args(parent):
    """Args exactly as the CLI builds them, so the fixture cannot drift."""
    return bulk.parse_args(["--compare", str(parent)])


def test_write_group_comparison_writes_combined_and_harmonised(
        parent, compare_args):
    out = bulk.write_group_comparison(parent, compare_args)
    combined = sorted(p.name for p in out.glob("*.png"))
    assert combined, "no combined figures written"
    for metric in ("length", "speed", "motility"):
        assert any(f"combined_{metric}_summary.png" in n for n in combined)
    harmonised = out / "harmonised"
    assert harmonised.is_dir()
    for group in ("AA", "BB", "CC", "DD"):
        assert list(harmonised.glob(f"{group}_length_summary.png"))


def test_write_group_comparison_writes_density_variants(parent, compare_args):
    out = bulk.write_group_comparison(parent, compare_args)
    for metric in ("length", "speed"):
        assert list(out.glob(f"*combined_{metric}_summary_density.png"))
        assert list((out / "harmonised").glob(f"AA_{metric}_summary_density.png"))


def test_write_group_comparison_writes_the_csv(parent, compare_args):
    out = bulk.write_group_comparison(parent, compare_args)
    csvs = list(out.glob("*_comparison_summary.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert set(df["group"]) == {"AA", "BB", "CC", "DD"}
    assert {"group", "stem", "n_tracks", "directed"} <= set(df.columns)
    # Every experiment of every group appears exactly once.
    assert len(df) == 6


def test_write_group_comparison_csv_counts_match_a_single_root_run(
        parent, compare_args):
    """The per-group numbers must equal what a normal run on that group gives."""
    out = bulk.write_group_comparison(parent, compare_args)
    df = pd.read_csv(next(iter(out.glob("*_comparison_summary.csv"))))
    rows = bulk.summary_rows_from_disk(
        parent / "AA", compare_args.min_track_length)
    expected = sum(sum(r["counts"].values()) for r in rows)
    assert df[df["group"] == "AA"]["n_tracks"].sum() == expected


def test_write_group_comparison_shares_limits_across_groups(
        parent, compare_args, monkeypatch):
    """Every figure of one metric is drawn on the same x range."""
    seen = {}
    original = bulk._plot_metric_summary

    def spy(per_file, column, xlabel, scale, out_path, **kw):
        if kw.get("limits") is not None:
            seen.setdefault(column, set()).add(kw["limits"])
        return original(per_file, column, xlabel, scale, out_path, **kw)

    monkeypatch.setattr(bulk, "_plot_metric_summary", spy)
    bulk.write_group_comparison(parent, compare_args)
    assert seen, "harmonised figures were drawn without shared limits"
    for column, limits in seen.items():
        assert len(limits) == 1, f"{column} drawn on {len(limits)} ranges"


def test_write_group_comparison_needs_at_least_one_group(tmp_path, compare_args):
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    with pytest.raises(SystemExit):
        bulk.write_group_comparison(empty, compare_args)


def test_write_group_comparison_survives_a_group_with_no_directed_tracks(
        tmp_path, compare_args):
    """A group can legitimately have no directed track; the run must go on."""
    root = tmp_path / "Root_NoDirected"
    _write_stack(root / "AA" / "stack", 100, 0.0, 1.0, 1)   # none above 1.2
    _write_stack(root / "BB" / "stack", 100, 0.0, 3.0, 2)
    out = bulk.write_group_comparison(root, compare_args)
    assert list(out.glob("*combined_length_summary.png"))


# ── CLI wiring ──────────────────────────────────────────────────────


def test_compare_flag_parses_without_input_or_output():
    args = bulk.parse_args(["--compare", "some/parent"])
    assert args.compare == Path("some/parent")


def test_compare_flag_rejects_summary_only():
    with pytest.raises(SystemExit):
        bulk.parse_args(["--compare", "some/parent", "--summary-only", "x"])
