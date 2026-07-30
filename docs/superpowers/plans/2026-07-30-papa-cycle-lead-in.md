# PAPA Cycle Lead-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PAPA cycle 1 start after the first group-1 block (the "lead-in"), so every group-2 readout is paired with the group-1 readout that *follows* it, and drop incomplete trailing cycles.

**Architecture:** All the real logic change lives in one function, `build_cycle_info` in `_papa_analysis.py`. It is rewritten as a two-stage operation — walk the raw acquisition blocks, then pair them into cycles — with the first block excluded. Everything downstream (`annotate_with_cycle_group`, `compute_per_cycle_summary`, both plotting layers) consumes the same `list[dict]` shape as before, so the widget and the headless script change only in presentation and in handling the newly-possible "zero complete cycles" case.

**Tech Stack:** Python 3, numpy, pandas, matplotlib, pytest, qtpy/Qt (widget layer only).

**Spec:** [`docs/superpowers/specs/2026-07-30-papa-cycle-lead-in-design.md`](../specs/2026-07-30-papa-cycle-lead-in-design.md)

## Global Constraints

- **Logic modules import no Qt.** `_papa_analysis.py` is pure Python and unit-tested in isolation. Put algorithm changes there, not in the widget.
- **Frames are 1-indexed** everywhere in `_papa_analysis.py`, in the analysis DataFrames, and in CSVs. napari Points layers are 0-indexed; the conversion already happens in `compute_particles_per_frame`.
- **Group identity is tied to the block alternation from frame 1**, not to position within a cycle. A `group1_size`-sized block is always `group: 1` (label `label_group1`, default `DR`); a `group2_size`-sized block is always `group: 2` (label `label_group2`, default `PAPA`). No block is ever re-sliced.
- **`build_cycle_info` keeps its exact signature** `(max_time, group1_size, group2_size, skip_frames)` and its return type `list[dict]` with keys `cycle`, `group`, `start_frame`, `end_frame`.
- **Entries are emitted in ascending frame order** — the group-2 entry then the group-1 entry for each cycle.
- **A cycle needs `group1_size + skip_frames + group2_size + skip_frames + group1_size` frames.** Both of a cycle's blocks must fit entirely within `max_time`; there is no clipping and no truncated group.
- Run tests with the workspace venv: `.venv/Scripts/python.exe -m pytest ...` (never system Python).
- **Reference dataset** for verifying the plotting layers against real data — a blob CSV from a previous run of `scripts/papa_analysis.py`, columns `y,x,intensity,time` with `time` 1-indexed, 53,802 spots over frames 1–1826 (every frame populated):
  `D:/Data/PAPA/14032026/Hela/cell1/helamouse_papa_cell7_posXY0_channels_t0_posZ0_crop_seg_blobs.csv`
  Detection has already been run on this stack, so there is no need to re-run Spotiflow — reduce the CSV to per-frame counts the same way `main()` does and feed the plotting functions directly.
- **Baseline before you start:** `43 passed, 1 failed`. The single failure is `test_papa_widget.py::TestPAPAWidgetExportCSV::test_export_writes_csv` and it is **pre-existing and unrelated** to this change. Task 4 repairs it. Do not treat it as a regression you caused, and do not let it mask new failures.
- The napari widget tests are slow (~2 minutes for the two PAPA test files). Run the single test you are working on by node id during the red/green cycle; run the whole PAPA pair before committing.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/napari_spotiflow_tracking/_papa_analysis.py` | Cycle/block maths. Gains `_iter_blocks` and `lead_in_span`; `build_cycle_info` rewritten; `compute_per_cycle_summary` made safe for zero cycles. | 1, 2 |
| `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py` | Unit tests for the above. `TestBuildCycleInfo` rewritten; `TestLeadInSpan` and `TestComputePerCycleSummary` added. | 1, 2 |
| `scripts/papa_analysis.py` | Headless pipeline. `make_combined_graph` shades the lead-in; `make_ratio_graph` bails on zero cycles; `main()` printout updated. | 3 |
| `src/napari_spotiflow_tracking/_papa_widget.py` | Qt widget. Lead-in band, updated title and status message, zero-cycle ratio placeholder. | 4 |
| `src/napari_spotiflow_tracking/_tests/test_papa_widget.py` | Widget tests. Fixtures need ≥17 frames to produce a cycle; new zero-cycle test; stale export test repaired. | 4 |
| `documentation/docs/widgets/papa.md` | User-facing description of the cycle pattern. | 5 |
| `README.md` | One-line PAPA summary. | 5 |

---

### Task 1: Rewrite `build_cycle_info` and add `lead_in_span`

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_analysis.py:55-93` (replace `build_cycle_info`)
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py:7-11` (imports), `:73-119` (replace `TestBuildCycleInfo`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `build_cycle_info(max_time: int, group1_size: int, group2_size: int, skip_frames: int) -> list[dict]` — unchanged signature; each dict has keys `cycle` (int, from 1), `group` (int, 1 or 2), `start_frame` (int), `end_frame` (int). Raises `ValueError` on `group1_size < 1`, `group2_size < 1`, or `skip_frames < 0`.
  - `lead_in_span(max_time: int, group1_size: int) -> tuple[int, int] | None` — inclusive 1-indexed `(first_frame, last_frame)` of the excluded lead-in block, or `None` when `max_time < 1`.
  - `_iter_blocks(max_time: int, group1_size: int, group2_size: int, skip_frames: int) -> list[dict]` — module-private; each dict has `group`, `start_frame`, `end_frame`. Used only by `build_cycle_info`; do not import it elsewhere.

Tasks 3 and 4 both import `lead_in_span` from this module.

- [ ] **Step 1: Replace `TestBuildCycleInfo` and add `TestLeadInSpan`**

In `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py`, extend the import block at the top of the file to:

```python
from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
    lead_in_span,
)
```

Then delete the whole existing `TestBuildCycleInfo` class (lines 73–119, from `class TestBuildCycleInfo:` down to and including the `test_skip_zero` assertion) and put this in its place. Every frame number below was derived by hand from the block walk — copy them exactly, do not recompute:

```python
class TestBuildCycleInfo:
    # Block walk for g1=5, g2=5, skip=1:
    #   b0 1-5 (G1, lead-in)   b1 7-11 (G2)   b2 13-17 (G1)
    #   b3 19-23 (G2)          b4 25-29 (G1)
    # A cycle is (G2 block, G1 block), so cycle 1 = b1 + b2.

    def test_exact_fit_one_cycle(self):
        # 17 frames is the minimum for one complete cycle: 5 + 1 + 5 + 1 + 5.
        info = build_cycle_info(max_time=17, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 1, "start_frame": 13, "end_frame": 17},
        ]

    def test_truncated_final_block_dropped(self):
        # One frame short: b2 would end at 17 but max_time is 16, so cycle 1 is
        # incomplete and nothing is reported. No group is ever clipped.
        info = build_cycle_info(max_time=16, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    def test_trailing_odd_block_dropped(self):
        # 23 frames fits b0..b3. After the lead-in three blocks remain, so b3
        # (a G2 block) has no G1 partner and is dropped.
        info = build_cycle_info(max_time=23, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 1, "start_frame": 13, "end_frame": 17},
        ]

    def test_two_full_cycles(self):
        # 29 frames fits b0..b4: lead-in plus two complete (G2, G1) pairs.
        info = build_cycle_info(max_time=29, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 2, "start_frame": 7, "end_frame": 11},
            {"cycle": 1, "group": 1, "start_frame": 13, "end_frame": 17},
            {"cycle": 2, "group": 2, "start_frame": 19, "end_frame": 23},
            {"cycle": 2, "group": 1, "start_frame": 25, "end_frame": 29},
        ]

    def test_unequal_group_sizes(self):
        # g1=200, g2=500, skip=1: lead-in 1-200, G2 202-701, G1 703-902,
        # then G2 904-1403 with no G1 partner. Sizes are never re-sliced: the
        # 500-frame block stays 500 frames and stays group 2.
        info = build_cycle_info(max_time=1403, group1_size=200,
                                group2_size=500, skip_frames=1)
        assert info == [
            {"cycle": 1, "group": 2, "start_frame": 202, "end_frame": 701},
            {"cycle": 1, "group": 1, "start_frame": 703, "end_frame": 902},
        ]

    def test_skip_zero(self):
        # g1=3, g2=3, skip=0: lead-in 1-3, G2 4-6, G1 7-9.
        info = build_cycle_info(max_time=9, group1_size=3, group2_size=3,
                                skip_frames=0)
        assert info == [
            {"cycle": 1, "group": 2, "start_frame": 4, "end_frame": 6},
            {"cycle": 1, "group": 1, "start_frame": 7, "end_frame": 9},
        ]

    def test_stack_only_covers_lead_in(self):
        info = build_cycle_info(max_time=5, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    def test_zero_max_time(self):
        info = build_cycle_info(max_time=0, group1_size=5, group2_size=5,
                                skip_frames=1)
        assert info == []

    @pytest.mark.parametrize("g1,g2", [(0, 5), (5, 0)])
    def test_zero_group_size_raises(self, g1, g2):
        # A zero-size block never advances the cursor when skip_frames is 0,
        # which used to hang forever. Fail loudly instead.
        with pytest.raises(ValueError, match="must be >= 1"):
            build_cycle_info(max_time=100, group1_size=g1, group2_size=g2,
                             skip_frames=1)

    def test_negative_skip_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            build_cycle_info(max_time=100, group1_size=5, group2_size=5,
                             skip_frames=-1)


class TestLeadInSpan:
    def test_normal(self):
        assert lead_in_span(max_time=100, group1_size=20) == (1, 20)

    def test_clamped_to_short_stack(self):
        # Shading must not run off the end of a stack shorter than the lead-in.
        assert lead_in_span(max_time=8, group1_size=20) == (1, 8)

    def test_no_frames(self):
        assert lead_in_span(max_time=0, group1_size=20) is None
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py -q`

Expected: collection fails with `ImportError: cannot import name 'lead_in_span' from 'napari_spotiflow_tracking._papa_analysis'`.

- [ ] **Step 3: Implement `_iter_blocks`, rewrite `build_cycle_info`, add `lead_in_span`**

In `src/napari_spotiflow_tracking/_papa_analysis.py`, replace the entire existing `build_cycle_info` function (lines 55–93) with the three functions below. Keep them in this order, between `compute_particles_per_frame` and `annotate_with_cycle_group`.

```python
def _iter_blocks(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Walk the raw acquisition blocks from frame 1.

    Block *k* (0-indexed) holds ``group1_size`` frames when *k* is even and
    ``group2_size`` when *k* is odd; its group number is ``(k % 2) + 1``. The
    next block starts ``skip_frames`` frames after this one ends.

    Only blocks that fit entirely within ``max_time`` are returned — a block
    that would overrun ends the walk, since nothing after it can fit either.
    """
    sizes = (group1_size, group2_size)
    blocks: list[dict] = []
    current = 1
    k = 0
    while current <= max_time:
        end = current + sizes[k % 2] - 1
        if end > max_time:
            break
        blocks.append({
            "group": (k % 2) + 1,
            "start_frame": current,
            "end_frame": end,
        })
        current = end + 1 + skip_frames
        k += 1
    return blocks


def build_cycle_info(
    max_time: int,
    group1_size: int,
    group2_size: int,
    skip_frames: int,
) -> list[dict]:
    """Generate cycle/group boundary information. Cycles and frames are 1-indexed.

    The first ``group1_size`` frames are the *lead-in*: readout that precedes
    the first reactivation pulse, so it belongs to no cycle (see
    :func:`lead_in_span`). Cycle 1 starts at the block after the lead-in, which
    makes a cycle ``(group 2 block, group 1 block)`` in frame order — each
    group-2 readout is paired with the group-1 readout that *follows* it.

    Group identity follows the block alternation from frame 1 and never moves:
    a ``group1_size``-sized block is always group 1, a ``group2_size``-sized
    block always group 2.

    Only complete cycles are returned — both blocks must fit entirely within
    ``max_time``, so one cycle needs
    ``group1_size + skip_frames + group2_size + skip_frames + group1_size``
    frames. Nothing is clipped and no group is ever truncated.

    Entries are in ascending frame order: the group-2 entry then the group-1
    entry for each cycle.
    """
    if group1_size < 1 or group2_size < 1:
        raise ValueError("group1_size and group2_size must be >= 1")
    if skip_frames < 0:
        raise ValueError("skip_frames must be >= 0")

    # Drop block 0 — the lead-in — then take what remains two blocks at a time.
    # An odd trailing block is an incomplete cycle and is discarded.
    after_lead_in = _iter_blocks(
        max_time, group1_size, group2_size, skip_frames,
    )[1:]

    cycle_info: list[dict] = []
    for cycle_num, i in enumerate(range(0, len(after_lead_in) - 1, 2), start=1):
        for block in (after_lead_in[i], after_lead_in[i + 1]):
            cycle_info.append({
                "cycle": cycle_num,
                "group": block["group"],
                "start_frame": block["start_frame"],
                "end_frame": block["end_frame"],
            })
    return cycle_info


def lead_in_span(max_time: int, group1_size: int) -> tuple[int, int] | None:
    """Inclusive 1-indexed frame range of the excluded lead-in block.

    Returns ``None`` when the stack has no frames. The end is clamped to
    ``max_time`` so a stack shorter than ``group1_size`` still shades correctly.
    """
    if max_time < 1:
        return None
    return (1, min(group1_size, max_time))
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py -q`

Expected: PASS. `TestAnnotateWithCycleGroup` builds its `cycle_info` by hand, so it must still pass untouched — if it fails, you changed something you should not have.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py
git commit -m "feat(papa): start cycle 1 after the lead-in block

The first group1_size frames precede the first reactivation pulse, so
they belong to no cycle. Cycle 1 now begins at the next block, making a
cycle (G2, G1) in frame order and pairing each G2 readout with the G1
readout that follows it. Incomplete trailing cycles are dropped instead
of being clipped, and zero/negative sizes raise instead of hanging."
```

---

### Task 2: Keep `compute_per_cycle_summary` columns when there are no cycles

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_analysis.py` (the `compute_per_cycle_summary` body — after Task 1 it sits at the end of the file)
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py` (imports; add `TestComputePerCycleSummary`)

**Why this task exists:** `build_cycle_info` can now legitimately return `[]` — the widget's default 5/5/1 pattern needs 17 frames, so short stacks produce zero cycles. `pd.DataFrame([])` has no columns, so every consumer's `per_cycle["cycle"]` raises `KeyError`. Tasks 3 and 4 depend on this being fixed first.

**Interfaces:**
- Consumes: `build_cycle_info` from Task 1.
- Produces: `compute_per_cycle_summary(per_frame, cycle_info, label1="G1", label2="G2") -> pd.DataFrame` — unchanged signature. Always has exactly these columns in this order: `cycle`, `{label1}_count`, `{label2}_count`, `{label1}_over_{label2}`, `{label2}_over_{label1}`, even when the frame is empty.

- [ ] **Step 1: Write the failing tests**

Add `compute_per_cycle_summary` to the import block in `src/napari_spotiflow_tracking/_tests/test_papa_analysis.py` so it reads:

```python
from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
    compute_per_cycle_summary,
    lead_in_span,
)
```

Then append this class to the end of the file:

```python
class TestComputePerCycleSummary:
    def test_counts_and_ratios(self):
        # 17 frames, g1=5/g2=5/skip=1 -> one cycle: G2 7-11, G1 13-17.
        per_frame = pd.DataFrame({
            "frame": list(range(1, 18)),
            "n_particles": [0] * 17,
        })
        per_frame.loc[per_frame["frame"].between(7, 11), "n_particles"] = 2
        per_frame.loc[per_frame["frame"].between(13, 17), "n_particles"] = 1
        cycle_info = build_cycle_info(max_time=17, group1_size=5,
                                      group2_size=5, skip_frames=1)

        out = compute_per_cycle_summary(per_frame, cycle_info, "DR", "PAPA")

        assert out["cycle"].tolist() == [1]
        assert out["PAPA_count"].tolist() == [10]   # 5 frames x 2 spots
        assert out["DR_count"].tolist() == [5]      # 5 frames x 1 spot
        assert out["PAPA_over_DR"].tolist() == [2.0]
        assert out["DR_over_PAPA"].tolist() == [0.5]

    def test_empty_cycle_info_keeps_columns(self):
        # A stack too short for one complete cycle must still yield a frame the
        # plotting code can index into.
        per_frame = pd.DataFrame({"frame": [1, 2], "n_particles": [1, 1]})
        out = compute_per_cycle_summary(per_frame, [], "DR", "PAPA")
        assert len(out) == 0
        assert list(out.columns) == [
            "cycle", "DR_count", "PAPA_count", "DR_over_PAPA", "PAPA_over_DR",
        ]
```

- [ ] **Step 2: Run the tests and verify the empty case fails**

Run: `.venv/Scripts/python.exe -m pytest "src/napari_spotiflow_tracking/_tests/test_papa_analysis.py::TestComputePerCycleSummary" -v`

Expected: `test_counts_and_ratios` PASSES (existing behaviour is correct for non-empty input); `test_empty_cycle_info_keeps_columns` FAILS on the `list(out.columns)` assertion, because an all-empty `pd.DataFrame([])` reports `[]` for its columns.

- [ ] **Step 3: Declare the columns up front and pass them to the constructor**

In `compute_per_cycle_summary`, add a `columns` list immediately after the `cycles = sorted(...)` line:

```python
    cycles = sorted({info["cycle"] for info in cycle_info})
    columns = [
        "cycle",
        f"{label1}_count",
        f"{label2}_count",
        f"{label1}_over_{label2}",
        f"{label2}_over_{label1}",
    ]
    rows = []
```

and change the final line of the function from `return pd.DataFrame(rows)` to:

```python
    # Pass `columns` explicitly so a zero-cycle result is still an empty frame
    # with the right columns rather than a shapeless one the plots can't index.
    return pd.DataFrame(rows, columns=columns)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_papa_analysis.py -q`

Expected: PASS, all classes.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_analysis.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py
git commit -m "fix(papa): keep per-cycle summary columns when no cycles fit

build_cycle_info can now return an empty list, and pd.DataFrame([]) has
no columns, so consumers raised KeyError. Declare the columns explicitly."
```

---

### Task 3: Shade the lead-in and handle zero cycles in the headless script

**Files:**
- Modify: `scripts/papa_analysis.py:24-27` (import), `:60-109` (`make_combined_graph`), `:112-123` (`make_ratio_graph` head), `:310-317` (`main` printout)
- Smoke check: `<scratchpad>/papa_graph_smoke.py` (throwaway, not committed)

**Interfaces:**
- Consumes: `lead_in_span` (Task 1), `compute_per_cycle_summary` with guaranteed columns (Task 2).
- Produces: nothing later tasks depend on. `make_combined_graph` and `make_ratio_graph` keep their existing signatures.

There is no pytest coverage for `scripts/` (`testpaths` is `src/napari_spotiflow_tracking/_tests`), so this task is verified by a scratchpad smoke script that drives both plotting functions with the **reference blob CSV** from Global Constraints and checks the PNGs land. Real counts over 1826 frames make the output eyeball-checkable in a way synthetic data would not.

- [ ] **Step 1: Write the smoke check**

Write this to `papa_graph_smoke.py` in your scratchpad directory. Both frame layouts asserted below were derived by hand from the block walk for `max_time=1826` — copy them exactly.

```python
"""Smoke-check scripts/papa_analysis.py plotting against a real PAPA blob CSV."""
import sys
from pathlib import Path

REPO = Path("d:/Code/Lab/spotiflow_tracking")
BLOBS = Path("D:/Data/PAPA/14032026/Hela/cell1/"
             "helamouse_papa_cell7_posXY0_channels_t0_posZ0_crop_seg_blobs.csv")
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd

from napari_spotiflow_tracking._papa_analysis import build_cycle_info
import papa_analysis

out = Path(__file__).parent

# Same reduction main() applies: blob rows -> 1-indexed per-frame counts.
df = pd.read_csv(BLOBS)
per_frame = (df.groupby("time").size().reset_index(name="n_particles")
               .rename(columns={"time": "frame"}))
max_time = int(per_frame["frame"].max())
assert max_time == 1826, max_time

# 200/500/skip=1 over 1826 frames: lead-in 1-200, then 202-701 (G2),
# 703-902 (G1), 904-1403 (G2), 1405-1604 (G1). The 500-frame block starting at
# 1606 would end at 2105, so it never fits and cycle 3 is incomplete.
cycle_info = build_cycle_info(max_time, 200, 500, 1)
assert cycle_info == [
    {"cycle": 1, "group": 2, "start_frame": 202, "end_frame": 701},
    {"cycle": 1, "group": 1, "start_frame": 703, "end_frame": 902},
    {"cycle": 2, "group": 2, "start_frame": 904, "end_frame": 1403},
    {"cycle": 2, "group": 1, "start_frame": 1405, "end_frame": 1604},
], cycle_info

papa_analysis.make_combined_graph(
    per_frame, cycle_info, max_time, 200, 500, 1, "DR", "PAPA",
    str(out / "real_200_500_combined.png"),
)
papa_analysis.make_ratio_graph(
    per_frame, cycle_info, "DR", "PAPA", str(out / "real_200_500_ratio.png"),
)

# 200/200/skip=1 over the same stack: 9 blocks fit (stride 201, last is
# 1609-1808), so after the lead-in there are 8 blocks -> 4 complete cycles.
cycle_info_200 = build_cycle_info(max_time, 200, 200, 1)
assert max(c["cycle"] for c in cycle_info_200) == 4, cycle_info_200
assert cycle_info_200[0] == {
    "cycle": 1, "group": 2, "start_frame": 202, "end_frame": 401,
}, cycle_info_200[0]

papa_analysis.make_combined_graph(
    per_frame, cycle_info_200, max_time, 200, 200, 1, "DR", "PAPA",
    str(out / "real_200_200_combined.png"),
)
papa_analysis.make_ratio_graph(
    per_frame, cycle_info_200, "DR", "PAPA",
    str(out / "real_200_200_ratio.png"),
)

for name in ("real_200_500_combined.png", "real_200_500_ratio_total.png",
             "real_200_500_ratio_g1_g2.png", "real_200_500_ratio_g2_g1.png",
             "real_200_200_combined.png", "real_200_200_ratio_total.png"):
    assert (out / name).exists(), f"missing {name}"

# Zero-cycle path: the real stack is far too long to produce it, so trim.
short = per_frame[per_frame["frame"] <= 12].copy()
empty_info = build_cycle_info(12, 5, 5, 1)
assert empty_info == [], empty_info

papa_analysis.make_combined_graph(
    short, empty_info, 12, 5, 5, 1, "DR", "PAPA",
    str(out / "zero_cycle_combined.png"),
)
papa_analysis.make_ratio_graph(
    short, empty_info, "DR", "PAPA", str(out / "zero_cycle_ratio.png"),
)
assert (out / "zero_cycle_combined.png").exists()
assert not (out / "zero_cycle_ratio_total.png").exists(), \
    "ratio graphs should be skipped when no cycle is complete"

print("OK")
```

- [ ] **Step 2: Run the smoke check and verify it fails**

Run: `.venv/Scripts/python.exe <scratchpad>/papa_graph_smoke.py`

Expected: FAIL — `AssertionError: ratio graphs should be skipped when no cycle is complete` (or, depending on matplotlib's ordering, an earlier error from plotting an empty summary). The point is that the zero-cycle path is not yet handled. The two real-data sections should already pass, since Task 1 fixed `build_cycle_info`; if either `cycle_info` assertion fails, stop and re-check Task 1 rather than editing the assertion.

- [ ] **Step 3: Add the lead-in band, update the title, and guard zero cycles**

**3a.** Extend the `_papa_analysis` import at the top of `scripts/papa_analysis.py`:

```python
from napari_spotiflow_tracking._papa_analysis import (
    build_cycle_info,
    compute_per_cycle_summary,
    lead_in_span,
)
```

**3b.** In `make_combined_graph`, update the docstring and insert the grey band immediately before the `# Background band per group.` comment:

```python
    """Create and save the combined particles-per-frame graph.

    The excluded lead-in block is shaded grey; each cycle's group ranges are
    shaded as background bands (group 1 blue, group 2 orange) and a dotted
    separator marks each cycle boundary. The particle trace covers the whole
    stack, so lead-in data stays visible even though it belongs to no cycle.
    """
```

```python
    # Excluded lead-in: readout before the first pulse, in no cycle. Shaded so
    # it is visibly present but visibly not part of the analysis.
    lead_in = lead_in_span(max_time, group1_size)
    if lead_in is not None:
        ax.axvspan(lead_in[0] - 0.5, lead_in[1] + 0.5, color="gray",
                   alpha=0.18, linewidth=0, label="lead-in (excluded)",
                   zorder=0)

    # Background band per group. Bands span whole frames, hence the ±0.5.
```

**3c.** Still in `make_combined_graph`, replace the `ax.set_title(...)` call with:

```python
    ax.set_title(
        f"All Cycles Combined — Particles per Frame  "
        f"(lead-in {group1_size}f excluded; cycle = {label_group2} {group2_size}f "
        f"+ skip{skip_frames} + {label_group1} {group1_size}f + skip{skip_frames})"
    )
```

**3d.** In `make_ratio_graph`, bail out before any plotting. Insert directly after the `summary = compute_per_cycle_summary(...)` call and before `cycles = summary["cycle"].tolist()`:

```python
    if summary.empty:
        print("No complete cycles in range; skipping ratio graphs.")
        return

    cycles = summary["cycle"].tolist()
```

- [ ] **Step 4: Run the smoke check and verify it passes**

Run: `.venv/Scripts/python.exe <scratchpad>/papa_graph_smoke.py`

Expected: prints `OK`. Then open `real_200_500_combined.png` and confirm by eye:

- Frames 1–200 are a grey band labelled "lead-in (excluded)", with the particle trace still drawn across them.
- Within each cycle the orange band (PAPA / G2) comes **before** the blue band (DR / G1) — the opposite order from the old output.
- The wide bands are 500 frames and the narrow ones 200; no band is a re-sliced 200 frames sitting inside what should be a 500-frame block.
- Frames 1606–1826 at the right edge are unshaded — the dropped incomplete cycle.
- The title reads `lead-in 200f excluded; cycle = PAPA 500f + skip1 + DR 200f + skip1`.

Also open `real_200_500_ratio_total.png` and confirm there are exactly two cycles on the x-axis with no blank third bar.

- [ ] **Step 5: Update `main()`'s summary printout**

Replace the block at the end of `main()` (from `# Print summary` through the `for info in cycle_info:` loop) with:

```python
    # Print summary
    n_cycles = max((c["cycle"] for c in cycle_info), default=0)
    frames_per_cycle = (
        args.group1_size + args.skip_frames + args.group2_size
        + args.skip_frames + args.group1_size
    )
    print(f"\nTotal complete cycles: {n_cycles}")
    print(f"Pattern: lead-in {args.group1_size}f (excluded), then "
          f"{args.label_group2}={args.group2_size}f + skip{args.skip_frames} "
          f"+ {args.label_group1}={args.group1_size}f + skip{args.skip_frames}")
    lead_in = lead_in_span(max_time, args.group1_size)
    if lead_in is not None:
        print(f"  Lead-in (excluded): frames {lead_in[0]}-{lead_in[1]}")
    for info in cycle_info:
        print(f"  Cycle {info['cycle']}, Group {info['group']}: "
              f"frames {info['start_frame']}-{info['end_frame']}")
    if n_cycles == 0:
        print(f"  No complete cycles fit in {max_time} frames "
              f"(need at least {frames_per_cycle}).")
```

- [ ] **Step 6: Verify the script still imports and its CLI help works**

Run: `.venv/Scripts/python.exe scripts/papa_analysis.py --help`

Expected: the argparse help text prints with no traceback. (A full run needs a TIFF and a Spotiflow model, which is out of scope here — the smoke check already covered the changed functions.)

- [ ] **Step 7: Commit**

```bash
git add scripts/papa_analysis.py
git commit -m "feat(scripts): show the excluded lead-in on PAPA graphs

Shade frames 1..group1_size grey with a legend entry, name the (G2, G1)
cycle composition in the title, skip the ratio graphs when no cycle is
complete, and report the lead-in span in the summary printout."
```

---

### Task 4: Update the PAPA widget and its tests

**Files:**
- Modify: `src/napari_spotiflow_tracking/_papa_widget.py:18-24` (import block), `:304-311` (status message), `:354-362` (title + legend), `:365-372` (`_render_ratio_plot` guard), and the lead-in band inside `_render_combined_plot`
- Modify: `src/napari_spotiflow_tracking/_tests/test_papa_widget.py:143-191`, `:193-199`, `:218-221`, `:244-272`

**Interfaces:**
- Consumes: `lead_in_span` (Task 1), `compute_per_cycle_summary` with guaranteed columns (Task 2).
- Produces: nothing later tasks depend on.

**Note on the export test:** `TestPAPAWidgetExportCSV::test_export_writes_csv` fails on `main` today — it monkeypatches `QFileDialog.getSaveFileName` and expects one CSV, but the widget uses `QFileDialog.getExistingDirectory` and writes `papa_per_frame.csv` and `papa_per_cycle.csv`. That is a pre-existing bug in the test, not something this change caused. Repair it here, because this task edits that class's fixture anyway and the suite must be green to verify the rest.

- [ ] **Step 1: Update the widget tests**

In `src/napari_spotiflow_tracking/_tests/test_papa_widget.py`:

**1a.** In `TestPAPAWidgetPlotPipeline`, change the fixture default and every call site to 17 frames — with the default 5/5/1 pattern, 12 frames no longer contains a complete cycle. Replace the whole class body's helper and its five tests with:

```python
class TestPAPAWidgetPlotPipeline:
    # The default 5/5/1 pattern needs 5 + 1 + 5 + 1 + 5 = 17 frames for one
    # complete cycle, because the first group-1 block is an excluded lead-in.
    def _make_synthetic_layer(self, viewer, n_frames=17, spots_per_frame=3):
        """Add a Points layer with `spots_per_frame` spots on each of `n_frames` frames."""
        rows = []
        rng = np.random.default_rng(0)
        for f in range(n_frames):
            for _ in range(spots_per_frame):
                rows.append([f, rng.uniform(0, 100), rng.uniform(0, 100)])
        viewer.add_points(np.array(rows), name="spots")

    def test_plot_renders_combined_axes(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._group1_size.setText("5")
        widget._group2_size.setText("5")
        widget._skip_frames.setText("1")
        widget._plot_btn.click()
        assert len(widget._fig_combined.axes) == 1

    def test_plot_caches_per_frame_df(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17, spots_per_frame=3)
        widget._plot_btn.click()
        df = widget._last_per_frame_df
        assert df is not None
        assert list(df.columns) == ["frame", "n_particles", "cycle", "group"]
        assert df["n_particles"].tolist() == [3] * 17

    def test_lead_in_frames_have_no_cycle(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        df = widget._last_per_frame_df
        # Frames 1-5 are the excluded lead-in; 7-11 are cycle 1 group 2 and
        # 13-17 are cycle 1 group 1.
        lead_in = df[df["frame"].between(1, 5)]
        assert lead_in["cycle"].isna().all()
        assert lead_in["group"].isna().all()
        assert df.loc[df["frame"] == 9, "group"].tolist() == [2]
        assert df.loc[df["frame"] == 15, "group"].tolist() == [1]

    def test_plot_status_message(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        assert "plotted" in widget._status_label.text().lower()
        assert "1 cycles" in widget._status_label.text() or \
               "1 cycle" in widget._status_label.text()

    def test_zero_complete_cycles_reports_requirement(self, papa_widget):
        widget, viewer = papa_widget
        # 12 frames is one block short of a complete cycle.
        self._make_synthetic_layer(viewer, n_frames=12)
        widget._plot_btn.click()
        text = widget._status_label.text()
        assert "no complete cycles" in text.lower()
        assert "17" in text
        # Both figures must still render rather than raising.
        assert len(widget._fig_combined.axes) == 1
        assert len(widget._fig_ratio.axes) == 1

    def test_export_button_enabled_after_plot(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        widget._plot_btn.click()
        assert widget._export_btn.isEnabled() is True

    def test_active_tab_switches_to_plots(self, papa_widget):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer, n_frames=17)
        assert widget._tabs.currentIndex() == 0
        widget._plot_btn.click()
        assert widget._tabs.currentIndex() == 1
```

**1b.** In `TestPAPAWidgetTheming`, change the helper's frame count from 12 to 17 so it exercises the real three-axis ratio figure:

```python
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(17) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")
```

**1c.** In `TestPAPAWidgetExportCSV`, change the helper to 17 frames and rewrite `test_export_writes_csv` against the directory-picker API the widget actually uses:

```python
class TestPAPAWidgetExportCSV:
    def _make_synthetic_layer(self, viewer):
        rows = [[f, 10.0, 10.0] for f in range(17) for _ in range(3)]
        viewer.add_points(np.array(rows), name="spots")

    def test_export_writes_csv(self, papa_widget, tmp_path, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            lambda *a, **kw: str(tmp_path),
        )
        widget._export_btn.click()

        per_frame_path = tmp_path / "papa_per_frame.csv"
        per_cycle_path = tmp_path / "papa_per_cycle.csv"
        assert per_frame_path.exists()
        assert per_cycle_path.exists()

        df = pd.read_csv(per_frame_path)
        assert list(df.columns) == ["frame", "n_particles", "cycle", "group"]
        # Frames are 1-indexed.
        assert df["frame"].min() == 1
        # Frame 3 is in the excluded lead-in and frame 6 is a skip gap: both
        # carry NaN cycle/group but keep their particle count.
        for frame in (3, 6):
            row = df[df["frame"] == frame].iloc[0]
            assert pd.isna(row["cycle"])
            assert pd.isna(row["group"])
            assert int(row["n_particles"]) == 3

        per_cycle = pd.read_csv(per_cycle_path)
        assert per_cycle["cycle"].tolist() == [1]

    def test_export_cancel_does_nothing(self, papa_widget, monkeypatch):
        widget, viewer = papa_widget
        self._make_synthetic_layer(viewer)
        widget._plot_btn.click()

        from qtpy.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            lambda *a, **kw: "",
        )
        # Should not raise.
        widget._export_btn.click()
```

Leave `TestPAPAWidgetRatioPlot` alone: its 24-frame fixture fits blocks 1–5, 7–11, 13–17, 19–23, which after the lead-in yields one complete cycle, so its three-axis and title assertions still hold.

- [ ] **Step 2: Run the widget tests and verify the new ones fail**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py -q`

Expected: `test_zero_complete_cycles_reports_requirement` FAILS (the status text still says "Plotted 0 cycles"), and `test_export_writes_csv` may still fail until the plot path stops raising on the empty summary. `test_plot_status_message` should already pass, since 17 frames now yields one cycle.

- [ ] **Step 3: Update the widget**

**3a.** Add `lead_in_span` to the `_papa_analysis` import block near line 18–24 of `src/napari_spotiflow_tracking/_papa_widget.py`, keeping the existing alphabetical order. The file uses absolute imports, so it becomes:

```python
from napari_spotiflow_tracking._papa_analysis import (
    annotate_with_cycle_group,
    build_cycle_info,
    compute_particles_per_frame,
    compute_per_cycle_summary,
    lead_in_span,
)
```

**3b.** In `_render_combined_plot`, insert the lead-in band immediately after the `ax.plot(...)` call and before `sep_colors = ["tab:blue", "tab:orange"]`:

```python
        # Excluded lead-in: readout before the first pulse, in no cycle.
        lead_in = lead_in_span(max_time, g1)
        if lead_in is not None:
            ax.axvspan(lead_in[0] - 0.5, lead_in[1] + 0.5, color="gray",
                       alpha=0.25, linewidth=0,
                       label="lead-in (excluded)", zorder=0)
```

**3c.** Still in `_render_combined_plot`, replace the `ax.set_title(...)` call and the `if sep1_labeled or sep2_labeled:` legend guard with:

```python
        ax.set_title(
            f"All Cycles Combined — Particles per Frame  "
            f"(lead-in {g1}f excluded; cycle = {label2} {g2}f + skip{skip}f "
            f"+ {label1} {g1}f + skip{skip}f)"
        )
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Number of Particles")
        # The lead-in band alone is worth a legend, so ask matplotlib what it
        # has rather than tracking the separator flags.
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
```

`sep1_labeled` / `sep2_labeled` are still needed by the separator loop above — keep those assignments, just stop using them for the legend guard.

**3d.** In `_render_ratio_plot`, add a zero-cycle placeholder right after `fig.clear()`:

```python
    def _render_ratio_plot(self, fig, per_cycle, label1, label2):
        fig.clear()
        if len(per_cycle) == 0:
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No complete cycles in range",
                    ha="center", va="center", transform=ax.transAxes)
            self._style_figure(fig)
            return

        cycles = per_cycle["cycle"].tolist()
```

**3e.** In `_on_plot_clicked`, replace the status-message block (the `n_cycles = ...` line through the `if g1 + skip + g2 + skip > max_time ...` clause) with:

```python
        n_cycles = max((c["cycle"] for c in cycle_info), default=0)
        if n_cycles == 0:
            frames_per_cycle = g1 + skip + g2 + skip + g1
            msg = (
                f"No complete cycles in {max_time} frames: the excluded "
                f"lead-in plus one cycle needs {frames_per_cycle} frames."
            )
        else:
            msg = f"Plotted {n_cycles} cycles across {max_time} frames."
        if n_neg > 0:
            msg += f" Negative frame indices found; clipped to 0 ({n_neg} rows)."
```

The old "cycle pattern exceeds frame range; only partial cycles drawn" clause goes away — partial cycles no longer exist, and the zero-cycle branch says something actionable instead.

- [ ] **Step 4: Run the full PAPA test suite and verify it passes**

Run: `.venv/Scripts/python.exe -m pytest src/napari_spotiflow_tracking/_tests/test_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_analysis.py -q`

Expected: all PASS, including the previously-failing `test_export_writes_csv`. This takes roughly two minutes.

- [ ] **Step 5: Run the whole suite to check for collateral damage**

Run: `.venv/Scripts/python.exe -m pytest -q`

Expected: no failures. If anything outside the PAPA files fails, compare against the baseline before assuming you caused it.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking/_papa_widget.py src/napari_spotiflow_tracking/_tests/test_papa_widget.py
git commit -m "feat(papa-widget): show the excluded lead-in and handle zero cycles

Shade the lead-in on the combined plot, name the (G2, G1) cycle
composition in the title, and replace the partial-cycle warning with a
message stating how many frames one cycle needs. The ratio figure now
renders a placeholder instead of raising when no cycle is complete.

Also repairs test_export_writes_csv, which had been failing against the
widget's directory-picker export since it moved off getSaveFileName."
```

---

### Task 5: Update the documentation

**Files:**
- Modify: `documentation/docs/widgets/papa.md:23-35`, `:57-65`, `:69-73`
- Modify: `README.md:112`

**Interfaces:**
- Consumes: the final behaviour from Tasks 1–4.
- Produces: nothing.

**Note on scope:** the MkDocs tree has no dedicated page for `scripts/papa_analysis.py` — `documentation/docs/` contains only `index.md`, `installation.md`, `maintaining.md`, `workflow.md`, and `widgets/`, and the script's CLI is documented in `README.md`. Do not go looking for a scripts page to edit; `papa.md` plus the README is the whole surface. Also leave `documentation/site/` alone — it is generated output that happens to be committed, and regenerating it here would bury the real change in a huge diff.

- [ ] **Step 1: Rewrite the Cycle Pattern section of the widget page**

In `documentation/docs/widgets/papa.md`, replace the `### Cycle Pattern` section (the heading down to and including the "Partial cycles are handled gracefully." paragraph) with:

```markdown
### Cycle Pattern

Define how frames repeat. The **first `Group 1 size` frames are a lead-in** —
readout that precedes the first reactivation pulse — and belong to no cycle.
Cycle 1 starts at the block after that, so a cycle runs
`G2 → (skip) → G1 → (skip)`: each group-2 readout is paired with the group-1
readout that **follows** it.

| Field | Notes |
|---|---|
| **Group 1 size** | Number of frames in group 1 per cycle. Also the length of the excluded lead-in. |
| **Group 2 size** | Number of frames in group 2 per cycle. |
| **Skip frames** | Number of frames between blocks that are excluded from both groups (e.g. shutter / settling frames). This is the pulse that starts a cycle. |

Group identity follows the block alternation from frame 1 and never moves: a
`Group 1 size`-length block is always group 1, a `Group 2 size`-length block
always group 2. No block is ever re-sliced.

One complete cycle needs
`Group 1 size + Skip + Group 2 size + Skip + Group 1 size` frames — with
`G1=10, G2=10, Skip=2` that is 34. **Incomplete cycles are not reported:** both
of a cycle's blocks must fit entirely within the stack, so a trailing block with
no partner is dropped rather than truncated. If no complete cycle fits, the
status bar says so and names the frame count required.
```

- [ ] **Step 2: Update the Plots tab and Outputs sections**

In the same file, replace list item 1 under `## Plots tab` with:

```markdown
1. **Particles per frame** — line plot of detected particle count over the
   entire experiment. The excluded lead-in is shaded grey, G1 / G2 regions get
   colored backgrounds, and vertical separators mark cycle boundaries. The
   trace covers every frame, so lead-in and skip data stay visible even though
   they belong to no cycle.
```

and replace the **CSV export** row of the Outputs table with:

```markdown
| **CSV export** | `papa_per_frame.csv` — one row per frame with `frame`, `n_particles`, `cycle`, `group`; lead-in and skip frames carry an empty `cycle`/`group`. `papa_per_cycle.csv` — one row per complete cycle with per-group counts and both ratios. |
```

- [ ] **Step 3: Update the README's PAPA paragraph**

In `README.md`, replace the PAPA usage line (the paragraph beginning "For cyclical / alternating experiments:") with:

```markdown
For cyclical / alternating experiments: pick the detected Points layer, set the cycle pattern (group sizes and skip frames), label the groups, and **Plot** to see particles-per-frame and per-cycle bar charts. The first `group 1 size` frames are treated as an excluded lead-in — the readout before the first pulse — so cycle 1 starts at the following block and each group-2 readout is paired with the group-1 readout after it. Incomplete trailing cycles are not reported. **Export CSV** for the annotated per-frame and per-cycle tables.
```

- [ ] **Step 4: Verify the docs build**

Run: `.venv/Scripts/python.exe -m mkdocs build -f documentation/mkdocs.yml --strict -d <scratchpad>/mkdocs-check`

The `-d` is deliberate — it sends output to your scratchpad instead of overwriting the committed `documentation/site/` tree.

Expected: builds with no warnings. If `mkdocs` is not installed in the venv, install the docs requirements first with `.venv/Scripts/python.exe -m pip install -r documentation/requirements.txt`, or skip this step and state plainly in your report that the docs build was not verified.

- [ ] **Step 5: Commit**

```bash
git add documentation/docs/widgets/papa.md README.md
git commit -m "docs(papa): document the excluded lead-in and (G2, G1) cycles"
```

---

## Verification Summary

After all five tasks:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: green, with **zero** failures — one better than the starting baseline, because Task 4 repairs the pre-existing `test_export_writes_csv`.

Report honestly if the docs build step was skipped, and mention in the final summary that per-cycle CSVs and ratio graphs produced before this change are **not comparable** with ones produced after it: the pairing has shifted by one block by design.
