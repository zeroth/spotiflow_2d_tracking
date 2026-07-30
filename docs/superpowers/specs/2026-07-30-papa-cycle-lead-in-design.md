# PAPA cycle lead-in — design

**Date:** 2026-07-30
**Scope:** `_papa_analysis.py`, `_papa_widget.py`, `scripts/papa_analysis.py`, tests, docs

## Problem

`build_cycle_info` anchors cycle 1 at frame 1, so the first `group1_size` frames of
the stack are treated as cycle 1's group 1. In a PAPA acquisition those opening
frames are readout that precedes the first reactivation pulse — there is no pulse
before them, so they are not valid group-1 data. Every pairing downstream is
therefore off by one block: each group-2 readout gets paired with the group-1
readout *before* it instead of the one *after* it.

The skip frame between blocks is the pulse. The pulse is what starts a cycle, so
the first block of frames — which has no pulse in front of it — must be dropped
and cycle 1 must begin at the block following the first pulse.

Current output for `group1_size=200, group2_size=500, skip_frames=1`:

```
C1 G1: 1–200      skip 201
C1 G2: 202–701    skip 702
C2 G1: 703–902    skip 903
C2 G2: 904–1403   ...
```

## Desired behaviour

Physical block boundaries are unchanged, and **group identity stays tied to the
alternation from frame 1**: a `group1_size`-sized block is always group 1
(`label_group1`, e.g. DR); a `group2_size`-sized block is always group 2
(`label_group2`, e.g. PAPA). No block is ever re-sliced and no label moves.

What changes is which blocks are grouped into a cycle:

```
lead-in: 1–200    (200f, excluded from all cycles)
         skip 201
C1 G2:   202–701  (500f)
         skip 702
C1 G1:   703–902  (200f)
         skip 903
C2 G2:   904–1403 (500f)
         ...
```

A cycle is `(group 2 block, group 1 block)` in frame order. One complete cycle
requires `group1_size + skip + group2_size + skip + group1_size` frames, not
`group1_size + skip + group2_size`.

**Incomplete cycles are dropped.** A cycle is emitted only when both of its
blocks fit entirely within `max_time`. The existing `min(..., max_time)` clipping
is removed, so no group is ever truncated and no ratio is ever NaN from a missing
counterpart.

### Consequence

For equal group sizes this inverts the pairing relative to current output: what
is cycle *n*'s G2 today is paired with the *following* G1 instead of the
preceding one. Previously exported per-cycle CSVs and ratio graphs are not
comparable with new ones. This is the intent of the change, not a side effect.

## Design

### `_papa_analysis.py`

`build_cycle_info(max_time, group1_size, group2_size, skip_frames)` keeps its
signature and its return type (`list[dict]` with keys `cycle`, `group`,
`start_frame`, `end_frame`), emitted in ascending frame order — the group-2 entry
then the group-1 entry for each cycle.

Implementation: walk the raw acquisition blocks, then pair them.

1. Generate blocks from frame 1. Block *k* (0-indexed) has size
   `group1_size` when *k* is even and `group2_size` when odd; its group is
   `(k % 2) + 1`. The next block starts at `end_frame + 1 + skip_frames`.
   Record whether each block fits (`end_frame <= max_time`).
2. Discard a trailing block that does not fit. At most one block can fail to
   fit, because a non-fitting block pushes the cursor past `max_time` and ends
   the walk.
3. Drop block 0 — the lead-in.
4. Pair the remainder two at a time: cycle *n* comes from blocks `2n-1`, `2n`
   of the original numbering. An odd trailing block is dropped (incomplete
   cycle).

Guard: raise `ValueError` when `group1_size < 1` or `group2_size < 1`. The
current walk loops forever on a zero size with `skip_frames=0`; the rewrite makes
that explicit rather than inheriting the hang.

New helper:

```python
def lead_in_span(max_time: int, group1_size: int) -> tuple[int, int] | None:
    """Inclusive 1-indexed frame range of the excluded lead-in block, or None."""
```

Returns `(1, min(group1_size, max_time))`, or `None` when `max_time < 1`. It
exists so the widget and the script do not each hardcode `1..group1_size`.

`annotate_with_cycle_group` and `compute_per_cycle_summary` need no change.
Lead-in frames get `NA` for `cycle`/`group`, the same as skip frames.
`compute_per_cycle_summary` keys on cycle + group, so it picks up the new spans
as they are.

### `scripts/papa_analysis.py`

`make_combined_graph`:
- Shade the lead-in with a grey `axvspan`, legend entry `"lead-in (excluded)"`.
  The particle trace still covers the whole stack, so the data stays visible and
  it is obvious it belongs to no cycle.
- Group bands keep their existing colours (group 1 blue, group 2 orange), which
  now means the orange band comes first within each cycle.
- The cycle-boundary separator logic is positional ("the block in this cycle with
  the largest `start_frame`") and works unchanged now that the group-1 block is
  last.
- Title updated to describe the new pattern, including the skipped lead-in.

`make_ratio_graph`: no structural change. It reads `compute_per_cycle_summary`,
which already keys off cycle + group. Dropping incomplete cycles removes the
trailing blank bar without any code change here.

`main()`: the per-cycle printout gains a lead-in line, and the pattern summary
line is updated.

### `_papa_widget.py`

- `_render_combined_plot`: add the same grey lead-in band via `lead_in_span`,
  keeping the existing coloured per-group separator lines.
- Status message: the completeness check `g1 + skip + g2 + skip > max_time` is
  now wrong. Use `g1 + skip + g2 + skip + g1 > max_time` and, when zero complete
  cycles fit, say so explicitly instead of reporting "0 cycles" with no reason.
- Title string matched to the script's.

No new widget controls: the new behaviour is unconditional, so there is nothing
to toggle.

## Testing

`_tests/test_papa_analysis.py` — rewrite `TestBuildCycleInfo`:

- **Exact fit, one cycle.** `g1=5, g2=5, skip=1, max_time=17` →
  `[(1, G2, 7, 11), (1, G1, 13, 17)]`.
- **Truncated final block dropped.** `max_time=16` with the same parameters →
  `[]`, because the group-1 block would end at 17.
- **Trailing odd block dropped.** `g1=5, g2=5, skip=1, max_time=23` fits four
  blocks; after the lead-in, three remain, so only cycle 1 is emitted
  (`G2 7–11`, `G1 13–17`) and the `G2 19–23` block is dropped.
- **Two full cycles.** `g1=5, g2=5, skip=1, max_time=29` → cycles 1 and 2.
- **Unequal sizes.** `g1=200, g2=500, skip=1, max_time=1403` → cycle 1
  `G2 202–701`, `G1 703–902`; cycle 2 `G2 904–1403` alone is incomplete and
  dropped.
- **`skip_frames=0`.** `g1=3, g2=3, max_time=9` → `[(1, G2, 4, 6), (1, G1, 7, 9)]`.
- **Too-short stack.** `max_time` at or below `group1_size` → `[]`.
- **`max_time=0`** → `[]`.
- **Invalid sizes.** `group1_size=0` or `group2_size=0` raises `ValueError`.

New `TestLeadInSpan`: normal case, stack shorter than `group1_size` (clamped),
`max_time=0` → `None`.

Existing `TestAnnotateWithCycleGroup` and the `compute_per_cycle_summary` tests
use hand-built `cycle_info` lists, so they are unaffected; verify rather than
assume. `_tests/test_papa_widget.py` asserts on plotted cycle counts and status
text — audit and update its fixtures for the longer frame requirement.

## Documentation

Update the PAPA widget page and the `papa_analysis.py` script section under
`documentation/docs/` to describe the lead-in exclusion, the `(G2, G1)` cycle
composition, the frame count needed for one cycle, and the fact that incomplete
trailing cycles are not reported.
