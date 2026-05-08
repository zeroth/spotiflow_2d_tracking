"""Bulk track analysis CLI.

Standalone script: preprocess, segment (Spotiflow), track (trackpy), and
analyze every .tif/.tiff stack in an input directory. One sub-folder of
artifacts per input file. See
docs/superpowers/specs/2026-05-08-bulk-track-analysis-cli-design.md
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import trackpy as tp
from scipy.ndimage import gaussian_filter
from torchimize.functions import lsq_lma_parallel
from tqdm import tqdm

# ── helpers ─────────────────────────────────────────────────────────


# ── segmentation ────────────────────────────────────────────────────


# ── fitting + mask ──────────────────────────────────────────────────


# ── tracking + analysis ─────────────────────────────────────────────


# ── plotting ────────────────────────────────────────────────────────


# ── per-file pipeline ───────────────────────────────────────────────


# ── CLI ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    sys.exit(main())
