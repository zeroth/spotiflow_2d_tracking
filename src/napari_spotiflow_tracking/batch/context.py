"""Everything an analysis gets for one movie."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .inputs import MovieJob


@dataclass
class MovieContext:
    job: MovieJob
    out_dir: Path
    config: PipelineConfig
    raw: np.ndarray
    preprocessed: np.ndarray
    spots: pd.DataFrame
    tracks: Optional[pd.DataFrame] = None
    mask: Optional[np.ndarray] = None

    def image(self, measure_on: str) -> np.ndarray:
        if measure_on == "raw":
            return self.raw
        if measure_on == "preprocessed":
            return self.preprocessed
        raise ValueError(f"measure_on must be raw or preprocessed, got '{measure_on}'")

    def analysis_dir(self, name: str) -> Path:
        path = self.out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path
