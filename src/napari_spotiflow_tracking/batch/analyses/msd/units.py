"""Scales and unit strings for the MSD analysis.

Phase 1's ``Calibration`` carries the two user-facing values; ``sigma_nm`` is an
``msd`` setting, and the derived scales live here so the figures and the fits
share one source. Ported from ``scripts/bulk_track_analysis.py`` (Calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import Calibration


@dataclass(frozen=True)
class Units:
    """Optional physical calibration for the MSD analysis.

    When both ``pixel_size_nm`` and ``frame_interval_s`` are supplied, lengths
    are converted px → µm and lags frames → s, so D comes out in µm²/s^α and
    the fixed localization error ``sigma_nm`` (35 nm in Commun Biol 4, 137)
    is meaningful. Otherwise everything stays in px / frames and the σ offset
    is dropped, because a nanometre figure has no meaning in pixel units.
    """

    calibration: Calibration
    sigma_nm: float = 35.0

    @property
    def is_physical(self) -> bool:
        return self.calibration.is_physical

    @property
    def length_scale(self) -> float:
        """Multiplier taking pixel coordinates to the working length unit."""
        px = self.calibration.pixel_size_nm
        return px / 1000.0 if self.is_physical and px is not None else 1.0

    @property
    def delta(self) -> float:
        """Working time unit per frame."""
        dt = self.calibration.frame_interval_s
        return dt if self.is_physical and dt is not None else 1.0

    @property
    def sigma(self) -> float:
        """Localization error in the working length unit (0 when uncalibrated)."""
        return self.sigma_nm / 1000.0 if self.is_physical else 0.0

    @property
    def length_unit(self) -> str:
        return "µm" if self.is_physical else "px"

    @property
    def time_unit(self) -> str:
        return "s" if self.is_physical else "frames"

    @property
    def msd_unit(self) -> str:
        return f"{self.length_unit}²"

    @property
    def d_unit(self) -> str:
        time = "s" if self.is_physical else "frame"
        return f"{self.msd_unit}/{time}^α"

    def describe(self) -> str:
        """One-line summary. ASCII-only — this is printed to the console, and
        a Windows cp1252 terminal cannot encode 'σ' or 'α'."""
        if not self.is_physical:
            return "uncalibrated (px, frames; sigma=0)"
        return (
            f"{self.calibration.pixel_size_nm:g} nm/px, "
            f"{self.calibration.frame_interval_s:g} s/frame, "
            f"sigma={self.sigma_nm:g} nm"
        )
