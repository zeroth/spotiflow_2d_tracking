"""MSD / motility analysis (ported from scripts/bulk_track_analysis.py)."""

from ..base import register
from .analysis import MsdAnalysis

register(MsdAnalysis())

__all__ = ["MsdAnalysis"]
