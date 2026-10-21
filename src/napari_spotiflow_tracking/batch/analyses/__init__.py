"""Registered batch analyses. Phases 2–4 import their modules here to register them.

Analysis modules (and this package) must not import ``batch.runner`` or
``batch.validation`` — both of those import this package, so importing them
back from here would be a cycle.
"""

from . import base
from .base import (
    Analysis,
    enabled_analyses,
    get_analysis,
    measure_on_for,
    register,
    settings_for,
)
from . import msd  # noqa: F401  (registers the msd analysis)

__all__ = ["Analysis", "base", "enabled_analyses", "get_analysis", "measure_on_for", "register",
           "settings_for"]
