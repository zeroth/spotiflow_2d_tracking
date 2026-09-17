"""Registered batch analyses. Phases 2–4 import their modules here to register them."""

from . import base
from .base import Analysis, enabled_analyses, get_analysis, register, settings_for

__all__ = ["Analysis", "base", "enabled_analyses", "get_analysis", "register", "settings_for"]
