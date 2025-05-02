# -*- coding: utf-8 -*-
"""
Modules for modeling player behavior.

Includes:
- Prospect theory value functions and probability weighting.
- Player decision rules (e.g., stop-loss, win-goal, playtime targets).
"""

from . import prospect_theory
from . import decision_rules

__all__ = [
    "prospect_theory",
    "decision_rules",
]