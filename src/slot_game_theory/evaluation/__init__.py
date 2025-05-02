# -*- coding: utf-8 -*-
"""
Modules for evaluating slot game metrics.

Includes:
- Symbolic evaluation for exact calculation in simple cases.
- Monte Carlo simulation for estimating metrics in complex games.
"""

from . import symbolic
from . import simulation

__all__ = [
    "symbolic",
    "simulation",
]