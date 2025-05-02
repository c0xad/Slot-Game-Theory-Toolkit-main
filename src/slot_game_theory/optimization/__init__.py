# -*- coding: utf-8 -*-
"""
Modules for optimizing slot game design parameters.

Includes:
- Defining objective functions (e.g., operator profit, player engagement).
- Defining constraints (regulatory, design).
- Optimization algorithms (e.g., bi-level solvers, evolutionary algorithms).
"""

from . import objective
from . import constraints
from . import solver

__all__ = [
    "objective",
    "constraints",
    "solver",
]