# -*- coding: utf-8 -*-
"""
General utility functions used across the package.
"""

import numpy as np
import random
import time
from typing import Any, Dict, List

def set_random_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If using other libraries with random elements (e.g., TensorFlow, PyTorch, CuPy), set their seeds too.
    # try:
    #     import cupy as cp
    #     cp.random.seed(seed)
    # except ImportError:
    #     pass
    print(f"Random seed set to: {seed}")


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Formats a dictionary of metrics into a readable string."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            # Format floats nicely
            if abs(value) > 1000 or (abs(value) < 0.001 and value != 0):
                formatted_value = f"{value:.4e}" # Scientific notation for very large/small
            else:
                formatted_value = f"{value:.4f}" # Standard decimal format
        else:
            formatted_value = str(value)
        lines.append(f"- {key.replace('_', ' ').title()}: {formatted_value}")
    return "\n".join(lines)


class Timer:
    """A simple context manager for timing code blocks."""
    def __init__(self, name: str = "Execution"):
        self.name = name
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed = end_time - self._start_time
        print(f"[{self.name}] Elapsed time: {elapsed:.4f} seconds")


# Add other common utilities as needed:
# - Data loading/saving functions (CSV, JSON, Parquet)
# - Configuration file parsers
# - Validation functions for inputs
# - Math helpers (e.g., specific statistical functions)