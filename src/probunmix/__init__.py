"""Probabilistic Unmixing.

Tools for probabilistic spectral unmixing with uncertain
concentrations and progressively richer spectral knowledge.
"""

from .data import UnmixingData
from .synthetic import generate_example
from .io import from_arrays


__version__ = "0.1.0"

__all__ = [
    "UnmixingData",
    "generate_example",
    "from_arrays",
]
