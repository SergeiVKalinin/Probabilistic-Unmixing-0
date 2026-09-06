"""Probabilistic Unmixing.

Tools for probabilistic spectral unmixing with uncertain
concentrations and progressively richer spectral knowledge.
"""

from .data import UnmixingData
from .results import UnmixingResult
from .synthetic import generate_example
from .io import from_arrays
from .fit import fit

from .plotting import (
    plot_data,
    plot_result,
    plot_level_comparison,
)

from .diagnostics import (
    data_summary,
    print_data_summary,
    validate_data,
)


__version__ = "0.1.0"

__all__ = [
    "UnmixingData",
    "UnmixingResult",
    "generate_example",
    "from_arrays",
    "fit",
    "plot_data",
    "plot_result",
    "plot_level_comparison",
    "data_summary",
    "print_data_summary",
    "validate_data",
]
