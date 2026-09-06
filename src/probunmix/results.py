"""Common result object for probabilistic unmixing."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class UnmixingResult:
    """
    Common output object for Level 0, Level 1, and Level 2 inference.

    Parameters
    ----------
    level
        Unmixing level: 0, 1, or 2.

    W_mean
        Posterior mean concentrations / mixture fractions,
        shape (N, K).

    W_sd
        Posterior standard deviation of concentrations,
        shape (N, K).

    S_mean
        Posterior mean endmember spectra,
        shape (K, M).

    S_sd
        Posterior pointwise standard deviation of endmember spectra,
        shape (K, M).

        For Level 0 this is identically zero because the
        endmember spectra are assumed known exactly.

    Y_pred
        Posterior mean reconstructed spectra,
        shape (N, M).

    residuals
        Observed minus reconstructed spectra,
        shape (N, M).

    metrics
        Dictionary containing numerical diagnostics and
        performance metrics.

    metadata
        Dictionary describing the inference method and settings.
    """

    level: int

    W_mean: np.ndarray
    W_sd: np.ndarray

    S_mean: np.ndarray
    S_sd: np.ndarray

    Y_pred: np.ndarray
    residuals: np.ndarray

    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional synthetic truth for evaluation/plotting.
    W_true: Optional[np.ndarray] = None
    S_true: Optional[np.ndarray] = None
    Y_clean: Optional[np.ndarray] = None

    @property
    def n_locations(self) -> int:
        """Number of measurement locations."""
        return self.W_mean.shape[0]

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return self.W_mean.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of spectral channels."""
        return self.S_mean.shape[1]
