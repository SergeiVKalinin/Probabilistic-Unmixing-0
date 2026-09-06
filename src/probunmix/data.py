"""Core data structure for probabilistic unmixing."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class UnmixingData:
    """
    Universal input data object for probabilistic unmixing.

    For the current implementation, composition and mixture fraction
    are treated as the same quantity.

    Parameters
    ----------
    E
        Spectral coordinate, shape (M,).
    Y
        Observed spectra, shape (N, M).
    W_mean
        Mean measured concentrations/fractions, shape (N, K).
    W_sigma
        Uncertainty in concentrations/fractions, shape (N, K).
    sigma_y
        Spectral measurement uncertainty. May be a scalar or an array.

    Optional spectral information
    -----------------------------
    S_fixed
        Fixed endmember spectra for Level 0, shape (K, M).
    theta_mean
        Mean spectral parameters for Level 1.
    theta_sigma
        Spectral parameter uncertainties for Level 1.
    spectral_model
        Name or model object used for Level 1.
    S_library
        Ensemble of plausible endmember spectra for Level 2,
        shape (K, R, M).

    Optional synthetic truth
    ------------------------
    S_true
        True endmember spectra, used only for synthetic evaluation.
    W_true
        True concentrations/fractions, used only for synthetic evaluation.
    Y_clean
        Noise-free mixture spectra, used only for synthetic evaluation.
    """

    E: np.ndarray
    Y: np.ndarray
    W_mean: np.ndarray
    W_sigma: np.ndarray
    sigma_y: Any

    # Level 0 spectral knowledge
    S_fixed: Optional[np.ndarray] = None

    # Level 1 spectral knowledge
    theta_mean: Optional[np.ndarray] = None
    theta_sigma: Optional[np.ndarray] = None
    spectral_model: Optional[Any] = None

    # Level 2 spectral knowledge
    S_library: Optional[np.ndarray] = None

    # Synthetic truth: never required for experimental data
    S_true: Optional[np.ndarray] = None
    W_true: Optional[np.ndarray] = None
    Y_clean: Optional[np.ndarray] = None

    @property
    def n_locations(self) -> int:
        return self.Y.shape[0]

    @property
    def n_channels(self) -> int:
        return self.Y.shape[1]

    @property
    def n_components(self) -> int:
        return self.W_mean.shape[1]
