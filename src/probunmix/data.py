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
        Mean measured concentrations / mixture fractions,
        shape (N, K).

    W_sigma
        Uncertainty in concentrations / mixture fractions,
        shape (N, K).

    sigma_y
        Spectral measurement uncertainty.
        May be a scalar or an array.

    Optional spectral information
    -----------------------------
    S_fixed
        Fixed endmember spectra for Level 0,
        shape (K, M).

    theta_mean
        Mean spectral parameters for Level 1,
        shape (K, P).

    theta_sigma
        Spectral parameter uncertainties for Level 1,
        shape (K, P).

    spectral_model
        Name or model object used for Level 1.

    S_library
        Ensemble of plausible endmember spectra for Level 2,
        shape (K, R, M).

    Optional synthetic truth
    ------------------------
    S_true
        True endmember spectra,
        shape (K, M).

    W_true
        True concentrations / mixture fractions,
        shape (N, K).

    Y_clean
        Noise-free mixture spectra,
        shape (N, M).

    Notes
    -----
    S_true, W_true, and Y_clean are intended only for synthetic
    benchmarks and evaluation. They are not required for experimental
    datasets and should not be used by inference.
    """

    # ------------------------------------------------------------
    # Core observed data
    # ------------------------------------------------------------

    E: np.ndarray
    Y: np.ndarray
    W_mean: np.ndarray
    W_sigma: np.ndarray
    sigma_y: Any

    # ------------------------------------------------------------
    # Level 0 spectral knowledge
    # ------------------------------------------------------------

    S_fixed: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Level 1 spectral knowledge
    # ------------------------------------------------------------

    theta_mean: Optional[np.ndarray] = None
    theta_sigma: Optional[np.ndarray] = None
    spectral_model: Optional[Any] = None

    # ------------------------------------------------------------
    # Level 2 spectral knowledge
    # ------------------------------------------------------------

    S_library: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Synthetic truth
    # ------------------------------------------------------------

    S_true: Optional[np.ndarray] = None
    W_true: Optional[np.ndarray] = None
    Y_clean: Optional[np.ndarray] = None

    # ============================================================
    # Basic dataset properties
    # ============================================================

    @property
    def n_locations(self) -> int:
        """Number of measurement locations."""
        return self.Y.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of spectral channels."""
        return self.Y.shape[1]

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return self.W_mean.shape[1]

    # ============================================================
    # Spectral knowledge setters
    # ============================================================

    def set_level0(self, S_fixed):
        """
        Supply fixed endmember spectra for Level 0.

        Parameters
        ----------
        S_fixed : array-like, shape (K, M)
            Fixed endmember spectra, where K is the number of
            components and M is the number of spectral channels.

        Returns
        -------
        UnmixingData
            The current data object.
        """

        S_fixed = np.asarray(S_fixed, dtype=float)

        expected_shape = (
            self.n_components,
            self.n_channels,
        )

        if S_fixed.shape != expected_shape:
            raise ValueError(
                f"S_fixed must have shape {expected_shape}, "
                f"but received {S_fixed.shape}."
            )

        self.S_fixed = S_fixed

        return self

    def set_level1(
        self,
        model,
        theta_mean,
        theta_sigma,
    ):
        """
        Supply a parametric spectral prior for Level 1.

        Parameters
        ----------
        model
            Spectral model name or model object.

        theta_mean : array-like, shape (K, P)
            Mean spectral parameters.

        theta_sigma : array-like, shape (K, P)
            Spectral parameter uncertainties.

        Returns
        -------
        UnmixingData
            The current data object.
        """

        theta_mean = np.asarray(theta_mean, dtype=float)
        theta_sigma = np.asarray(theta_sigma, dtype=float)

        if theta_mean.ndim != 2:
            raise ValueError(
                "theta_mean must have shape "
                "(n_components, n_parameters)."
            )

        if theta_mean.shape[0] != self.n_components:
            raise ValueError(
                "theta_mean must contain one row per component."
            )

        if theta_sigma.shape != theta_mean.shape:
            raise ValueError(
                "theta_sigma must have the same shape as theta_mean."
            )

        if np.any(theta_sigma <= 0):
            raise ValueError(
                "All Level 1 parameter uncertainties must be positive."
            )

        self.spectral_model = model
        self.theta_mean = theta_mean
        self.theta_sigma = theta_sigma

        return self

    def set_level2(self, S_library):
        """
        Supply a functional spectral library for Level 2.

        Parameters
        ----------
        S_library : array-like, shape (K, R, M)
            Ensemble of plausible spectra for each component.

            K = number of components
            R = number of library realizations
            M = number of spectral channels

        Returns
        -------
        UnmixingData
            The current data object.
        """

        S_library = np.asarray(S_library, dtype=float)

        if S_library.ndim != 3:
            raise ValueError(
                "S_library must have shape "
                "(n_components, n_library, n_channels)."
            )

        if S_library.shape[0] != self.n_components:
            raise ValueError(
                "The first dimension of S_library must equal "
                "the number of components."
            )

        if S_library.shape[2] != self.n_channels:
            raise ValueError(
                "The last dimension of S_library must equal "
                "the number of spectral channels."
            )

        self.S_library = S_library

        return self
