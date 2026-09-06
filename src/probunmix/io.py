"""Input utilities for experimental probabilistic-unmixing data."""

import numpy as np

from .data import UnmixingData


def from_arrays(
    E,
    Y,
    W_mean,
    W_sigma,
    sigma_y,
):
    """
    Create an UnmixingData object from experimental arrays.

    For the current implementation, measured composition and
    mixture fraction are treated as the same quantity.

    Parameters
    ----------
    E : array-like, shape (M,)
        Spectral coordinate.

    Y : array-like, shape (N, M)
        Measured spectra.

    W_mean : array-like, shape (N, K)
        Mean measured concentrations / mixture fractions.

    W_sigma : array-like or scalar
        Uncertainty in the measured concentrations.
        May be a scalar or an array with shape (N, K).

    sigma_y : array-like or scalar
        Spectral measurement uncertainty.
        May be a scalar or an array compatible with Y.

    Returns
    -------
    UnmixingData
        Experimental dataset ready for probabilistic unmixing.
    """

    E = np.asarray(E, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W_mean = np.asarray(W_mean, dtype=float)

    # ------------------------------------------------------------
    # Basic dimensional checks
    # ------------------------------------------------------------

    if E.ndim != 1:
        raise ValueError("E must be a one-dimensional spectral axis.")

    if Y.ndim != 2:
        raise ValueError(
            "Y must have shape (n_locations, n_channels)."
        )

    if W_mean.ndim != 2:
        raise ValueError(
            "W_mean must have shape (n_locations, n_components)."
        )

    if Y.shape[1] != E.size:
        raise ValueError(
            "The number of columns in Y must equal len(E)."
        )

    if W_mean.shape[0] != Y.shape[0]:
        raise ValueError(
            "Y and W_mean must contain the same number of locations."
        )

    # ------------------------------------------------------------
    # Concentration uncertainty
    # ------------------------------------------------------------

    W_sigma = np.asarray(W_sigma, dtype=float)

    if W_sigma.ndim == 0:
        W_sigma = np.full_like(W_mean, float(W_sigma))

    if W_sigma.shape != W_mean.shape:
        raise ValueError(
            "W_sigma must be a scalar or have the same shape as W_mean."
        )

    if np.any(W_sigma <= 0):
        raise ValueError(
            "All concentration uncertainties must be positive."
        )

    # ------------------------------------------------------------
    # Spectral measurement uncertainty
    # ------------------------------------------------------------

    sigma_y = np.asarray(sigma_y, dtype=float)

    if sigma_y.ndim == 0:
        sigma_y = np.full_like(Y, float(sigma_y))

    try:
        sigma_y = np.broadcast_to(sigma_y, Y.shape).copy()
    except ValueError as exc:
        raise ValueError(
            "sigma_y must be a scalar or broadcastable to the shape of Y."
        ) from exc

    if np.any(sigma_y <= 0):
        raise ValueError(
            "All spectral uncertainties must be positive."
        )

    # ------------------------------------------------------------
    # Basic composition checks
    # ------------------------------------------------------------

    if np.any(W_mean < 0):
        raise ValueError(
            "W_mean contains negative concentrations/fractions."
        )

    row_sums = W_mean.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError(
            "Every location must have a positive total concentration."
        )

    # Normalize small closure errors.
    W_mean = W_mean / row_sums[:, None]

    # ------------------------------------------------------------
    # Create the universal data object
    # ------------------------------------------------------------

    return UnmixingData(
        E=E,
        Y=Y,
        W_mean=W_mean,
        W_sigma=W_sigma,
        sigma_y=sigma_y,
    )
