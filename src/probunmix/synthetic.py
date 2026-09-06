"""Synthetic benchmark data for probabilistic unmixing."""

import numpy as np

from .data import UnmixingData


def _gaussian(E, mu, sigma):
    """Normalized Gaussian line shape."""
    g = np.exp(-0.5 * ((E - mu) / sigma) ** 2)
    return g / np.trapz(g, E)


def _two_peak_spectrum(E, r, mu1, sigma1, mu2, sigma2):
    """Two-Gaussian spectrum with relative integrated area r."""
    s = (
        r * _gaussian(E, mu1, sigma1)
        + (1.0 - r) * _gaussian(E, mu2, sigma2)
    )
    return s / np.trapz(s, E)


def generate_example(
    n_locations=100,
    n_channels=240,
    n_library=200,
    seed=42,
):
    """
    Generate the canonical dummy dataset.

    The same observed spectra and concentration information are used
    for Level 0, Level 1, and Level 2.

    Level 0:
        Exact endmember spectra are supplied.

    Level 1:
        A two-Gaussian spectral model and uncertain parameters
        are supplied.

    Level 2:
        An ensemble of plausible complete spectra is supplied.

    Returns
    -------
    UnmixingData
        Dataset containing observations, spectral information,
        and synthetic ground truth.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Spectral coordinate
    # ------------------------------------------------------------

    E = np.linspace(0.0, 12.0, n_channels)

    # ------------------------------------------------------------
    # True endmember parameters
    #
    # columns:
    # [relative_area, mu1, sigma1, mu2, sigma2]
    # ------------------------------------------------------------

    theta_true = np.array([
        [0.68, 1.40, 0.22, 6.20, 0.65],
        [0.55, 2.85, 0.32, 8.10, 0.28],
        [0.60, 4.50, 0.72, 9.70, 0.42],
        [0.38, 2.05, 0.60, 7.00, 0.48],
    ])

    n_components = theta_true.shape[0]

    S_true = np.array([
        _two_peak_spectrum(E, *theta)
        for theta in theta_true
    ])

    # ------------------------------------------------------------
    # True concentrations / mixture fractions
    #
    # For the current model:
    #
    # concentration = mixture fraction
    # ------------------------------------------------------------

    W_true = rng.dirichlet(
        alpha=np.ones(n_components),
        size=n_locations,
    )

    # ------------------------------------------------------------
    # Clean and noisy measured spectra
    # ------------------------------------------------------------

    Y_clean = W_true @ S_true

    # Simple heteroscedastic measurement noise.
    sigma_y = 0.005 + 0.02 * np.sqrt(np.maximum(Y_clean, 0.0))

    Y = Y_clean + rng.normal(
        loc=0.0,
        scale=sigma_y,
        size=Y_clean.shape,
    )

    # ------------------------------------------------------------
    # Measured concentration information
    # ------------------------------------------------------------

    W_sigma = np.full_like(W_true, 0.03)

    W_mean = W_true + rng.normal(
        loc=0.0,
        scale=W_sigma,
        size=W_true.shape,
    )

    # Keep concentrations on the simplex.
    W_mean = np.clip(W_mean, 1e-6, None)
    W_mean /= W_mean.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------
    # LEVEL 0
    #
    # Exact endmember spectra are known.
    # ------------------------------------------------------------

    S_fixed = S_true.copy()

    # ------------------------------------------------------------
    # LEVEL 1
    #
    # Approximate parametric spectral knowledge.
    # ------------------------------------------------------------

    theta_mean = theta_true.copy()

    # Deliberately shift the prior centers slightly away from truth.
    theta_mean[:, 0] += np.array([0.04, -0.03, 0.04, -0.04])
    theta_mean[:, 1] += np.array([0.10, -0.08, 0.12, -0.10])
    theta_mean[:, 3] += np.array([-0.10, 0.10, -0.08, 0.12])

    theta_sigma = np.tile(
        np.array([0.08, 0.15, 0.08, 0.15, 0.08]),
        (n_components, 1),
    )

    spectral_model = "two_gaussian"

    # ------------------------------------------------------------
    # LEVEL 2
    #
    # Ensemble of plausible complete spectra.
    # ------------------------------------------------------------

    S_library = np.empty(
        (n_components, n_library, n_channels)
    )

    for k in range(n_components):
        for j in range(n_library):

            theta = theta_mean[k].copy()

            theta[0] += rng.normal(0.0, 0.08)
            theta[1] += rng.normal(0.0, 0.15)
            theta[2] *= np.exp(rng.normal(0.0, 0.18))
            theta[3] += rng.normal(0.0, 0.15)
            theta[4] *= np.exp(rng.normal(0.0, 0.18))

            theta[0] = np.clip(theta[0], 0.05, 0.95)

            S_library[k, j] = _two_peak_spectrum(
                E,
                *theta,
            )

    # ------------------------------------------------------------
    # Assemble universal data object
    # ------------------------------------------------------------

    return UnmixingData(
        E=E,
        Y=Y,
        W_mean=W_mean,
        W_sigma=W_sigma,
        sigma_y=sigma_y,

        # Level 0
        S_fixed=S_fixed,

        # Level 1
        theta_mean=theta_mean,
        theta_sigma=theta_sigma,
        spectral_model=spectral_model,

        # Level 2
        S_library=S_library,

        # Synthetic truth: evaluation only
        S_true=S_true,
        W_true=W_true,
        Y_clean=Y_clean,
    )
