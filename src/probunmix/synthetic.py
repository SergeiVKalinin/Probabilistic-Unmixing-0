"""Synthetic benchmark data for probabilistic unmixing.

The generator creates one physical mixture experiment together with
different forms of prior spectral knowledge for Levels 0, 1, and 2.

For the current implementation:

    composition = mixture fraction

Uncertainty can be specified independently for each component.
"""

import numpy as np

from .data import UnmixingData


# ============================================================
# Basic spectral functions
# ============================================================

def _trapezoid(y, x, axis=-1):
    """Compatibility wrapper for trapezoidal integration."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(
            y,
            x,
            axis=axis,
        )

    return np.trapz(
        y,
        x,
        axis=axis,
    )


def _gaussian(E, mu, sigma):
    """Normalized Gaussian line shape."""

    g = np.exp(
        -0.5
        * ((E - mu) / sigma) ** 2
    )

    return (
        g
        / _trapezoid(
            g,
            E,
        )
    )


def _two_peak_spectrum(
    E,
    r,
    mu1,
    sigma1,
    mu2,
    sigma2,
):
    """
    Two-Gaussian spectrum with relative integrated area r.
    """

    s = (
        r
        * _gaussian(
            E,
            mu1,
            sigma1,
        )
        +
        (1.0 - r)
        * _gaussian(
            E,
            mu2,
            sigma2,
        )
    )

    return (
        s
        / _trapezoid(
            s,
            E,
        )
    )


# ============================================================
# Uncertainty controls
# ============================================================

def _component_vector(
    value,
    n_components,
    name,
):
    """
    Convert a scalar or component-specific sequence to shape (K,).

    Parameters
    ----------
    value
        Either a positive scalar or an array-like object of length K.

    n_components
        Number of components.

    name
        Parameter name used in error messages.

    Returns
    -------
    ndarray, shape (K,)
        Positive component-specific values.
    """

    value = np.asarray(
        value,
        dtype=float,
    )

    if value.ndim == 0:

        result = np.full(
            n_components,
            float(value),
        )

    elif value.ndim == 1 and value.size == n_components:

        result = value.copy()

    else:

        raise ValueError(
            f"{name} must be either a scalar or "
            f"a length-{n_components} sequence."
        )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if np.any(
        result <= 0.0
    ):
        raise ValueError(
            f"All values of {name} must be positive."
        )

    return result


# ============================================================
# Public synthetic generator
# ============================================================

def generate_example(
    n_locations=100,
    n_channels=240,
    n_library=200,
    concentration_uncertainty=0.03,
    spectral_uncertainty=1.0,
    seed=42,
):
    """
    Generate a controlled synthetic probabilistic-unmixing dataset.

    The same physical truth and measured spectra are used for
    Level 0, Level 1, and Level 2.

    Parameters
    ----------
    n_locations : int, default=100
        Number of mixture measurements.

    n_channels : int, default=240
        Number of spectral channels.

    n_library : int, default=200
        Number of Level 2 library spectra per component.

    concentration_uncertainty : float or array-like, default=0.03
        Nominal uncertainty in measured concentrations / mixture
        fractions.

        A scalar applies the same uncertainty to every component.

        A sequence of length K specifies a different uncertainty
        for each component.

        Example:

            [0.01, 0.03, 0.08, 0.15]

    spectral_uncertainty : float or array-like, default=1.0
        Relative spectral-prior uncertainty for each component.

        A value of 1.0 reproduces the canonical uncertainty scale.

        Values below 1.0 produce more tightly known spectra.
        Values above 1.0 produce more uncertain spectra.

        A scalar applies the same scale to every component.

        A sequence of length K specifies a different spectral
        uncertainty for each component.

        Example:

            [0.25, 0.75, 1.5, 3.0]

        This scale affects both:

        - Level 1 parameter-prior widths
        - Level 2 spectral-library spread

        Level 0 remains exact by definition.

    seed : int, default=42
        Random seed.

    Returns
    -------
    UnmixingData
        Dataset containing observations, all three forms of spectral
        knowledge, and synthetic ground truth.

    Notes
    -----
    For a fixed seed, changing concentration_uncertainty or
    spectral_uncertainty does NOT change:

        S_true
        W_true
        Y_clean
        Y
        sigma_y

    It changes only the uncertain information supplied to inference.
    """

    rng = np.random.default_rng(
        seed
    )

    # ========================================================
    # Spectral coordinate
    # ========================================================

    E = np.linspace(
        0.0,
        12.0,
        n_channels,
    )

    # ========================================================
    # True endmember parameters
    #
    # columns:
    #
    # [relative_area, mu1, sigma1, mu2, sigma2]
    # ========================================================

    theta_true = np.array(
        [
            [0.68, 1.40, 0.22, 6.20, 0.65],
            [0.55, 2.85, 0.32, 8.10, 0.28],
            [0.60, 4.50, 0.72, 9.70, 0.42],
            [0.38, 2.05, 0.60, 7.00, 0.48],
        ],
        dtype=float,
    )

    n_components = (
        theta_true.shape[0]
    )

    # ========================================================
    # Convert uncertainty controls to component-specific vectors
    # ========================================================

    concentration_uncertainty = (
        _component_vector(
            concentration_uncertainty,
            n_components,
            "concentration_uncertainty",
        )
    )

    spectral_uncertainty = (
        _component_vector(
            spectral_uncertainty,
            n_components,
            "spectral_uncertainty",
        )
    )

    # ========================================================
    # True endmember spectra
    # ========================================================

    S_true = np.array(
        [
            _two_peak_spectrum(
                E,
                *theta,
            )
            for theta in theta_true
        ]
    )

    # ========================================================
    # True concentrations / mixture fractions
    #
    # For the current model:
    #
    # concentration = mixture fraction
    # ========================================================

    W_true = rng.dirichlet(
        alpha=np.ones(
            n_components
        ),
        size=n_locations,
    )

    # ========================================================
    # Clean physical mixture spectra
    # ========================================================

    Y_clean = (
        W_true
        @ S_true
    )

    # ========================================================
    # Spectral measurement noise
    #
    # This remains independent of the concentration and spectral
    # prior uncertainty controls.
    # ========================================================

    sigma_y = (
        0.005
        +
        0.02
        * np.sqrt(
            np.maximum(
                Y_clean,
                0.0,
            )
        )
    )

    Y = (
        Y_clean
        +
        rng.normal(
            loc=0.0,
            scale=sigma_y,
            size=Y_clean.shape,
        )
    )

    # ========================================================
    # Measured concentration information
    #
    # Different components may have different uncertainties.
    # ========================================================

    W_sigma = np.broadcast_to(
        concentration_uncertainty[
            None,
            :
        ],
        W_true.shape,
    ).copy()

    # Draw one standardized error realization.
    #
    # The component-specific uncertainty scales multiply these
    # standardized draws. Thus comparisons at fixed seed use the
    # same underlying random perturbation directions.
    concentration_error = rng.normal(
        loc=0.0,
        scale=1.0,
        size=W_true.shape,
    )

    W_mean = (
        W_true
        +
        concentration_error
        * W_sigma
    )

    # Keep measured concentrations on the simplex.
    W_mean = np.clip(
        W_mean,
        1e-6,
        None,
    )

    W_mean /= W_mean.sum(
        axis=1,
        keepdims=True,
    )

    # ========================================================
    # LEVEL 0
    #
    # Exact endmember spectra are supplied.
    #
    # Spectral uncertainty is zero by definition.
    # ========================================================

    S_fixed = (
        S_true.copy()
    )

    # ========================================================
    # LEVEL 1
    #
    # Parametric spectral prior.
    #
    # Prior centers remain fixed while the uncertainty width can
    # differ independently for each component.
    # ========================================================

    theta_mean = (
        theta_true.copy()
    )

    # Fixed prior bias.
    #
    # This does not depend on spectral_uncertainty.
    theta_mean[:, 0] += np.array(
        [
            0.04,
            -0.03,
            0.04,
            -0.04,
        ]
    )

    theta_mean[:, 1] += np.array(
        [
            0.10,
            -0.08,
            0.12,
            -0.10,
        ]
    )

    theta_mean[:, 3] += np.array(
        [
            -0.10,
            0.10,
            -0.08,
            0.12,
        ]
    )

    # Canonical Level 1 parameter uncertainty.
    #
    # [r, mu1, sigma1, mu2, sigma2]
    base_theta_sigma = np.array(
        [
            0.08,
            0.15,
            0.08,
            0.15,
            0.08,
        ]
    )

    theta_sigma = (
        spectral_uncertainty[
            :,
            None,
        ]
        * base_theta_sigma[
            None,
            :
        ]
    )

    spectral_model = (
        "two_gaussian"
    )

    # ========================================================
    # LEVEL 2
    #
    # Functional spectral libraries.
    #
    # Each component gets its own uncertainty multiplier.
    # ========================================================

    S_library = np.empty(
        (
            n_components,
            n_library,
            n_channels,
        )
    )

    # Canonical Level 2 perturbation scales.
    base_r_sd = 0.08
    base_mu_sd = 0.15
    base_log_width_sd = 0.18

    for k in range(
        n_components
    ):

        scale = (
            spectral_uncertainty[k]
        )

        for j in range(
            n_library
        ):

            theta = (
                theta_mean[k]
                .copy()
            )

            # ------------------------------------------------
            # Use standardized random variables and multiply
            # them by the component-specific spectral scale.
            # ------------------------------------------------

            theta[0] += (
                scale
                * base_r_sd
                * rng.normal()
            )

            theta[1] += (
                scale
                * base_mu_sd
                * rng.normal()
            )

            theta[2] *= np.exp(
                scale
                * base_log_width_sd
                * rng.normal()
            )

            theta[3] += (
                scale
                * base_mu_sd
                * rng.normal()
            )

            theta[4] *= np.exp(
                scale
                * base_log_width_sd
                * rng.normal()
            )

            # Keep the relative peak area physical.
            theta[0] = np.clip(
                theta[0],
                0.05,
                0.95,
            )

            S_library[
                k,
                j,
            ] = (
                _two_peak_spectrum(
                    E,
                    *theta,
                )
            )

    # ========================================================
    # Assemble universal data object
    # ========================================================

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
