"""Dataset diagnostics for probabilistic unmixing."""

import numpy as np


def data_summary(data):
    """
    Return a compact summary of an UnmixingData object.

    Works for both synthetic and experimental datasets.
    """

    row_sums = data.W_mean.sum(axis=1)

    summary = {
        "n_locations": data.n_locations,
        "n_channels": data.n_channels,
        "n_components": data.n_components,

        "spectral_axis_min": float(np.min(data.E)),
        "spectral_axis_max": float(np.max(data.E)),

        "spectra_min": float(np.min(data.Y)),
        "spectra_max": float(np.max(data.Y)),

        "composition_sum_min": float(np.min(row_sums)),
        "composition_sum_max": float(np.max(row_sums)),
        "composition_sum_mean": float(np.mean(row_sums)),

        "level0_available": data.S_fixed is not None,

        "level1_available": (
            data.theta_mean is not None
            and data.theta_sigma is not None
            and data.spectral_model is not None
        ),

        "level2_available": data.S_library is not None,

        "synthetic_truth_available": (
            data.S_true is not None
            and data.W_true is not None
        ),
    }

    return summary


def print_data_summary(data):
    """
    Print a human-readable summary of an UnmixingData object.
    """

    s = data_summary(data)

    print("Probabilistic Unmixing Dataset")
    print("=" * 34)

    print()
    print("Dimensions")
    print("----------")
    print(f"Locations:          {s['n_locations']}")
    print(f"Spectral channels:  {s['n_channels']}")
    print(f"Components:         {s['n_components']}")

    print()
    print("Spectral data")
    print("-------------")
    print(
        "Spectral axis:      "
        f"{s['spectral_axis_min']:.4g} "
        f"to {s['spectral_axis_max']:.4g}"
    )
    print(
        "Measured intensity: "
        f"{s['spectra_min']:.4g} "
        f"to {s['spectra_max']:.4g}"
    )

    print()
    print("Composition / mixture fractions")
    print("-------------------------------")
    print(
        "Row-sum range:      "
        f"{s['composition_sum_min']:.6f} "
        f"to {s['composition_sum_max']:.6f}"
    )
    print(
        "Mean row sum:       "
        f"{s['composition_sum_mean']:.6f}"
    )

    print()
    print("Available spectral knowledge")
    print("----------------------------")
    print(
        "Level 0 fixed spectra:      "
        f"{'YES' if s['level0_available'] else 'NO'}"
    )
    print(
        "Level 1 parametric prior:   "
        f"{'YES' if s['level1_available'] else 'NO'}"
    )
    print(
        "Level 2 functional library: "
        f"{'YES' if s['level2_available'] else 'NO'}"
    )

    print()
    print("Evaluation information")
    print("----------------------")
    print(
        "Synthetic truth:            "
        f"{'YES' if s['synthetic_truth_available'] else 'NO'}"
    )


def validate_data(data):
    """
    Perform basic validation of an UnmixingData object.

    Raises
    ------
    ValueError
        If a basic structural inconsistency is found.

    Returns
    -------
    bool
        True when all checks pass.
    """

    # --------------------------------------------------------
    # Core dimensions
    # --------------------------------------------------------

    if data.E.ndim != 1:
        raise ValueError(
            "E must be one-dimensional."
        )

    if data.Y.ndim != 2:
        raise ValueError(
            "Y must have shape (n_locations, n_channels)."
        )

    if data.W_mean.ndim != 2:
        raise ValueError(
            "W_mean must have shape "
            "(n_locations, n_components)."
        )

    if data.W_sigma.shape != data.W_mean.shape:
        raise ValueError(
            "W_sigma must have the same shape as W_mean."
        )

    if data.Y.shape[0] != data.W_mean.shape[0]:
        raise ValueError(
            "Y and W_mean must have the same number "
            "of measurement locations."
        )

    if data.Y.shape[1] != data.E.size:
        raise ValueError(
            "The number of spectral channels in Y "
            "must equal len(E)."
        )

    # --------------------------------------------------------
    # Finite values
    # --------------------------------------------------------

    for name, array in [
        ("E", data.E),
        ("Y", data.Y),
        ("W_mean", data.W_mean),
        ("W_sigma", data.W_sigma),
        ("sigma_y", data.sigma_y),
    ]:
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # --------------------------------------------------------
    # Composition simplex
    # --------------------------------------------------------

    if np.any(data.W_mean < 0):
        raise ValueError(
            "W_mean contains negative fractions."
        )

    if np.any(data.W_mean > 1):
        raise ValueError(
            "W_mean contains fractions greater than one."
        )

    np.testing.assert_allclose(
        data.W_mean.sum(axis=1),
        1.0,
        atol=1e-8,
        err_msg=(
            "Each row of W_mean must sum to one."
        ),
    )

    # --------------------------------------------------------
    # Uncertainties
    # --------------------------------------------------------

    if np.any(data.W_sigma <= 0):
        raise ValueError(
            "W_sigma must be strictly positive."
        )

    if np.any(np.asarray(data.sigma_y) <= 0):
        raise ValueError(
            "sigma_y must be strictly positive."
        )

    # --------------------------------------------------------
    # Optional Level 0 information
    # --------------------------------------------------------

    if data.S_fixed is not None:

        expected = (
            data.n_components,
            data.n_channels,
        )

        if data.S_fixed.shape != expected:
            raise ValueError(
                f"S_fixed must have shape {expected}."
            )

    # --------------------------------------------------------
    # Optional Level 2 information
    # --------------------------------------------------------

    if data.S_library is not None:

        if data.S_library.ndim != 3:
            raise ValueError(
                "S_library must have three dimensions."
            )

        if data.S_library.shape[0] != data.n_components:
            raise ValueError(
                "S_library has the wrong number of components."
            )

        if data.S_library.shape[2] != data.n_channels:
            raise ValueError(
                "S_library has the wrong number of spectral channels."
            )

    return True

def uncertainty_summary(data):
    """
    Summarize component-specific concentration and spectral uncertainty.

    Returns
    -------
    dict
        Dictionary containing:

        concentration_sd
            Mean supplied concentration uncertainty for each component.

        level1_theta_sigma
            Level 1 parameter-prior standard deviations, if available.

        level2_relative_spectral_uncertainty
            Dimensionless Level 2 functional uncertainty for each
            component, computed as

                RMS(pointwise library SD)
                -------------------------
                RMS(library mean spectrum)

            This allows spectral uncertainty to be compared between
            components even when their spectral intensities differ.
    """

    summary = {}

    # ========================================================
    # Concentration uncertainty
    # ========================================================

    summary["concentration_sd"] = np.mean(
        data.W_sigma,
        axis=0,
    )

    # ========================================================
    # Level 1 parameter uncertainty
    # ========================================================

    if data.theta_sigma is not None:

        summary["level1_theta_sigma"] = (
            np.asarray(
                data.theta_sigma,
                dtype=float,
            ).copy()
        )

    else:

        summary["level1_theta_sigma"] = None

    # ========================================================
    # Level 2 functional spectral uncertainty
    # ========================================================

    if data.S_library is not None:

        library = np.asarray(
            data.S_library,
            dtype=float,
        )

        library_mean = np.mean(
            library,
            axis=1,
        )

        library_sd = np.std(
            library,
            axis=1,
            ddof=1,
        )

        rms_sd = np.sqrt(
            np.mean(
                library_sd ** 2,
                axis=1,
            )
        )

        rms_mean = np.sqrt(
            np.mean(
                library_mean ** 2,
                axis=1,
            )
        )

        relative_uncertainty = (
            rms_sd
            / np.maximum(
                rms_mean,
                1e-15,
            )
        )

        summary[
            "level2_relative_spectral_uncertainty"
        ] = relative_uncertainty

    else:

        summary[
            "level2_relative_spectral_uncertainty"
        ] = None

    return summary

