"""Visualization utilities for probabilistic unmixing."""

import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, n_spectra=8):
    """
    Visualize the main elements of an UnmixingData object.

    Shows:
    1. representative measured spectra;
    2. measured concentration / fraction information;
    3. available spectral knowledge.

    Parameters
    ----------
    data : UnmixingData
        Dataset to visualize.

    n_spectra : int, default=8
        Maximum number of measured spectra to display.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    # --------------------------------------------------------
    # 1. Measured spectra
    # --------------------------------------------------------

    ax = axes[0]

    indices = np.linspace(
        0,
        data.n_locations - 1,
        min(n_spectra, data.n_locations),
        dtype=int,
    )

    for i in indices:
        ax.plot(
            data.E,
            data.Y[i],
            alpha=0.7,
        )

    ax.set_xlabel("Spectral coordinate")
    ax.set_ylabel("Intensity")
    ax.set_title("Measured spectra")

    # --------------------------------------------------------
    # 2. Measured concentrations / fractions
    # --------------------------------------------------------

    ax = axes[1]

    for k in range(data.n_components):
        ax.plot(
            data.W_mean[:, k],
            label=f"Component {k + 1}",
        )

    ax.set_xlabel("Measurement location")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Measured composition / fractions")
    ax.legend()

    # --------------------------------------------------------
    # 3. Available spectral knowledge
    # --------------------------------------------------------

    ax = axes[2]

    if data.S_fixed is not None:

        for k in range(data.n_components):
            ax.plot(
                data.E,
                data.S_fixed[k],
                label=f"Component {k + 1}",
            )

        ax.set_title("Level 0: fixed spectra")

    elif data.S_library is not None:

        for k in range(data.n_components):

            mean_spectrum = data.S_library[k].mean(axis=0)

            ax.plot(
                data.E,
                mean_spectrum,
                label=f"Component {k + 1}",
            )

        ax.set_title("Level 2: library mean")

    elif data.theta_mean is not None:

        ax.text(
            0.5,
            0.5,
            "Level 1 parametric\nspectral prior available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title("Level 1 spectral information")

    else:

        ax.text(
            0.5,
            0.5,
            "No spectral knowledge\nattached yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title("Spectral information")

    ax.set_xlabel("Spectral coordinate")
    ax.set_ylabel("Intensity")

    if (
        data.S_fixed is not None
        or data.S_library is not None
    ):
        ax.legend()

    fig.tight_layout()

    return fig


def plot_result(data, result):
    """
    Visualize the main outputs of an unmixing analysis.

    Shows:
    1. inferred concentrations / fractions;
    2. inferred endmember spectra;
    3. measured versus reconstructed spectra.

    Parameters
    ----------
    data : UnmixingData
        Input dataset.

    result : UnmixingResult
        Output of probabilistic unmixing.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    # --------------------------------------------------------
    # 1. Inferred concentrations
    # --------------------------------------------------------

    ax = axes[0]

    for k in range(result.n_components):
        ax.plot(
            result.W_mean[:, k],
            label=f"Component {k + 1}",
        )

    ax.set_xlabel("Measurement location")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Level {result.level}: inferred fractions"
    )
    ax.legend()

    # --------------------------------------------------------
    # 2. Inferred endmember spectra
    # --------------------------------------------------------

    ax = axes[1]

    for k in range(result.n_components):

        ax.plot(
            data.E,
            result.S_mean[k],
            label=f"Component {k + 1}",
        )

        if np.any(result.S_sd[k] > 0):

            ax.fill_between(
                data.E,
                result.S_mean[k] - 2.0 * result.S_sd[k],
                result.S_mean[k] + 2.0 * result.S_sd[k],
                alpha=0.15,
            )

    ax.set_xlabel("Spectral coordinate")
    ax.set_ylabel("Intensity")
    ax.set_title("Recovered endmembers")
    ax.legend()

    # --------------------------------------------------------
    # 3. Reconstruction
    # --------------------------------------------------------

    ax = axes[2]

    i = 0

    ax.plot(
        data.E,
        data.Y[i],
        label="Measured",
    )

    ax.plot(
        data.E,
        result.Y_pred[i],
        label="Reconstructed",
    )

    ax.set_xlabel("Spectral coordinate")
    ax.set_ylabel("Intensity")
    ax.set_title("Example reconstruction")
    ax.legend()

    fig.tight_layout()

    return fig
