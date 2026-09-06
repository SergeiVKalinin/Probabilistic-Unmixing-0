"""Tests for the basic Level 2 workflow."""

import numpy as np

import probunmix as pu


def test_level2_fit():
    """
    Level 2 should jointly infer concentrations and
    functional endmember spectra from a spectral library.
    """

    # Keep the problem small enough for a fast test.
    data = pu.generate_example(
        n_locations=6,
        n_channels=60,
        n_library=30,
        seed=31,
    )

    result = pu.fit(
        data,
        level=2,
        n_draws=100,
        seed=32,
        max_nfev=150,
    )

    # --------------------------------------------------------
    # Common result structure
    # --------------------------------------------------------

    assert result.level == 2

    assert result.W_mean.shape == (6, 4)
    assert result.W_sd.shape == (6, 4)

    assert result.S_mean.shape == (4, 60)
    assert result.S_sd.shape == (4, 60)

    assert result.Y_pred.shape == (6, 60)
    assert result.residuals.shape == (6, 60)

    # --------------------------------------------------------
    # Fractions must remain on the simplex
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(result.W_mean)
    )

    assert np.all(
        result.W_mean >= 0.0
    )

    assert np.all(
        result.W_mean <= 1.0
    )

    np.testing.assert_allclose(
        result.W_mean.sum(axis=1),
        1.0,
        atol=1e-10,
    )

    # --------------------------------------------------------
    # Functional spectra
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(result.S_mean)
    )

    assert np.all(
        np.isfinite(result.S_sd)
    )

    assert np.any(
        result.S_sd > 0.0
    )

    # Recovered spectra should be positive.
    assert np.all(
        result.S_mean > 0.0
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    required_metrics = [
        "reconstruction_rmse",
        "concentration_rmse",
        "spectral_rmse",
        "mean_concentration_sd",
        "mean_spectral_sd",
        "n_functional_parameters",
        "n_total_parameters",
    ]

    for name in required_metrics:
        assert name in result.metrics

    assert np.isfinite(
        result.metrics["reconstruction_rmse"]
    )

    assert np.isfinite(
        result.metrics["concentration_rmse"]
    )

    assert np.isfinite(
        result.metrics["spectral_rmse"]
    )

    # --------------------------------------------------------
    # Functional representation metadata
    # --------------------------------------------------------

    assert result.metadata["model"] == "functional_log_pca"

    assert len(
        result.metadata["ranks"]
    ) == 4

    assert all(
        rank >= 1
        for rank in result.metadata["ranks"]
    )

    assert (
        result.metrics["n_functional_parameters"]
        ==
        sum(result.metadata["ranks"])
    )
