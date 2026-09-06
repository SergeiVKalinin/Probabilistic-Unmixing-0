"""Tests for the basic Level 1 workflow."""

import numpy as np

import probunmix as pu


def test_level1_fit():
    """
    Level 1 should jointly infer concentrations and
    parametric endmember spectra.
    """

    # Keep the test deliberately small so that it runs quickly.
    data = pu.generate_example(
        n_locations=8,
        n_channels=60,
        n_library=10,
        seed=21,
    )

    result = pu.fit(
        data,
        level=1,
        n_draws=100,
        seed=22,
        max_nfev=150,
    )

    # --------------------------------------------------------
    # Common result structure
    # --------------------------------------------------------

    assert result.level == 1

    assert result.W_mean.shape == (8, 4)
    assert result.W_sd.shape == (8, 4)

    assert result.S_mean.shape == (4, 60)
    assert result.S_sd.shape == (4, 60)

    assert result.Y_pred.shape == (8, 60)
    assert result.residuals.shape == (8, 60)

    # --------------------------------------------------------
    # Fractions must remain on the simplex
    # --------------------------------------------------------

    assert np.all(result.W_mean >= 0.0)
    assert np.all(result.W_mean <= 1.0)

    np.testing.assert_allclose(
        result.W_mean.sum(axis=1),
        1.0,
        atol=1e-10,
    )

    # --------------------------------------------------------
    # Level 1 must contain spectral uncertainty
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

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    assert "reconstruction_rmse" in result.metrics
    assert "concentration_rmse" in result.metrics
    assert "spectral_rmse" in result.metrics
    assert "mean_concentration_sd" in result.metrics
    assert "mean_spectral_sd" in result.metrics

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
    # Level 1 is not Level 0:
    # endmember uncertainty should not be identically zero.
    # --------------------------------------------------------

    assert not np.allclose(
        result.S_sd,
        0.0,
    )
