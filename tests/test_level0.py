"""Tests for the basic Level 0 workflow."""

import numpy as np

import probunmix as pu


def test_generate_example():
    """Synthetic generator should create a valid common dataset."""

    data = pu.generate_example(
        n_locations=20,
        n_channels=100,
        n_library=20,
        seed=1,
    )

    assert data.Y.shape == (20, 100)
    assert data.W_mean.shape == (20, 4)
    assert data.W_sigma.shape == (20, 4)

    assert data.S_fixed.shape == (4, 100)
    assert data.theta_mean.shape[0] == 4
    assert data.S_library.shape == (4, 20, 100)

    # Concentrations / fractions must lie on the simplex.
    assert np.all(data.W_mean >= 0.0)

    np.testing.assert_allclose(
        data.W_mean.sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_level0_fit():
    """Level 0 should run and return the common result structure."""

    data = pu.generate_example(
        n_locations=20,
        n_channels=100,
        n_library=20,
        seed=2,
    )

    result = pu.fit(
        data,
        level=0,
        n_draws=200,
        seed=3,
    )

    assert result.level == 0

    assert result.W_mean.shape == (20, 4)
    assert result.W_sd.shape == (20, 4)

    assert result.S_mean.shape == (4, 100)
    assert result.S_sd.shape == (4, 100)

    assert result.Y_pred.shape == (20, 100)
    assert result.residuals.shape == (20, 100)

    # Posterior concentration means must remain on the simplex.
    np.testing.assert_allclose(
        result.W_mean.sum(axis=1),
        1.0,
        atol=1e-10,
    )

    # Level 0 assumes exact fixed endmembers.
    np.testing.assert_allclose(
        result.S_mean,
        data.S_fixed,
    )

    np.testing.assert_allclose(
        result.S_sd,
        0.0,
    )

    # Synthetic data provide truth-based evaluation metrics.
    assert "concentration_rmse" in result.metrics
    assert "spectral_rmse" in result.metrics
    assert "reconstruction_rmse" in result.metrics

    # Because Level 0 uses the exact synthetic endmembers,
    # spectral RMSE against truth should be zero.
    np.testing.assert_allclose(
        result.metrics["spectral_rmse"],
        0.0,
        atol=1e-14,
    )
