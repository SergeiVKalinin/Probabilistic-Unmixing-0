"""Integration test for the common Level 0/1/2 workflow."""

import numpy as np

import probunmix as pu


def test_same_dataset_all_levels():
    """
    The same UnmixingData object should run through
    Level 0, Level 1, and Level 2.
    """

    data = pu.generate_example(
        n_locations=6,
        n_channels=50,
        n_library=25,
        seed=100,
    )

    # --------------------------------------------------------
    # Run all three analyses on exactly the same dataset
    # --------------------------------------------------------

    r0 = pu.fit(
        data,
        level=0,
        n_draws=100,
        seed=101,
    )

    r1 = pu.fit(
        data,
        level=1,
        n_draws=100,
        seed=102,
        max_nfev=150,
    )

    r2 = pu.fit(
        data,
        level=2,
        n_draws=100,
        seed=103,
        max_nfev=150,
    )

    # --------------------------------------------------------
    # All levels must return the same common result structure
    # --------------------------------------------------------

    for result, level in [
        (r0, 0),
        (r1, 1),
        (r2, 2),
    ]:

        assert isinstance(
            result,
            pu.UnmixingResult,
        )

        assert result.level == level

        assert result.W_mean.shape == (
            data.n_locations,
            data.n_components,
        )

        assert result.S_mean.shape == (
            data.n_components,
            data.n_channels,
        )

        assert result.Y_pred.shape == data.Y.shape

        # Fractions stay on the simplex.
        np.testing.assert_allclose(
            result.W_mean.sum(axis=1),
            1.0,
            atol=1e-10,
        )

        assert np.all(
            np.isfinite(result.W_mean)
        )

        assert np.all(
            np.isfinite(result.S_mean)
        )

        assert np.isfinite(
            result.metrics["reconstruction_rmse"]
        )

    # --------------------------------------------------------
    # Expected hierarchy of spectral uncertainty
    # --------------------------------------------------------

    # Level 0: exact spectra
    np.testing.assert_allclose(
        r0.S_sd,
        0.0,
    )

    # Levels 1 and 2: uncertain spectra
    assert np.any(
        r1.S_sd > 0.0
    )

    assert np.any(
        r2.S_sd > 0.0
    )

    # --------------------------------------------------------
    # Comparison visualization should work
    # --------------------------------------------------------

    fig = pu.plot_level_comparison(
        data,
        r0,
        r1,
        r2,
    )

    assert fig is not None
