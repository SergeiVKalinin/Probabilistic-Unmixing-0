"""Tests for component-specific uncertainty controls."""

import numpy as np

import probunmix as pu


def test_component_specific_uncertainties():
    """Different components may have different uncertainty levels."""

    concentration_uncertainty = [
        0.01,
        0.02,
        0.08,
        0.15,
    ]

    spectral_uncertainty = [
        0.25,
        0.75,
        1.5,
        3.0,
    ]

    data = pu.generate_example(
        n_locations=20,
        n_channels=80,
        n_library=40,
        concentration_uncertainty=concentration_uncertainty,
        spectral_uncertainty=spectral_uncertainty,
        seed=123,
    )

    # --------------------------------------------------------
    # Concentration uncertainty
    # --------------------------------------------------------

    assert data.W_sigma.shape == (20, 4)

    np.testing.assert_allclose(
        data.W_sigma[0],
        concentration_uncertainty,
    )

    # Every location should use the same component-specific
    # uncertainty vector in this synthetic example.
    np.testing.assert_allclose(
        data.W_sigma,
        np.broadcast_to(
            concentration_uncertainty,
            (20, 4),
        ),
    )

    # --------------------------------------------------------
    # Level 1 spectral uncertainty
    # --------------------------------------------------------

    base_theta_sigma = np.array(
        [
            0.08,
            0.15,
            0.08,
            0.15,
            0.08,
        ]
    )

    expected_theta_sigma = (
        np.asarray(spectral_uncertainty)[:, None]
        * base_theta_sigma[None, :]
    )

    np.testing.assert_allclose(
        data.theta_sigma,
        expected_theta_sigma,
    )



def test_uncertainty_changes_do_not_change_physical_truth():
    """
    Different uncertainty assumptions at fixed seed should describe
    the same underlying physical experiment.
    """

    common = dict(
        n_locations=20,
        n_channels=80,
        n_library=40,
        seed=456,
    )

    narrow = pu.generate_example(
        **common,
        concentration_uncertainty=[
            0.01,
            0.01,
            0.01,
            0.01,
        ],
        spectral_uncertainty=[
            0.25,
            0.25,
            0.25,
            0.25,
        ],
    )

    heterogeneous = pu.generate_example(
        **common,
        concentration_uncertainty=[
            0.01,
            0.03,
            0.08,
            0.15,
        ],
        spectral_uncertainty=[
            0.25,
            0.75,
            1.5,
            3.0,
        ],
    )

    # --------------------------------------------------------
    # Physical system must be identical
    # --------------------------------------------------------

    np.testing.assert_allclose(
        narrow.S_true,
        heterogeneous.S_true,
    )

    np.testing.assert_allclose(
        narrow.W_true,
        heterogeneous.W_true,
    )

    np.testing.assert_allclose(
        narrow.Y_clean,
        heterogeneous.Y_clean,
    )

    np.testing.assert_allclose(
        narrow.Y,
        heterogeneous.Y,
    )

    np.testing.assert_allclose(
        narrow.sigma_y,
        heterogeneous.sigma_y,
    )

    # --------------------------------------------------------
    # Level 0 information is also identical
    # --------------------------------------------------------

    np.testing.assert_allclose(
        narrow.S_fixed,
        heterogeneous.S_fixed,
    )

    # --------------------------------------------------------
    # Uncertain information should differ
    # --------------------------------------------------------

    assert not np.allclose(
        narrow.W_sigma,
        heterogeneous.W_sigma,
    )

    assert not np.allclose(
        narrow.theta_sigma,
        heterogeneous.theta_sigma,
    )

    assert not np.allclose(
        narrow.S_library,
        heterogeneous.S_library,
    )



def test_scalar_uncertainty_still_supported():
    """The original scalar API should remain valid."""

    data = pu.generate_example(
        n_locations=10,
        n_channels=50,
        n_library=20,
        concentration_uncertainty=0.03,
        spectral_uncertainty=1.0,
        seed=789,
    )

    np.testing.assert_allclose(
        data.W_sigma,
        0.03,
    )

    np.testing.assert_allclose(
        data.theta_sigma,
        np.tile(
            np.array(
                [
                    0.08,
                    0.15,
                    0.08,
                    0.15,
                    0.08,
                ]
            ),
            (4, 1),
        ),
    )
