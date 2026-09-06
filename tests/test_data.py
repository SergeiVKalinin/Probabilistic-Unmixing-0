"""Tests for the common synthetic/experimental data interface."""

import numpy as np

import probunmix as pu


def test_from_arrays():
    """Experimental arrays should create a valid UnmixingData object."""

    rng = np.random.default_rng(10)

    n_locations = 12
    n_channels = 80
    n_components = 3

    E = np.linspace(0.0, 10.0, n_channels)

    Y = rng.random(
        (n_locations, n_channels)
    )

    W = rng.dirichlet(
        np.ones(n_components),
        size=n_locations,
    )

    data = pu.from_arrays(
        E=E,
        Y=Y,
        W_mean=W,
        W_sigma=0.03,
        sigma_y=0.01,
    )

    assert isinstance(data, pu.UnmixingData)

    assert data.E.shape == (n_channels,)
    assert data.Y.shape == (n_locations, n_channels)

    assert data.W_mean.shape == (
        n_locations,
        n_components,
    )

    assert data.W_sigma.shape == (
        n_locations,
        n_components,
    )

    assert data.sigma_y.shape == (
        n_locations,
        n_channels,
    )

    np.testing.assert_allclose(
        data.W_mean.sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_level0_setter():
    """Fixed reference spectra should attach to experimental data."""

    data = _make_small_dataset()

    S = np.ones(
        (data.n_components, data.n_channels)
    )

    returned = data.set_level0(S)

    assert returned is data

    np.testing.assert_allclose(
        data.S_fixed,
        S,
    )


def test_level1_setter():
    """Parametric spectral knowledge should attach correctly."""

    data = _make_small_dataset()

    theta_mean = np.ones(
        (data.n_components, 5)
    )

    theta_sigma = np.full(
        (data.n_components, 5),
        0.1,
    )

    returned = data.set_level1(
        model="two_gaussian",
        theta_mean=theta_mean,
        theta_sigma=theta_sigma,
    )

    assert returned is data
    assert data.spectral_model == "two_gaussian"

    np.testing.assert_allclose(
        data.theta_mean,
        theta_mean,
    )

    np.testing.assert_allclose(
        data.theta_sigma,
        theta_sigma,
    )


def test_level2_setter():
    """Functional spectral libraries should attach correctly."""

    data = _make_small_dataset()

    S_library = np.ones(
        (
            data.n_components,
            25,
            data.n_channels,
        )
    )

    returned = data.set_level2(
        S_library
    )

    assert returned is data

    assert data.S_library.shape == (
        data.n_components,
        25,
        data.n_channels,
    )


def _make_small_dataset():
    """Create a small experimental-style dataset for tests."""

    rng = np.random.default_rng(11)

    n_locations = 10
    n_channels = 50
    n_components = 3

    E = np.linspace(
        0.0,
        5.0,
        n_channels,
    )

    Y = rng.random(
        (n_locations, n_channels)
    )

    W = rng.dirichlet(
        np.ones(n_components),
        size=n_locations,
    )

    return pu.from_arrays(
        E=E,
        Y=Y,
        W_mean=W,
        W_sigma=0.03,
        sigma_y=0.01,
    )
