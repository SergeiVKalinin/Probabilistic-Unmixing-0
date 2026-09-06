"""Level 0 probabilistic unmixing.

Level 0 assumes that the endmember spectra are known exactly.
Only the mixture fractions / concentrations are uncertain.

For the current implementation:

    composition = mixture fraction

Inference is performed independently at each measurement location.
"""

import numpy as np
from scipy.optimize import least_squares

from .results import UnmixingResult


# ============================================================
# Simplex transformations
# ============================================================

def _alr_to_w(z):
    """
    Convert additive log-ratio coordinates to simplex fractions.

    The final component is used as the ALR reference.
    """
    z = np.asarray(z, dtype=float)

    logits = np.concatenate(
        [z, np.zeros(z.shape[:-1] + (1,))],
        axis=-1,
    )

    logits = logits - np.max(logits, axis=-1, keepdims=True)

    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _w_to_alr(w):
    """
    Convert simplex fractions to additive log-ratio coordinates.

    A very small numerical floor is used only for the coordinate
    transformation so that experimental zero values do not produce
    infinite log ratios.
    """
    w = np.asarray(w, dtype=float)

    w_safe = np.clip(w, 1e-12, None)
    w_safe = w_safe / w_safe.sum(axis=-1, keepdims=True)

    return np.log(
        w_safe[..., :-1] / w_safe[..., [-1]]
    )


def _softmax_jacobian(z):
    """
    Jacobian dw/dz for the ALR-to-simplex transformation.

    Returns
    -------
    J : ndarray, shape (K, K-1)
    """
    w = _alr_to_w(z)

    k = w.size
    J = np.empty((k, k - 1))

    for i in range(k):
        for j in range(k - 1):
            delta = 1.0 if i == j else 0.0
            J[i, j] = w[i] * (delta - w[j])

    return J


# ============================================================
# Concentration uncertainty
# ============================================================

def _fraction_sigma_to_alr_sigma(w, sigma_w):
    """
    Approximate concentration uncertainty in ALR coordinates.

    The current public data object stores uncertainty in the
    physically intuitive concentration/fraction coordinates.

    For Level 0 inference we convert these uncertainties to an
    approximate diagonal uncertainty in ALR coordinates using
    first-order error propagation.

    For

        z_j = log(w_j / w_ref)

    the approximation is

        Var(z_j)
        ~= (sigma_j / w_j)^2
         + (sigma_ref / w_ref)^2

    Cross-covariances are neglected in this first implementation.
    """
    w = np.asarray(w, dtype=float)
    sigma_w = np.asarray(sigma_w, dtype=float)

    w_safe = np.clip(w, 1e-12, None)

    relative = sigma_w / w_safe

    sigma_z = np.sqrt(
        relative[:-1] ** 2
        + relative[-1] ** 2
    )

    return np.maximum(sigma_z, 1e-8)


# ============================================================
# One-location posterior
# ============================================================

def _residual(z, y, sigma_y, S, z0, sigma_z):
    """
    Residual vector whose squared norm defines the MAP objective.
    """
    w = _alr_to_w(z)

    y_pred = w @ S

    spectral_residual = (y_pred - y) / sigma_y
    prior_residual = (z - z0) / sigma_z

    return np.concatenate(
        [spectral_residual, prior_residual]
    )


def _jacobian(z, y, sigma_y, S, z0, sigma_z):
    """
    Analytic Jacobian of the Level 0 residual vector.
    """
    Jw = _softmax_jacobian(z)

    # S has shape (K, M)
    #
    # derivative of predicted spectrum with respect to ALR
    # coordinates has shape (M, K-1)
    J_spectral = S.T @ Jw

    J_spectral = J_spectral / sigma_y[:, None]

    J_prior = np.diag(1.0 / sigma_z)

    return np.vstack(
        [J_spectral, J_prior]
    )


def _fit_location(
    y,
    sigma_y,
    w_prior,
    sigma_w,
    S,
):
    """
    Fit one measurement location.
    """
    z0 = _w_to_alr(w_prior)

    sigma_z = _fraction_sigma_to_alr_sigma(
        w_prior,
        sigma_w,
    )

    solution = least_squares(
        _residual,
        x0=z0,
        jac=_jacobian,
        args=(
            y,
            sigma_y,
            S,
            z0,
            sigma_z,
        ),
    )

    if not solution.success:
        raise RuntimeError(
            "Level 0 MAP optimization failed."
        )

    z_map = solution.x

    # --------------------------------------------------------
    # Local Laplace approximation
    #
    # The first implementation uses the Gauss-Newton Hessian
    # J^T J. The more extensive controlled notebook contains
    # additional exact-Hessian and sampling diagnostics that
    # can be ported later.
    # --------------------------------------------------------

    J = _jacobian(
        z_map,
        y,
        sigma_y,
        S,
        z0,
        sigma_z,
    )

    precision = J.T @ J

    covariance = np.linalg.inv(precision)

    covariance = 0.5 * (
        covariance + covariance.T
    )

    return z_map, covariance


# ============================================================
# Public Level 0 fitting function
# ============================================================

def fit_level0(
    data,
    n_draws=2000,
    seed=0,
):
    """
    Run Level 0 probabilistic unmixing.

    Parameters
    ----------
    data : UnmixingData
        Input dataset.

        The dataset must contain ``S_fixed``.

    n_draws : int, default=2000
        Number of Gaussian-Laplace posterior draws used to
        propagate uncertainty from ALR coordinates to fractions.

    seed : int, default=0
        Random seed for posterior sampling.

    Returns
    -------
    UnmixingResult
        Common probabilistic-unmixing result object.
    """

    if data.S_fixed is None:
        raise ValueError(
            "Level 0 requires fixed endmember spectra. "
            "Use data.set_level0(S_fixed) first."
        )

    S = np.asarray(data.S_fixed, dtype=float)

    expected_shape = (
        data.n_components,
        data.n_channels,
    )

    if S.shape != expected_shape:
        raise ValueError(
            f"S_fixed must have shape {expected_shape}."
        )

    rng = np.random.default_rng(seed)

    N = data.n_locations
    K = data.n_components

    W_mean = np.empty((N, K))
    W_sd = np.empty((N, K))

    # --------------------------------------------------------
    # Infer concentrations independently at each location
    # --------------------------------------------------------

    for i in range(N):

        z_map, z_cov = _fit_location(
            y=data.Y[i],
            sigma_y=data.sigma_y[i],
            w_prior=data.W_mean[i],
            sigma_w=data.W_sigma[i],
            S=S,
        )

        # Draw from the local Gaussian approximation in ALR space.
        L = np.linalg.cholesky(z_cov)

        standard_normal = rng.normal(
            size=(n_draws, K - 1)
        )

        z_draws = (
            z_map[None, :]
            + standard_normal @ L.T
        )

        W_draws = _alr_to_w(z_draws)

        W_mean[i] = W_draws.mean(axis=0)
        W_sd[i] = W_draws.std(
            axis=0,
            ddof=1,
        )

    # --------------------------------------------------------
    # Level 0 spectra are known exactly
    # --------------------------------------------------------

    S_mean = S.copy()
    S_sd = np.zeros_like(S)

    # --------------------------------------------------------
    # Posterior mean spectral reconstruction
    # --------------------------------------------------------

    Y_pred = W_mean @ S_mean
    residuals = data.Y - Y_pred

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    metrics = {
        "reconstruction_rmse": float(
            np.sqrt(
                np.mean(residuals ** 2)
            )
        ),
        "mean_concentration_sd": float(
            np.mean(W_sd)
        ),
    }

    # Synthetic-only evaluation.
    if data.W_true is not None:

        metrics["concentration_rmse"] = float(
            np.sqrt(
                np.mean(
                    (W_mean - data.W_true) ** 2
                )
            )
        )

    if data.S_true is not None:

        metrics["spectral_rmse"] = float(
            np.sqrt(
                np.mean(
                    (S_mean - data.S_true) ** 2
                )
            )
        )

    # --------------------------------------------------------
    # Return common result object
    # --------------------------------------------------------

    return UnmixingResult(
        level=0,

        W_mean=W_mean,
        W_sd=W_sd,

        S_mean=S_mean,
        S_sd=S_sd,

        Y_pred=Y_pred,
        residuals=residuals,

        metrics=metrics,

        metadata={
            "level": 0,
            "model": "known_fixed_endmembers",
            "posterior": "MAP + local Laplace approximation",
            "n_draws": int(n_draws),
            "seed": int(seed),
        },

        W_true=data.W_true,
        S_true=data.S_true,
        Y_clean=data.Y_clean,
    )
