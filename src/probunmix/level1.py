"""Level 1 probabilistic unmixing.

Level 1 assumes that each endmember spectrum has a known parametric
form, but the parameters of that spectrum are uncertain.

Current implementation
----------------------
The supported spectral model is:

    "two_gaussian"

with five parameters per component:

    [r, mu1, sigma1, mu2, sigma2]

where r is the integrated-area fraction of the first Gaussian.

The endmember parameters are global: one spectrum for each component
is shared by every measurement location.

For the current package:

    composition = mixture fraction
"""

import numpy as np
from scipy.optimize import least_squares

from .results import UnmixingResult


# ============================================================
# Simplex transformations
# ============================================================

def _alr_to_w(z):
    """Convert additive log-ratio coordinates to simplex fractions."""

    z = np.asarray(z, dtype=float)

    logits = np.concatenate(
        [
            z,
            np.zeros(
                z.shape[:-1] + (1,)
            ),
        ],
        axis=-1,
    )

    logits -= np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp_logits = np.exp(logits)

    return (
        exp_logits
        / exp_logits.sum(
            axis=-1,
            keepdims=True,
        )
    )


def _w_to_alr(w):
    """Convert simplex fractions to additive log-ratio coordinates."""

    w = np.asarray(w, dtype=float)

    w = np.clip(
        w,
        1e-12,
        None,
    )

    w /= w.sum(
        axis=-1,
        keepdims=True,
    )

    return np.log(
        w[..., :-1]
        / w[..., [-1]]
    )


def _fraction_sigma_to_alr_sigma(
    w,
    sigma_w,
):
    """
    Approximate uncertainty in ALR coordinates.

    For z_j = log(w_j / w_ref),

    Var(z_j) is approximated as

        (sigma_j / w_j)^2
        +
        (sigma_ref / w_ref)^2.

    This is the same simple approximation currently used in Level 0.
    """

    w = np.asarray(
        w,
        dtype=float,
    )

    sigma_w = np.asarray(
        sigma_w,
        dtype=float,
    )

    w = np.clip(
        w,
        1e-12,
        None,
    )

    relative = (
        sigma_w / w
    )

    sigma_z = np.sqrt(
        relative[..., :-1] ** 2
        +
        relative[..., [-1]] ** 2
    )

    return np.maximum(
        sigma_z,
        1e-8,
    )


# ============================================================
# Two-Gaussian spectral model
# ============================================================

def _trapezoid(y, x, axis=-1):
    """Compatibility wrapper for NumPy trapezoidal integration."""

    if hasattr(
        np,
        "trapezoid",
    ):
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


def _two_gaussian(
    E,
    theta,
):
    """
    Construct normalized two-Gaussian spectra.

    Parameters
    ----------
    E : ndarray, shape (M,)

    theta : ndarray, shape (..., 5)
        Parameters

        [r, mu1, sigma1, mu2, sigma2]

    Returns
    -------
    ndarray, shape (..., M)
    """

    theta = np.asarray(
        theta,
        dtype=float,
    )

    r = theta[..., 0, None]
    mu1 = theta[..., 1, None]
    sigma1 = theta[..., 2, None]
    mu2 = theta[..., 3, None]
    sigma2 = theta[..., 4, None]

    g1 = np.exp(
        -0.5
        * (
            (E - mu1)
            / sigma1
        ) ** 2
    ) / (
        sigma1
        * np.sqrt(
            2.0 * np.pi
        )
    )

    g2 = np.exp(
        -0.5
        * (
            (E - mu2)
            / sigma2
        ) ** 2
    ) / (
        sigma2
        * np.sqrt(
            2.0 * np.pi
        )
    )

    spectrum = (
        r * g1
        +
        (1.0 - r) * g2
    )

    area = _trapezoid(
        spectrum,
        E,
        axis=-1,
    )

    return (
        spectrum
        / area[..., None]
    )


# ============================================================
# Native spectral coordinates
# ============================================================

def _physical_to_native(theta):
    """
    Convert physical spectral parameters to optimization coordinates.

    Widths are represented in logarithmic coordinates.
    """

    theta = np.asarray(
        theta,
        dtype=float,
    ).copy()

    theta[..., 2] = np.log(
        theta[..., 2]
    )

    theta[..., 4] = np.log(
        theta[..., 4]
    )

    return theta


def _native_to_physical(theta_native):
    """
    Convert optimization coordinates back to physical parameters.
    """

    theta = np.asarray(
        theta_native,
        dtype=float,
    ).copy()

    theta[..., 2] = np.exp(
        theta[..., 2]
    )

    theta[..., 4] = np.exp(
        theta[..., 4]
    )

    return theta


def _physical_sigma_to_native(
    theta_mean,
    theta_sigma,
):
    """
    Convert approximate physical parameter SDs to native-coordinate SDs.

    For peak widths, first-order propagation gives approximately

        sigma_log_width = sigma_width / width.
    """

    theta_mean = np.asarray(
        theta_mean,
        dtype=float,
    )

    theta_sigma = np.asarray(
        theta_sigma,
        dtype=float,
    )

    scale = theta_sigma.copy()

    scale[..., 2] = (
        theta_sigma[..., 2]
        / theta_mean[..., 2]
    )

    scale[..., 4] = (
        theta_sigma[..., 4]
        / theta_mean[..., 4]
    )

    return np.maximum(
        scale,
        1e-8,
    )


# ============================================================
# Joint Level 1 problem
# ============================================================

class _Level1Problem:
    """
    Joint posterior over global spectral parameters and local fractions.
    """

    def __init__(self, data):

        if data.spectral_model != "two_gaussian":
            raise NotImplementedError(
                "The current Level 1 implementation supports only "
                "spectral_model='two_gaussian'."
            )

        self.data = data

        self.E = np.asarray(
            data.E,
            dtype=float,
        )

        self.Y = np.asarray(
            data.Y,
            dtype=float,
        )

        self.sigma_y = np.asarray(
            data.sigma_y,
            dtype=float,
        )

        self.N = data.n_locations
        self.K = data.n_components
        self.M = data.n_channels

        # ----------------------------------------------------
        # Spectral prior
        # ----------------------------------------------------

        self.theta_center = (
            _physical_to_native(
                data.theta_mean
            )
        )

        self.theta_scale = (
            _physical_sigma_to_native(
                data.theta_mean,
                data.theta_sigma,
            )
        )

        # ----------------------------------------------------
        # Concentration prior
        # ----------------------------------------------------

        self.z_center = (
            _w_to_alr(
                data.W_mean
            )
        )

        self.z_scale = (
            _fraction_sigma_to_alr_sigma(
                data.W_mean,
                data.W_sigma,
            )
        )

        self.n_theta = (
            self.K * 5
        )

        self.n_z = (
            self.N
            * (self.K - 1)
        )

        self.n_parameters = (
            self.n_theta
            +
            self.n_z
        )

    def unpack(self, x):
        """
        Convert standardized unknowns into physical spectra and fractions.
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        # Global spectral coordinates.
        u_theta = (
            x[:self.n_theta]
            .reshape(
                self.K,
                5,
            )
        )

        theta_native = (
            self.theta_center
            +
            self.theta_scale
            * u_theta
        )

        theta = (
            _native_to_physical(
                theta_native
            )
        )

        # Local concentration coordinates.
        u_z = (
            x[self.n_theta:]
            .reshape(
                self.N,
                self.K - 1,
            )
        )

        z = (
            self.z_center
            +
            self.z_scale
            * u_z
        )

        W = _alr_to_w(z)

        return (
            theta,
            W,
        )

    def residual(self, x):
        """
        Standardized residual vector defining the posterior objective.
        """

        theta, W = self.unpack(
            x
        )

        # ----------------------------------------------------
        # Basic physical support
        # ----------------------------------------------------

        r = theta[:, 0]

        if (
            np.any(r <= 0.01)
            or np.any(r >= 0.99)
            or np.any(theta[:, 2] <= 0.0)
            or np.any(theta[:, 4] <= 0.0)
        ):
            # least_squares requires a finite residual vector.
            return np.full(
                self.Y.size
                + self.n_parameters,
                1e12,
            )

        S = _two_gaussian(
            self.E,
            theta,
        )

        Y_pred = (
            W @ S
        )

        spectral_residual = (
            (Y_pred - self.Y)
            / self.sigma_y
        ).ravel()

        # Because all unknown coordinates are standardized relative
        # to their priors, their prior residuals are simply x.
        prior_residual = x

        return np.concatenate(
            [
                spectral_residual,
                prior_residual,
            ]
        )


# ============================================================
# Posterior summaries
# ============================================================

def _draw_posterior(
    problem,
    x_map,
    covariance,
    n_draws,
    rng,
):
    """
    Draw from the joint Gaussian Laplace approximation.
    """

    try:

        draws = rng.multivariate_normal(
            mean=x_map,
            cov=covariance,
            size=n_draws,
            check_valid="raise",
        )

    except (ValueError, np.linalg.LinAlgError):

        # Numerical fallback if tiny negative eigenvalues appear
        # from floating-point inversion.
        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.clip(
            eigenvalues,
            0.0,
            None,
        )

        transform = (
            eigenvectors
            * np.sqrt(
                eigenvalues
            )
        )

        draws = (
            x_map[None, :]
            +
            rng.normal(
                size=(
                    n_draws,
                    problem.n_parameters,
                )
            )
            @ transform.T
        )

    W_draws = np.empty(
        (
            n_draws,
            problem.N,
            problem.K,
        )
    )

    S_draws = np.empty(
        (
            n_draws,
            problem.K,
            problem.M,
        )
    )

    for j in range(
        n_draws
    ):

        theta, W = (
            problem.unpack(
                draws[j]
            )
        )

        # Reject obviously invalid r values arising in the
        # Gaussian Laplace tails by clipping only for summary
        # transformation.
        theta[:, 0] = np.clip(
            theta[:, 0],
            1e-6,
            1.0 - 1e-6,
        )

        W_draws[j] = W

        S_draws[j] = (
            _two_gaussian(
                problem.E,
                theta,
            )
        )

    return (
        W_draws,
        S_draws,
    )


# ============================================================
# Public fitting function
# ============================================================

def fit_level1(
    data,
    n_draws=1000,
    seed=0,
    max_nfev=200,
):
    """
    Run Level 1 probabilistic unmixing.

    The endmember spectral parameters and all local concentrations
    are inferred jointly.

    Parameters
    ----------
    data : UnmixingData
        Input dataset containing Level 1 spectral knowledge.

    n_draws : int, default=1000
        Number of posterior draws from the Gaussian Laplace
        approximation.

    seed : int, default=0
        Random seed used for posterior draws.

    max_nfev : int, default=200
        Maximum number of least-squares function evaluations.

    Returns
    -------
    UnmixingResult
        Common result object.
    """

    # --------------------------------------------------------
    # Check required information
    # --------------------------------------------------------

    if data.theta_mean is None:
        raise ValueError(
            "Level 1 requires theta_mean. "
            "Use data.set_level1(...) first."
        )

    if data.theta_sigma is None:
        raise ValueError(
            "Level 1 requires theta_sigma. "
            "Use data.set_level1(...) first."
        )

    if data.spectral_model is None:
        raise ValueError(
            "Level 1 requires a spectral model. "
            "Use data.set_level1(...) first."
        )

    problem = _Level1Problem(
        data
    )

    # --------------------------------------------------------
    # MAP estimation
    #
    # x = 0 corresponds exactly to all prior centers.
    # --------------------------------------------------------

    x0 = np.zeros(
        problem.n_parameters,
        dtype=float,
    )

    solution = least_squares(
        problem.residual,
        x0=x0,
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(
            "Level 1 MAP optimization failed: "
            + solution.message
        )

    x_map = solution.x

    # --------------------------------------------------------
    # Laplace covariance
    #
    # For this first package implementation we use the
    # Gauss-Newton Hessian J^T J.
    # --------------------------------------------------------

    J = solution.jac

    precision = (
        J.T @ J
    )

    # Symmetrize before inversion.
    precision = 0.5 * (
        precision
        +
        precision.T
    )

    try:

        covariance = np.linalg.inv(
            precision
        )

    except np.linalg.LinAlgError:

        covariance = np.linalg.pinv(
            precision
        )

    covariance = 0.5 * (
        covariance
        +
        covariance.T
    )

    # --------------------------------------------------------
    # Posterior draws
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    W_draws, S_draws = (
        _draw_posterior(
            problem=problem,
            x_map=x_map,
            covariance=covariance,
            n_draws=n_draws,
            rng=rng,
        )
    )

    # --------------------------------------------------------
    # Posterior summaries
    # --------------------------------------------------------

    W_mean = (
        W_draws.mean(
            axis=0
        )
    )

    W_sd = (
        W_draws.std(
            axis=0,
            ddof=1,
        )
    )

    S_mean = (
        S_draws.mean(
            axis=0
        )
    )

    S_sd = (
        S_draws.std(
            axis=0,
            ddof=1,
        )
    )

    Y_pred = (
        W_mean @ S_mean
    )

    residuals = (
        data.Y
        -
        Y_pred
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    metrics = {
        "reconstruction_rmse": float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        ),

        "mean_concentration_sd": float(
            np.mean(
                W_sd
            )
        ),

        "mean_spectral_sd": float(
            np.mean(
                S_sd
            )
        ),

        "optimizer_cost": float(
            solution.cost
        ),

        "optimizer_nfev": int(
            solution.nfev
        ),
    }

    # Synthetic evaluation only.
    if data.W_true is not None:

        metrics[
            "concentration_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        W_mean
                        -
                        data.W_true
                    ) ** 2
                )
            )
        )

    if data.S_true is not None:

        metrics[
            "spectral_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        S_mean
                        -
                        data.S_true
                    ) ** 2
                )
            )
        )

    # --------------------------------------------------------
    # MAP physical spectral parameters
    # --------------------------------------------------------

    theta_map, _ = (
        problem.unpack(
            x_map
        )
    )

    metadata = {
        "level": 1,
        "model": "parametric_two_gaussian",
        "posterior": (
            "joint MAP + Gauss-Newton Laplace approximation"
        ),
        "n_draws": int(
            n_draws
        ),
        "seed": int(
            seed
        ),
        "optimizer_success": bool(
            solution.success
        ),
        "optimizer_message": str(
            solution.message
        ),
        "theta_map": theta_map,
    }

    return UnmixingResult(
        level=1,

        W_mean=W_mean,
        W_sd=W_sd,

        S_mean=S_mean,
        S_sd=S_sd,

        Y_pred=Y_pred,
        residuals=residuals,

        metrics=metrics,
        metadata=metadata,

        W_true=data.W_true,
        S_true=data.S_true,
        Y_clean=data.Y_clean,
    )
