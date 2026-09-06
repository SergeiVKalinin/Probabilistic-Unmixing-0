"""Level 2 probabilistic unmixing.

Level 2 represents uncertainty in each endmember as uncertainty in
a complete spectral function.

The user supplies an ensemble of plausible spectra:

    S_library[k, realization, channel]

For each component, the library is compressed in log-spectrum space
using PCA. The retained functional coefficients have independent
standard-normal priors.

The spectral functions are global: one endmember function for each
component is shared by every measurement location.

For the current package:

    composition = mixture fraction

Inference
---------
The first package implementation uses

    joint MAP + Gauss-Newton Laplace approximation

for the global functional coefficients and all local concentrations.
"""

import numpy as np
from scipy.optimize import least_squares

from .results import UnmixingResult


LOG_FLOOR = 1e-8


# ============================================================
# Numerical utilities
# ============================================================

def _trapezoid(y, x, axis=-1):
    """Compatibility wrapper for trapezoidal integration."""

    if hasattr(np, "trapezoid"):
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


def _quadrature_weights(E):
    """
    Trapezoidal integration weights for an arbitrary 1D grid.
    """

    E = np.asarray(
        E,
        dtype=float,
    )

    if E.ndim != 1:
        raise ValueError(
            "E must be one-dimensional."
        )

    if E.size < 2:
        raise ValueError(
            "E must contain at least two channels."
        )

    dE = np.diff(E)

    if np.any(dE <= 0):
        raise ValueError(
            "E must be strictly increasing."
        )

    q = np.empty_like(E)

    q[0] = 0.5 * dE[0]
    q[-1] = 0.5 * dE[-1]

    if E.size > 2:
        q[1:-1] = 0.5 * (
            dE[:-1] + dE[1:]
        )

    return q


# ============================================================
# Simplex transformations
# ============================================================

def _alr_to_w(z):
    """
    Convert additive log-ratio coordinates to simplex fractions.

    The final component is used as the ALR reference.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

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

    exp_logits = np.exp(
        logits
    )

    return (
        exp_logits
        / exp_logits.sum(
            axis=-1,
            keepdims=True,
        )
    )


def _w_to_alr(w):
    """
    Convert simplex fractions to additive log-ratio coordinates.
    """

    w = np.asarray(
        w,
        dtype=float,
    )

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
    Approximate concentration uncertainty in ALR coordinates.

    For

        z_j = log(w_j / w_ref)

    use the first-order approximation

        Var(z_j)
        ~= (sigma_j / w_j)^2
         + (sigma_ref / w_ref)^2.

    Cross-covariances are neglected in this initial package version.
    """

    w = np.asarray(
        w,
        dtype=float,
    )

    sigma_w = np.asarray(
        sigma_w,
        dtype=float,
    )

    w_safe = np.clip(
        w,
        1e-12,
        None,
    )

    relative = (
        sigma_w
        / w_safe
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
# Functional prior construction
# ============================================================

def _normalize_library(
    E,
    S_library,
):
    """
    Normalize every library spectrum to unit integrated area.
    """

    S_library = np.asarray(
        S_library,
        dtype=float,
    ).copy()

    if not np.all(
        np.isfinite(S_library)
    ):
        raise ValueError(
            "S_library contains NaN or infinite values."
        )

    if np.any(
        S_library < 0.0
    ):
        raise ValueError(
            "The current Level 2 functional model requires "
            "nonnegative library spectra."
        )

    areas = _trapezoid(
        S_library,
        E,
        axis=-1,
    )

    if np.any(
        areas <= 0.0
    ):
        raise ValueError(
            "Every library spectrum must have positive integrated area."
        )

    return (
        S_library
        / areas[..., None]
    )


def _fit_functional_representation(
    functions,
    variance_threshold=0.995,
    min_rank=4,
    max_rank=12,
):
    """
    Fit a low-rank Gaussian representation in log-spectrum space.

    Parameters
    ----------
    functions : ndarray, shape (R, M)
        Library spectra for one component.

    variance_threshold : float
        Desired cumulative variance fraction in log-spectrum space.

    min_rank : int
        Minimum retained rank.

    max_rank : int
        Maximum retained rank.

    Returns
    -------
    dict
        Mean log spectrum, scaled PCA basis, retained rank,
        and variance diagnostics.
    """

    functions = np.asarray(
        functions,
        dtype=float,
    )

    logs = np.log(
        functions + LOG_FLOOR
    )

    mean_log = logs.mean(
        axis=0
    )

    centered = (
        logs - mean_log
    )

    _, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    if functions.shape[0] > 1:

        eigenvalues = (
            singular_values ** 2
            / (
                functions.shape[0]
                - 1
            )
        )

    else:

        raise ValueError(
            "Level 2 requires at least two library spectra "
            "per component."
        )

    total_variance = (
        eigenvalues.sum()
    )

    if total_variance <= 0.0:

        raise ValueError(
            "A component library contains no spectral variation."
        )

    cumulative = (
        np.cumsum(
            eigenvalues
        )
        / total_variance
    )

    required_rank = int(
        np.searchsorted(
            cumulative,
            variance_threshold,
        )
        + 1
    )

    available_rank = min(
        vt.shape[0],
        functions.shape[1],
    )

    rank = int(
        np.clip(
            required_rank,
            min_rank,
            min(
                max_rank,
                available_rank,
            ),
        )
    )

    # Each PCA vector is multiplied by sqrt(eigenvalue).
    #
    # Therefore a standard-normal coefficient carries the
    # variance scale learned from the spectral library.
    basis = (
        vt[:rank].T
        * np.sqrt(
            eigenvalues[:rank]
        )
    )

    # Reproducible sign convention.
    for j in range(rank):

        index = np.argmax(
            np.abs(
                basis[:, j]
            )
        )

        if basis[index, j] < 0.0:
            basis[:, j] *= -1.0

    return {
        "mean": mean_log,
        "basis": basis,
        "rank": rank,
        "required_rank": required_rank,
        "explained_variance": float(
            cumulative[
                rank - 1
            ]
        ),
    }


def _build_representations(
    E,
    S_library,
    variance_threshold=0.995,
    min_rank=4,
    max_rank=12,
):
    """
    Build one functional representation per component.
    """

    library = _normalize_library(
        E,
        S_library,
    )

    representations = []

    for k in range(
        library.shape[0]
    ):

        representation = (
            _fit_functional_representation(
                library[k],
                variance_threshold=variance_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
            )
        )

        representations.append(
            representation
        )

    return (
        representations,
        library,
    )


# ============================================================
# Functional spectra
# ============================================================

def _spectrum_from_coefficients(
    mean_log,
    basis,
    coefficients,
    quadrature,
):
    """
    Map functional coefficients to a positive unit-area spectrum.
    """

    u = (
        mean_log
        +
        basis
        @ coefficients
    )

    # Numerically stable exponentiation.
    v = np.exp(
        u - np.max(u)
    )

    spectrum = (
        v
        / (
            v
            @ quadrature
        )
    )

    return spectrum


def _spectrum_and_derivative(
    mean_log,
    basis,
    coefficients,
    quadrature,
):
    """
    Return a physical spectrum and derivative with respect
    to its functional coefficients.

    Returns
    -------
    spectrum : ndarray, shape (M,)

    derivative : ndarray, shape (M, rank)
    """

    spectrum = (
        _spectrum_from_coefficients(
            mean_log,
            basis,
            coefficients,
            quadrature,
        )
    )

    weighted_mean_basis = (
        (
            spectrum
            * quadrature
        )
        @ basis
    )

    derivative = (
        spectrum[:, None]
        * (
            basis
            -
            weighted_mean_basis[
                None,
                :
            ]
        )
    )

    return (
        spectrum,
        derivative,
    )


# ============================================================
# Joint Level 2 model
# ============================================================

class _Level2Problem:
    """
    Joint functional-spectral and concentration inference problem.
    """

    def __init__(
        self,
        data,
        representations,
    ):

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

        self.W_prior = np.asarray(
            data.W_mean,
            dtype=float,
        )

        self.W_sigma = np.asarray(
            data.W_sigma,
            dtype=float,
        )

        self.N = data.n_locations
        self.K = data.n_components
        self.M = data.n_channels

        self.representations = (
            representations
        )

        self.quadrature = (
            _quadrature_weights(
                self.E
            )
        )

        # ----------------------------------------------------
        # Spectral coefficient slices
        # ----------------------------------------------------

        self.ranks = [
            representation[
                "rank"
            ]
            for representation
            in representations
        ]

        self.spectral_slices = []

        offset = 0

        for rank in self.ranks:

            self.spectral_slices.append(
                slice(
                    offset,
                    offset + rank,
                )
            )

            offset += rank

        self.n_spectral = offset

        # ----------------------------------------------------
        # Concentration coordinates
        # ----------------------------------------------------

        self.z_center = (
            _w_to_alr(
                self.W_prior
            )
        )

        self.z_scale = (
            _fraction_sigma_to_alr_sigma(
                self.W_prior,
                self.W_sigma,
            )
        )

        self.n_concentration = (
            self.N
            * (self.K - 1)
        )

        self.n_parameters = (
            self.n_spectral
            +
            self.n_concentration
        )

    # --------------------------------------------------------
    # Decode parameter vector
    # --------------------------------------------------------

    def unpack(
        self,
        x,
        derivatives=False,
    ):

        x = np.asarray(
            x,
            dtype=float,
        )

        spectra = []
        spectral_derivatives = []

        for k, representation in enumerate(
            self.representations
        ):

            coefficients = (
                x[
                    self.spectral_slices[k]
                ]
            )

            if derivatives:

                spectrum, derivative = (
                    _spectrum_and_derivative(
                        representation["mean"],
                        representation["basis"],
                        coefficients,
                        self.quadrature,
                    )
                )

                spectral_derivatives.append(
                    derivative
                )

            else:

                spectrum = (
                    _spectrum_from_coefficients(
                        representation["mean"],
                        representation["basis"],
                        coefficients,
                        self.quadrature,
                    )
                )

            spectra.append(
                spectrum
            )

        S = np.asarray(
            spectra
        )

        # Standardized local concentration coordinates.
        u_z = (
            x[
                self.n_spectral:
            ]
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

        W = _alr_to_w(
            z
        )

        if derivatives:

            return (
                S,
                W,
                spectral_derivatives,
            )

        return (
            S,
            W,
        )

    # --------------------------------------------------------
    # Residual vector
    # --------------------------------------------------------

    def residual(
        self,
        x,
    ):

        S, W = self.unpack(
            x,
            derivatives=False,
        )

        Y_pred = (
            W @ S
        )

        spectral_residual = (
            (
                Y_pred
                -
                self.Y
            )
            /
            self.sigma_y
        ).ravel()

        # All unknowns are standardized to N(0,1) priors.
        prior_residual = x

        return np.concatenate(
            [
                spectral_residual,
                prior_residual,
            ]
        )

    # --------------------------------------------------------
    # Analytic residual Jacobian
    # --------------------------------------------------------

    def jacobian(
        self,
        x,
    ):

        S, W, dS = self.unpack(
            x,
            derivatives=True,
        )

        Y_pred = (
            W @ S
        )

        n_data_residuals = (
            self.N
            * self.M
        )

        J = np.zeros(
            (
                n_data_residuals
                + self.n_parameters,
                self.n_parameters,
            ),
            dtype=float,
        )

        # ----------------------------------------------------
        # Spectral coefficients
        #
        # dY_i / da_k = w_ik dS_k / da_k
        # ----------------------------------------------------

        for k in range(
            self.K
        ):

            block = (
                W[:, k, None, None]
                * dS[k][None, :, :]
            )

            block = (
                block
                /
                self.sigma_y[
                    :, :, None
                ]
            )

            J[
                :n_data_residuals,
                self.spectral_slices[k],
            ] = (
                block.reshape(
                    n_data_residuals,
                    self.ranks[k],
                )
            )

        # ----------------------------------------------------
        # Concentration coordinates
        #
        # For ALR coordinate j:
        #
        # dY_i / du_ij
        # =
        # sigma_z_ij * w_ij * (S_j - Y_pred_i)
        # ----------------------------------------------------

        for i in range(
            self.N
        ):

            row_slice = slice(
                i * self.M,
                (i + 1) * self.M,
            )

            for j in range(
                self.K - 1
            ):

                column = (
                    self.n_spectral
                    +
                    i * (
                        self.K - 1
                    )
                    +
                    j
                )

                derivative = (
                    self.z_scale[i, j]
                    * W[i, j]
                    * (
                        S[j]
                        -
                        Y_pred[i]
                    )
                )

                J[
                    row_slice,
                    column,
                ] = (
                    derivative
                    /
                    self.sigma_y[i]
                )

        # ----------------------------------------------------
        # Standard-normal priors
        # ----------------------------------------------------

        J[
            n_data_residuals:,
            :,
        ] = np.eye(
            self.n_parameters
        )

        return J


# ============================================================
# Posterior draws
# ============================================================

def _gaussian_draws(
    mean,
    covariance,
    n_draws,
    rng,
):
    """
    Draw from a Gaussian covariance robustly.
    """

    covariance = 0.5 * (
        covariance
        +
        covariance.T
    )

    try:

        L = np.linalg.cholesky(
            covariance
        )

    except np.linalg.LinAlgError:

        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
        )

        eigenvalues = np.clip(
            eigenvalues,
            0.0,
            None,
        )

        L = (
            eigenvectors
            * np.sqrt(
                eigenvalues
            )
        )

    normals = rng.normal(
        size=(
            n_draws,
            mean.size,
        )
    )

    return (
        mean[None, :]
        +
        normals
        @ L.T
    )


# ============================================================
# Public Level 2 fitting function
# ============================================================

def fit_level2(
    data,
    n_draws=1000,
    seed=0,
    variance_threshold=0.995,
    min_rank=4,
    max_rank=12,
    max_nfev=200,
):
    """
    Run Level 2 functional probabilistic unmixing.

    Parameters
    ----------
    data : UnmixingData
        Dataset containing ``S_library``.

    n_draws : int, default=1000
        Number of Gaussian-Laplace posterior draws.

    seed : int, default=0
        Random seed.

    variance_threshold : float, default=0.995
        Target fraction of log-library variance retained by PCA.

    min_rank : int, default=4
        Minimum functional rank per component.

    max_rank : int, default=12
        Maximum functional rank per component.

    max_nfev : int, default=200
        Maximum least-squares function evaluations.

    Returns
    -------
    UnmixingResult
        Common result object.
    """

    if data.S_library is None:

        raise ValueError(
            "Level 2 requires a spectral library. "
            "Use data.set_level2(S_library) first."
        )

    # --------------------------------------------------------
    # Construct the functional prior
    # --------------------------------------------------------

    representations, library = (
        _build_representations(
            data.E,
            data.S_library,
            variance_threshold=variance_threshold,
            min_rank=min_rank,
            max_rank=max_rank,
        )
    )

    problem = _Level2Problem(
        data,
        representations,
    )

    # --------------------------------------------------------
    # MAP
    #
    # x = 0 corresponds to:
    #
    #   functional prior centers
    #   measured concentration centers
    # --------------------------------------------------------

    x0 = np.zeros(
        problem.n_parameters,
        dtype=float,
    )

    solution = least_squares(
        problem.residual,
        x0=x0,
        jac=problem.jacobian,
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    if not solution.success:

        raise RuntimeError(
            "Level 2 MAP optimization failed: "
            + solution.message
        )

    x_map = (
        solution.x
    )

    # --------------------------------------------------------
    # Gauss-Newton Laplace covariance
    # --------------------------------------------------------

    J = solution.jac

    precision = (
        J.T @ J
    )

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
    # Posterior sampling from Laplace approximation
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    draws = _gaussian_draws(
        mean=x_map,
        covariance=covariance,
        n_draws=n_draws,
        rng=rng,
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

    for d in range(
        n_draws
    ):

        S, W = problem.unpack(
            draws[d],
            derivatives=False,
        )

        S_draws[d] = S
        W_draws[d] = W

    # --------------------------------------------------------
    # Posterior summaries
    # --------------------------------------------------------

    W_mean = W_draws.mean(
        axis=0
    )

    W_sd = W_draws.std(
        axis=0,
        ddof=1,
    )

    S_mean = S_draws.mean(
        axis=0
    )

    S_sd = S_draws.std(
        axis=0,
        ddof=1,
    )

    Y_pred = (
        W_mean
        @ S_mean
    )

    residuals = (
        data.Y
        -
        Y_pred
    )

    # --------------------------------------------------------
    # Metrics
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

        "n_functional_parameters": int(
            problem.n_spectral
        ),

        "n_total_parameters": int(
            problem.n_parameters
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
    # Functional-prior diagnostics
    # --------------------------------------------------------

    ranks = [
        int(
            representation[
                "rank"
            ]
        )
        for representation
        in representations
    ]

    required_ranks = [
        int(
            representation[
                "required_rank"
            ]
        )
        for representation
        in representations
    ]

    explained_variance = [
        float(
            representation[
                "explained_variance"
            ]
        )
        for representation
        in representations
    ]

    metadata = {
        "level": 2,

        "model": (
            "functional_log_pca"
        ),

        "posterior": (
            "joint MAP + Gauss-Newton Laplace approximation"
        ),

        "n_draws": int(
            n_draws
        ),

        "seed": int(
            seed
        ),

        "variance_threshold": float(
            variance_threshold
        ),

        "min_rank": int(
            min_rank
        ),

        "max_rank": int(
            max_rank
        ),

        "ranks": ranks,

        "required_ranks": required_ranks,

        "explained_variance": explained_variance,

        "optimizer_success": bool(
            solution.success
        ),

        "optimizer_message": str(
            solution.message
        ),
    }

    return UnmixingResult(
        level=2,

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
