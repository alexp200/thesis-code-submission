"""Linearized Laplace approximation (LA) in the eigenbasis of the generalized Gauss-Newton (GGN) matrix.

All functions work on the flat parameter vector theta of the MAP model. The likelihood is Gaussian,
y ~ N(f(theta), sigma_n^2), and the prior is the isotropic Gaussian N(0, alpha^-1 I) of src.priors.

    1. compute_ggn / compute_ggn_vector     GGN = J^T J of the model output at the MAP estimate
    2. GGN_eigendecompose                   GGN = U diag(lambda) U^T, eigenvalues clipped at 0
    3. neg_log_marginal_likelihood,         alpha and sigma_n from the Laplace evidence (Adam on the logs)
       calibrate_mll
    4. posterior_eigen_weights              posterior covariance Sigma = U diag(w) U^T, w_j = 1 / (lambda_j / sigma_n^2 + alpha)
    5. pushforward_linearized_std,          covariance of the linearized prediction, J Sigma J^T;
       pushforward_linearized_cov_fullr,    these use J projected onto the eigenvectors (project_jacobian*)
       project_jacobian, project_jacobian_vector
    6. chunked_eigen_jvp_marginal_variance  the same marginal variances without forming J

Because the prior curvature alpha * I commutes with the GGN, the posterior covariance and the log-determinant in
the marginal likelihood follow from the GGN eigenvalues alone: no matrix is inverted after the single
eigendecomposition, and the log-determinant is sum_j log(alpha + lambda_j / sigma_n^2).
"""

import jax
import jax.numpy as jnp
import optax
from collections.abc import Callable


def compute_ggn(
        f_scalar: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        p_flat: jnp.ndarray, 
        x_train: jnp.ndarray, 
        hessian_scale: float = 1.0
        ) -> jnp.ndarray:
    
    """ Compute the Generalized Gauss-Newton (GGN) matrix for a scalar function 
        f_scalar with respect to parameters p_flat and training data x_train.
    Args:
        f_scalar: A scalar function (NN) that takes parameters and input data and 
        returns a scalar output.
        p_flat: A flat array of parameters for the function f_scalar.
        x_train: Training inputs, where each row corresponds to a different training
        example.
        hessian_scale: Multiplicative scale for J^T J. For Gaussian-Laplace
        evidence this should stay at 1.0, so downstream uses
        A = alpha*I + J^T J / sigma_n^2. Use 2.0 only when explicitly
        approximating the Hessian of raw SSE; MSE would require 2.0 / N.
    Returns:
        GGN: The Generalized Gauss-Newton matrix computed from the Jacobian of f"""
    
    J_train = jax.vmap(lambda x: jax.grad(f_scalar)(p_flat, x))(x_train)  # (N, D)
    GGN     = hessian_scale * (J_train.T @ J_train)
    return GGN


def compute_ggn_vector(
        f_vec: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        p_flat: jnp.ndarray,
        x_train: jnp.ndarray,
        hessian_scale: float = 1.0
        ) -> jnp.ndarray:
    """ Generalized Gauss-Newton (GGN) matrix for a vector-valued function
        f_vec (e.g. a NODE's state, out_dim > 1), generalizing `compute_ggn`
        (which requires scalar output) via jax.jacobian instead of jax.grad.
        Both output dimensions and training points contribute additively to
        the same GGN, following the standard multi-output GGN sum J^T J.
    Args:
        f_vec: A vector-valued function that takes parameters and input data
        and returns an (out_dim,) array.
        p_flat: A flat array of parameters for the function f_vec.
        x_train: Training inputs, where each row corresponds to a different
        training example.
        hessian_scale: See `compute_ggn`.
    Returns:
        GGN: The Generalized Gauss-Newton matrix, shape (D, D).
    """
    J_train = jax.vmap(lambda x: jax.jacobian(f_vec)(p_flat, x))(x_train)  # (N, out_dim, D)
    D = p_flat.shape[0]
    J_flat = J_train.reshape(-1, D)  # (N * out_dim, D)
    GGN = hessian_scale * (J_flat.T @ J_flat)
    return GGN

def GGN_eigendecompose(
        GGN: jnp.ndarray,
        THRESHOLD: float = 1e-3,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Compute the eigenvalues and eigenvectors of the (GGN) matrix.
    Small negative eigenvalues from numerical roundoff are clipped to zero.
    Returns:
        eigvals: Eigenvalues of the GGN matrix.
        eigvecs: Eigenvectors of the GGN matrix.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(GGN)   # ascending
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    D = GGN.shape[0]
    print(f"Low-eigenvalue directions (lambda < {THRESHOLD}): "
        f"{int((eigenvalues < THRESHOLD).sum())}/{D}")
    return eigenvalues, eigenvectors
    

#### -------------- Linearized Pushforward -------------- ####


def chunked_eigen_jvp_marginal_variance(
        function: Callable[[jax.Array], jax.Array],
        p_map: jnp.ndarray,
        eigendirections: jnp.ndarray,
        weights: jnp.ndarray,
        *,
        chunk_size: int = 16,
        ) -> jnp.ndarray:
    """Compute diag(J U diag(weights) U.T J.T) without constructing J.

    ``jax.linearize`` evaluates ``function`` once at ``p_map`` and returns
    its reusable linear map ``v -> Jv``. Processing every posterior
    eigendirection therefore gives the same marginal variance as the dense
    Jacobian pushforward, while only materializing bounded output chunks.

    Args:
        function: Vector-valued model function of the flat parameters.
        p_map: MAP parameters, shape (D,).
        eigendirections: Parameter-space directions, shape (D, K).
        weights: Posterior variances along those directions, shape (K,).
        chunk_size: Number of directions evaluated simultaneously.
    Returns:
        Marginal function-space variance with the shape of function(p_map).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if p_map.ndim != 1:
        raise ValueError("p_map must have shape (D,).")
    if eigendirections.ndim != 2 or eigendirections.shape[0] != p_map.shape[0]:
        raise ValueError("eigendirections must have shape (D, K).")
    if weights.shape != (eigendirections.shape[1],):
        raise ValueError("weights must have shape (K,).")

    output_at_map, linearized_function = jax.linearize(function, p_map)
    variance = jnp.zeros_like(output_at_map)

    def contribution(direction_chunk, weight_chunk):
        projected = jax.vmap(
            linearized_function,
            in_axes=1,
            out_axes=0,
        )(direction_chunk)
        weight_shape = (weight_chunk.shape[0],) + (1,) * output_at_map.ndim
        return jnp.sum(
            projected**2 * weight_chunk.reshape(weight_shape),
            axis=0,
        )

    n_parameters, n_directions = eigendirections.shape
    n_full_chunks, remainder = divmod(n_directions, chunk_size)

    if n_full_chunks:
        starts = jnp.arange(n_full_chunks, dtype=jnp.int32) * chunk_size

        def scan_chunk(current_variance, start):
            zero = jnp.zeros((), dtype=start.dtype)
            direction_chunk = jax.lax.dynamic_slice(
                eigendirections,
                (zero, start),
                (n_parameters, chunk_size),
            )
            weight_chunk = jax.lax.dynamic_slice(
                weights,
                (start,),
                (chunk_size,),
            )
            return current_variance + contribution(
                direction_chunk,
                weight_chunk,
            ), None

        variance, _ = jax.lax.scan(scan_chunk, variance, starts)

    if remainder:
        start = n_full_chunks * chunk_size
        variance = variance + contribution(
            eigendirections[:, start:],
            weights[start:],
        )

    return jnp.maximum(variance, 0.0)  # guards against negative roundoff


def posterior_eigen_weights(
        eigenvalues: jnp.ndarray,
        active_idx: jnp.ndarray,
        tau_active: float,
        prior,
        sigma_noise: float = 1.0,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Posterior variances along the active GGN eigenvectors.

    With the convention Sigma_theta = (H_prior + G / sigma_n^2)^-1 and an isotropic prior
    (H_prior = alpha * I), H_post has the same eigenvectors as the GGN G, so the posterior
    variance along eigenvector j is the reciprocal 1 / (lambda_j / sigma_n^2 + alpha); no matrix
    inverse is needed. This only holds for an isotropic prior; other priors raise.

    `active_idx` determines the number k of directions used: the last k eigenvectors of the
    ascending eigendecomposition (the full spectrum in all experiments).
    """
    if not prior.is_isotropic():
        raise NotImplementedError("posterior_eigen_weights supports isotropic priors only (H_prior(alpha) = alpha * I).")
    fast_diag = prior.fast_path_diag()
    active_idx = jnp.asarray(active_idx)
    D = eigenvalues.shape[0]
    k = active_idx.shape[0]
    active_idx_array = jnp.arange(D - k, D)
    sigma_n2 = jnp.asarray(sigma_noise) ** 2
    prior_curv_active = tau_active * fast_diag[active_idx_array]
    weights = 1.0 / (jnp.asarray(eigenvalues)[active_idx_array] / sigma_n2 + prior_curv_active)
    return active_idx_array, weights


def pushforward_linearized_cov_fullr(
        alpha,
        eigenvalues,
        active_idx,
        tau_active,
        prior,
        sigma_noise=1.0,
        ):
    """Full covariance matrix of the linearized pushforward,
    Sigma_Y = J_active @ Sigma_theta_active @ J_active.T, including cross-covariances
    between all rows of alpha (e.g. all timesteps of a trajectory).
    Args:
        alpha: (T, D), J @ eigenvectors -- same as in pushforward_linearized_std.
        eigenvalues, active_idx, tau_active, prior, sigma_noise: same as in
            pushforward_linearized_std.
    Returns:
        Sigma_Y: (T, T) covariance matrix. diag(Sigma_Y) == pushforward_linearized_std(...)**2.
    """
    active_idx_array, w = posterior_eigen_weights(
        eigenvalues, active_idx, tau_active, prior, sigma_noise=sigma_noise
    )
    U = alpha[:, active_idx_array] * jnp.sqrt(w)[None, :]   # (T, k)
    return U @ U.T                                           # (T, T)


def pushforward_linearized_std (alpha: jnp.ndarray,
                           eigenvalues: jnp.ndarray,
                           active_idx: jnp.ndarray,
                           tau_active: float,
                           prior,
                           sigma_noise: float = 1.0
                           ) -> jnp.ndarray:
    """ Compute the pushforward of the linearized function along the active eigenvectors.
    Args:
        alpha: The projected Jacobian of the function onto the eigenvectors.
        eigenvalues: Eigenvalues of the GGN matrix.
        active_idx: Indices of the active eigenvectors (those with non-negligible eigenvalues).
        tau_active: Prior precision alpha.
        prior: A src.priors.IsotropicGaussianPrior -- see posterior_eigen_weights.
        sigma_noise: Observation-noise std used only to scale the likelihood
            curvature in Sigma_theta = (H_prior + G/sigma_n^2)^-1. It is not
            added as predictive observation noise here.
    Returns:
        pushforward: The pushforward of the linearized function along the active eigenvectors.
    """
    active_idx_array, w = posterior_eigen_weights(
        eigenvalues, active_idx, tau_active, prior, sigma_noise=sigma_noise
    )
    var = alpha[:, active_idx_array] ** 2 @ w              # (T,)
    return jnp.sqrt(jnp.maximum(var, 0.0))

#### -------------- Projection of the Jacobian onto the GGN eigenvectors -------------- ####

def project_jacobian(f_scalar: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                     p_flat: jnp.ndarray, 
                     eigenvectors: jnp.ndarray, 
                     x_eval: jnp.ndarray
                     ) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """ Project the Jacobian of f_scalar onto the subspace spanned by the
        eigenvectors. This is useful for analyzing the behavior of the function
        in the directions of the eigenvectors.
    Args:
        f_scalar: A scalar function (NN) that takes parameters and input data and 
        returns a scalar output.
        p_flat: A flat array of parameters for the function f_scalar.
        eigenvectors: Eigenvectors of the GGN matrix.
        x_eval: Evaluation inputs, where each row corresponds to a different evaluation
        example.
    Returns:
        projected_jacobian: A function that computes the projected Jacobian of f_scalar
        onto the subspace spanned by the eigenvectors.
    """
    jacobian_at = lambda x: jax.grad(f_scalar)(p_flat, x)
    J_plot = jax.vmap(jacobian_at)(x_eval)          # (T, D)
    alpha_all = jnp.array(J_plot @ eigenvectors)     # (T, D)
    return alpha_all


def project_jacobian_vector(f_vec: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                     p_flat: jnp.ndarray,
                     eigenvectors: jnp.ndarray,
                     x_eval: jnp.ndarray
                     ) -> jnp.ndarray:
    """ Vector-output generalization of `project_jacobian` (which requires
        scalar f_scalar), for e.g. a NODE's multi-dimensional state. Returns
        one alpha array per output dimension, so `pushforward_linearized_std`
        can be called once per dimension on the result.
    Args:
        f_vec: A vector-valued function that takes parameters and input data
        and returns an (out_dim,) array.
        p_flat: A flat array of parameters for the function f_vec.
        eigenvectors: Eigenvectors of the GGN matrix.
        x_eval: Evaluation inputs, where each row corresponds to a different
        evaluation example.
    Returns:
        alpha_all: shape (out_dim, T, D) -- alpha_all[i] is what
        `project_jacobian` would have returned for output dimension i.
    """
    jacobian_at = lambda x: jax.jacobian(f_vec)(p_flat, x)  # (out_dim, D)
    J_plot = jax.vmap(jacobian_at)(x_eval)                   # (T, out_dim, D)
    alpha_all = jnp.einsum('toD,DK->otK', J_plot, eigenvectors)  # (out_dim, T, D)
    return alpha_all


#### -------------- MLL Calibration -------------- ####


def neg_log_marginal_likelihood(
        log_prior_prec: jnp.ndarray,
        log_sigma_noise: jnp.ndarray,
        S_theta: float,
        theta: jnp.ndarray,
        N: int,
        eigvals_curv: jnp.ndarray,
        prior,
        ) -> jnp.ndarray:
    """ Negative log marginal likelihood (Laplace evidence) of a prior
        `prior` (see src/priors.py) and Gaussian noise N(0, sigma_noise^2),
        evaluated at the MAP estimate theta_hat:

        log p(D | alpha, sigma_n^2) =
            -(1/(2 sigma_n^2)) S(theta_hat) + prior.log_density(theta_hat, alpha)
            - (N/2) log(2 pi sigma_n^2)
            - (1/2) log det(A) + const,   A = H_prior(alpha) + curv_estimate/sigma_n^2

        Optimized on log(alpha), log(sigma_n) so both stay positive without
        constraints.

        For an isotropic prior the log-determinant has the closed form
        `sum(log(prior_prec + eigvals_curv / sigma_n2))`, because H_prior commutes with the GGN;
        other priors raise.
    Args:
        log_prior_prec: log(alpha), the log prior precision.
        log_sigma_noise: log(sigma_n), the log observation noise std.
        S_theta: Sum of squared training residuals at theta_hat,
            sum((f(theta_hat, x_i) - y_i)^2).
        theta: The flat MAP parameter vector theta_hat, shape (D,).
        N: Number of training points.
        eigvals_curv: Eigenvalues of the GGN curvature matrix, shape (D,)
            (from GGN_eigendecompose).
        prior: A src.priors.IsotropicGaussianPrior describing p(theta | alpha).
    Returns:
        Negative log marginal likelihood (scalar), to be minimized.
    """
    prior_prec = jnp.exp(log_prior_prec)
    sigma_n2 = jnp.exp(log_sigma_noise) ** 2

    log_likelihood = -S_theta / (2 * sigma_n2) - 0.5 * N * jnp.log(2 * jnp.pi * sigma_n2)
    log_prior = prior.log_density(theta, prior_prec)

    if not prior.is_isotropic():
        raise NotImplementedError("neg_log_marginal_likelihood supports isotropic priors only (H_prior(alpha) = alpha * I).")
    fast_diag = prior.fast_path_diag()
    logdet_A = jnp.sum(jnp.log(prior_prec * fast_diag + eigvals_curv / sigma_n2))

    log_evidence = log_likelihood + log_prior - 0.5 * logdet_A
    return -log_evidence


def _run_mll_adam_loop(
        objective,
        n_steps: int,
        lr: float,
        log_prior_prec_init: float,
        log_sigma_noise_init: float,
        verbose: bool,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Adam loop over (log_prior_prec, log_sigma_noise) minimizing `objective(params) -> scalar loss`.
    Returns the optimized prior precision and noise standard deviation (not in log space).
    """
    params = (jnp.array(log_prior_prec_init), jnp.array(log_sigma_noise_init))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def mll_step(params, opt_state):
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_steps):
        params, opt_state, loss = mll_step(params, opt_state)
        if verbose and (i % 50 == 0 or i == n_steps - 1):
            print(f"  step {i:4d}  -log p(D) = {float(loss):.5f}  "
                  f"prior_prec={float(jnp.exp(params[0])):.5g}  "
                  f"sigma_noise={float(jnp.exp(params[1])):.5g}")

    prior_prec_opt = jnp.exp(params[0])
    sigma_noise_opt = jnp.exp(params[1])
    return prior_prec_opt, sigma_noise_opt


def calibrate_mll(
        S_theta: float,
        theta: jnp.ndarray,
        N: int,
        eigvals_curv: jnp.ndarray,
        prior,
        n_steps: int = 200,
        lr: float = 1e-1,
        log_prior_prec_init: float = 0.0,
        log_sigma_noise_init: float = 0.0,
        verbose: bool = True,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Calibrate prior_prec and sigma_noise by minimizing
        neg_log_marginal_likelihood with Adam on (log_prior_prec,
        log_sigma_noise). Plain Adam loop over two scalars; the GGN
        eigenvalues are computed once beforehand, so a step costs O(D).

    Args:
        S_theta, theta, N, eigvals_curv, prior: See
            `neg_log_marginal_likelihood`.
        n_steps: Number of Adam steps.
        lr: Adam learning rate.
        log_prior_prec_init, log_sigma_noise_init: Starting point (in log
            space). Defaults to prior_prec = sigma_noise = 1.0.
        verbose: Print progress every 50 steps.
    Returns:
        prior_prec_opt, sigma_noise_opt: Calibrated values (NOT in log
            space).
    """
    objective = lambda p: neg_log_marginal_likelihood(
        p[0], p[1], S_theta, theta, N, eigvals_curv, prior,
    )
    return _run_mll_adam_loop(objective, n_steps, lr, log_prior_prec_init, log_sigma_noise_init, verbose)
