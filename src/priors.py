"""Prior over the flat parameter vector of the Laplace approximation.

All experiments use an isotropic Gaussian prior p(theta | alpha) = N(0, alpha^-1 I) on the flat raw parameter
vector. The class only describes the prior at the Laplace expansion point; the calibrated hyperparameters
(alpha, sigma_noise) are passed in at call time (see src.laplace.calibrate_mll).

For an isotropic prior the prior curvature alpha * I is diagonal in every orthonormal basis, in particular in the
eigenbasis of the GGN. This is what allows closed-form posterior variances and marginal likelihoods from the GGN
eigenvalues alone (src.laplace.posterior_eigen_weights, src.laplace.neg_log_marginal_likelihood).
"""

import jax.numpy as jnp


class IsotropicGaussianPrior:
    """p(theta | alpha) = N(0, alpha^-1 I) over the full D-dimensional raw parameter vector."""

    def __init__(self, D: int):
        self.D = D

    def log_density(self, theta: jnp.ndarray, alpha) -> jnp.ndarray:
        """log N(theta; 0, alpha^-1 I) without the constant -0.5 * D * log(2 pi), which does not
        affect the gradient of the marginal likelihood with respect to alpha or sigma_noise."""
        theta_sq_norm = jnp.sum(theta ** 2)
        return -0.5 * alpha * theta_sq_norm + 0.5 * self.D * jnp.log(alpha)

    def curvature(self, theta: jnp.ndarray, alpha) -> jnp.ndarray:
        """Diagonal of the prior curvature H_prior = alpha * I, shape (D,)."""
        return alpha * jnp.ones(self.D)

    def fast_path_diag(self):
        """Fixed diagonal d with H_prior(alpha) = alpha * d, here all ones."""
        return jnp.ones(self.D)

    def is_isotropic(self) -> bool:
        return True
