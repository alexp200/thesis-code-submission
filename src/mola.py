"""Mixture of (full linearized) Laplace Approximations -- MoLA.

Thin wrapper around K independently-fitted `src.models.Laplace_ODESolver`
instances (one per trained MAP model / seed). Does NOT duplicate any of the
single-component Laplace math (GGN, eigenbasis, MLL calibration, the
trajectory-space pushforward) -- every component quantity is obtained by
calling the existing `mean_linearized`/`std_linearized` methods on each
component and combining the results with plain mixture-moment formulas.

Math (uniform mixture weights pi_k = 1/K, K components):

    mu_MoLA  = mean_k(mu_k)                                  -- component means
    var_within  = mean_k(var_LA,k)                           -- avg. per-component epistemic variance
    var_between = var_k(mu_k)                                -- population variance (ddof=0) of the component means
    var_MoLA = var_within + var_between                      -- law of total variance, epistemic only

`var_within`/`var_between` mirror `src.ensembles.ens_trajectory_var`'s own
`jnp.var(..., ddof=0)` convention exactly, so `var_between` reduces to
`ens_trajectory_var`'s result when the K component means are the K deep-
ensemble member predictions (see tests/test_mola.py) -- no ddof mismatch to
special-case.

Everything here is purely EPISTEMIC (parameter uncertainty only), matching
`Laplace_ODESolver.std_linearized`'s own convention (see src/laplace.py's
docstrings) -- observation noise (`sigma_noise_opt`, fit per-component, see
`Laplace_ODESolver.fit_laplace_mll`) is never added silently. Call
`predictive_var`/`predictive_std` for the version with `mean_k(sigma_n,k^2)`
folded in.

Mixture weights are uniform (pi_k = 1/K). They are stored in `self.weights` and
enter `log_predictive_density`; the moment formulas above hold for uniform weights only.
"""

import jax.numpy as jnp
import jax.scipy.special as jsp


class MoLA_ODESolver:
    """Mixture of K full-linearized-Laplace ODE solvers.

    components: list of K already-fitted `Laplace_ODESolver` instances
    (`.fit_laplace_mll()` must have been called on each beforehand -- this
    class never calls it for you, same convention as using a single
    `Laplace_ODESolver` directly).
    """

    def __init__(self, components):
        if len(components) == 0:
            raise ValueError("MoLA_ODESolver requires at least one component.")
        self.components = list(components)
        self.K = len(self.components)
        # Stored explicitly rather than hardcoded as 1/K in every formula --
        # see module docstring on how a future evidence-weighted MoLA would
        # plug in here.
        self.weights = jnp.ones(self.K) / self.K

    def _component_means(self, t_eval, x0_eval, us_eval=None):
        """(K, T, state_size) stack of mu_k = component.mean_linearized(...)."""
        return jnp.stack([
            c.mean_linearized(t_eval, x0_eval, us_eval) for c in self.components
        ])

    def _component_vars(self, t_eval, x0_eval, us_eval=None):
        """(K, T, state_size) stack of var_LA,k = component.std_linearized(...)**2."""
        return jnp.stack([
            c.std_linearized(t_eval, x0_eval, us_eval) ** 2 for c in self.components
        ])

    def _component_covs(self, t_eval, x0_eval, us_eval=None):
        """(K, n, n) stack of Sigma_k = component.std_linearized(..., cov_type="dense"),
        n = T*state_size -- the dense analogue of `_component_vars`, kept as
        a SEPARATE helper (rather than a cov_type flag on _component_vars)
        so the five marginal-only methods below (mean/var/std/
        variance_components/log_predictive_density/nll) can never
        accidentally take the expensive dense path; only `dense_cov`/
        `predictive_dense_cov` call this.

        """
        return jnp.stack([
            c.std_linearized(t_eval, x0_eval, us_eval, cov_type="dense") for c in self.components
        ])

    def mean(self, t_eval, x0_eval, us_eval=None):
        """mu_MoLA = mean_k(mu_k), shape (T, state_size). Identical to the
        deep-ensemble mean if the same K MAP models are used (see
        tests/test_mola.py)."""
        return jnp.mean(self._component_means(t_eval, x0_eval, us_eval), axis=0)

    def variance_components(self, t_eval, x0_eval, us_eval=None):
        """Decomposition of the epistemic MoLA variance into within-mode and
        between-mode parts, shape (T, state_size) each.

        Returns:
            {"total": var_within + var_between, "within": var_within, "between": var_between}
        """
        mus = self._component_means(t_eval, x0_eval, us_eval)   # (K, T, state_size)
        vars_ = self._component_vars(t_eval, x0_eval, us_eval)  # (K, T, state_size)

        var_within = jnp.mean(vars_, axis=0)
        var_between = jnp.var(mus, axis=0)  # population variance (ddof=0), see module docstring
        return {
            "total": var_within + var_between,
            "within": var_within,
            "between": var_between,
        }

    def var(self, t_eval, x0_eval, us_eval=None):
        """var_total = var_within + var_between, purely epistemic. Shape (T, state_size)."""
        return self.variance_components(t_eval, x0_eval, us_eval)["total"]

    def std(self, t_eval, x0_eval, us_eval=None):
        """sqrt(max(var_total, 0)), purely epistemic. Shape (T, state_size)."""
        return jnp.sqrt(jnp.maximum(self.var(t_eval, x0_eval, us_eval), 0.0))

    def dense_cov(self, t_eval, x0_eval, us_eval=None):
        """Sigma_total = mean_k(Sigma_k) + Cov_k(mu_k), flattened over
        (time, state) into a single (T*state_size, T*state_size) matrix --
        the DENSE (full, non-diagonal) covariance, as opposed to `var`/`std`'s
        marginal-only diagonal. SECONDARY / expensive (O((T*state_size)^2)),
        matching `Laplace_ODESolver`'s own `pushforward_linearized_cov_fullr`.
        Only call this for a small evaluation horizon; prefer `var`/`std` for
        routine use (see module/class docstrings). Named with the "dense_"
        prefix specifically so it reads unambiguously at the call site (e.g.
        an active-learning acquisition strategy computing
        `jnp.trace(mola.dense_cov(...))` or `jnp.linalg.slogdet(mola.
        dense_cov(...))`) -- never confuse this with the marginal `var`.

        Purely EPISTEMIC (no observation noise) -- see `predictive_dense_cov`
        for the version with `mean_k(sigma_n,k^2)` added on the diagonal,
        kept as a genuinely separate method (not a flag here) so callers
        that need the pure epistemic covariance (e.g. an active-learning
        acquisition strategy) never risk observation noise leaking in.

        Both terms are EXACT: `mean_k(Sigma_k)` uses each component's own
        real dense covariance (`Laplace_ODESolver.std_linearized(...,
        cov_type="dense")`), not a diagonal approximation from its marginal
        variances -- so within-component cross-timestep/cross-state
        correlations are preserved, not just the between-component term
        (Cov_k(mu_k), the full covariance of the K component MEAN
        trajectories), which was already exact before this.
        """
        covs = self._component_covs(t_eval, x0_eval, us_eval)   # (K, n, n), n = T*state_size
        mus = self._component_means(t_eval, x0_eval, us_eval)   # (K, T, state_size)

        within_cov = jnp.mean(covs, axis=0)  # (n, n) -- exact, see docstring
        mus_flat = mus.reshape(self.K, -1)
        between_cov = jnp.cov(mus_flat, rowvar=False, ddof=0)  # (n, n), population convention -- see module docstring

        return within_cov + between_cov

    ## ------------------------------------------------------------ ##
    ## Observation noise (predictive, NOT epistemic) -- see module docstring
    ## ------------------------------------------------------------ ##

    def _sigma_noise2_mean(self):
        """mean_k(sigma_n,k^2), a scalar (one calibrated sigma_noise per
        component, see Laplace_ODESolver.fit_laplace_mll)."""
        sigma_n2 = jnp.stack([c.sigma_noise_opt ** 2 for c in self.components])
        return jnp.mean(sigma_n2)

    def sigma_noise_mean(self):
        """sqrt(mean_k(sigma_n,k^2)) -- the MoLA-level analogue of a single
        `Laplace_ODESolver`'s own `sigma_noise_opt`, for callers (e.g.
        the `sigma_noise` argument of the criteria in `src.active_learning`) that need
        one scalar observation-noise level to combine with `var`/`dense_cov`
        rather than reaching into the private `_sigma_noise2_mean`."""
        return jnp.sqrt(self._sigma_noise2_mean())

    def predictive_var(self, t_eval, x0_eval, us_eval=None):
        """var_MoLA,predictive = var_within + var_between + mean_k(sigma_n,k^2).
        Explicitly separate from `var` (epistemic-only) -- see module docstring."""
        return self.var(t_eval, x0_eval, us_eval) + self._sigma_noise2_mean()

    def predictive_std(self, t_eval, x0_eval, us_eval=None):
        return jnp.sqrt(jnp.maximum(self.predictive_var(t_eval, x0_eval, us_eval), 0.0))

    def predictive_dense_cov(self, t_eval, x0_eval, us_eval=None):
        """Sigma_MoLA,predictive = dense_cov(...) + mean_k(sigma_n,k^2) * I.
        Explicitly separate from `dense_cov` (epistemic-only) -- see module
        and `dense_cov`'s docstrings for why the two are kept as distinct
        methods rather than a flag: observation noise is i.i.d. across
        time/state (uncorrelated), so it only ever adds to the DIAGONAL,
        never to the off-diagonal (cross-timestep/cross-state) entries
        `dense_cov` already computes -- same convention as `predictive_var`
        adding mean_k(sigma_n,k^2) once, not K times (see that method's
        docstring).
        """
        Sigma = self.dense_cov(t_eval, x0_eval, us_eval)  # (n, n)
        n = Sigma.shape[0]
        return Sigma + self._sigma_noise2_mean() * jnp.eye(n)

    ## ------------------------------------------------------------ ##
    ## Mixture log-likelihood -- NOT a single-Gaussian approximation
    ## ------------------------------------------------------------ ##

    def log_predictive_density(self, y_true, t_eval, x0_eval, us_eval=None, predictive=True, min_std=1e-6):
        """Exact Gaussian-mixture log density at each element of y_true,
        log p(y) = logsumexp_k(log(pi_k) + log N(y; mu_k, variance_k)),
        via jax.scipy.special.logsumexp for numerical stability -- NOT
        computed by moment-matching a single N(mu_MoLA, var_MoLA) (see module
        docstring / the project's math spec for why that would be wrong:
        the mixture is not itself Gaussian).

        Per-component variance_k is `var_LA,k` (epistemic) or `var_LA,k +
        sigma_n,k^2` (predictive, default) depending on `predictive` --
        matching `predict`/`predictive_var`'s own epistemic-vs-predictive
        distinction. The likelihood is evaluated marginally per (time, state) element and averaged;
        no joint-trajectory likelihood is used.

        `variance_k` is clipped to `min_std**2` (default min_std = 1e-6) before use:
        std_linearized returns exactly 0 at points where the linearized pushforward
        has no epistemic uncertainty (e.g. t=0, where mean_linearized is pinned to
        x0_eval regardless of theta), which would otherwise divide by zero / take
        log(0) here.

        Args:
            y_true: ground truth, shape (T, state_size) (matching mean_linearized's output).
            t_eval, x0_eval, us_eval: forwarded to each component.
            predictive: if True (default), add mean_k(sigma_n,k^2) to the k'th
                component variance (NOT a mixture-averaged sigma_noise --
                each component keeps its OWN sigma_n,k^2, since the mixture
                density genuinely differs per-component).
            min_std: minimum std used in each mixture component's density
                (numerical floor, see above).

        Returns:
            Mean log-density (scalar) over every element of y_true.
        """
        mus = self._component_means(t_eval, x0_eval, us_eval)   # (K, T, state_size)
        vars_ = self._component_vars(t_eval, x0_eval, us_eval)  # (K, T, state_size)
        if predictive:
            sigma_n2 = jnp.stack([c.sigma_noise_opt ** 2 for c in self.components])  # (K,)
            vars_ = vars_ + sigma_n2[:, None, None]
        vars_ = jnp.maximum(vars_, min_std ** 2)

        log_weights = jnp.log(self.weights)[:, None, None]  # (K, 1, 1)
        y_true_b = y_true[None, ...]  # (1, T, state_size), broadcasts against (K, T, state_size)

        log_normal = -0.5 * (
            jnp.log(2 * jnp.pi * vars_) + (y_true_b - mus) ** 2 / vars_
        )  # (K, T, state_size)
        log_mixture = jsp.logsumexp(log_weights + log_normal, axis=0)  # (T, state_size)
        return jnp.mean(log_mixture)

    def nll(self, y_true, t_eval, x0_eval, us_eval=None, predictive=True, min_std=1e-6):
        """-log_predictive_density(...) -- the mixture NLL, see that method's docstring."""
        return -self.log_predictive_density(y_true, t_eval, x0_eval, us_eval, predictive=predictive, min_std=min_std)
