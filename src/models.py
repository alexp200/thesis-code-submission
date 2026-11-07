"""Model wrappers for uncertainty quantification of ODE solvers (dynax NeuralODE and sPHNN).

    Laplace_ODESolver, Laplace_NeuralODESolver, Laplace_sPHNNSolver
        Linearized Laplace approximation of the whole ODE solve. The flat parameter vector, the GGN of the
        flattened trajectory output, the marginal-likelihood calibration of (alpha, sigma_n) and the
        linearized pushforward (mean_linearized, std_linearized, marginal_variance_eigen_jvp) are built on
        the functions of src.laplace. For the sPHNN the parameters are the raw (unconstrained) coordinates
        and non-trainable leaves such as the equilibrium x* are excluded from the posterior.
    SWAG_ODESolver, SWAG_NeuralODESolver, SWAG_sPHNNSolver
        SWAG on the same flat parameter vector (src.swag): snapshot collection, sampling, prediction.
    softplus_FICNN
        Replaces the non-negativity constraint of a FICNN by a softplus reparametrization.
"""

import equinox as eqx
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrd
import klax
import optax
from jaxtyping import Array
from klax import Unwrappable
import dynax
import jax.flatten_util as fu
from src import laplace as lp
from src import ensembles as ens
from src import priors
from src import swag as sw
from klax import NonTrainable
from klax._wrappers import Constraint

### -------------- klax.nn.FICNN reparametrization ------------------------------

# differentiable non-negativity unwrappable
class SoftplusPositive(Unwrappable):
    parameter: Array

    def __init__(self, parameter: Array):
        eps = 1e-6
        parameter = jnp.clip(parameter, min=eps)
        self.parameter = jnp.log(-1 + jnp.exp(parameter))  # inverse softplus

    def unwrap(self) -> Array:
        return jax.nn.softplus(self.parameter)

# utility to make example scripts compact & standardized
def softplus_FICNN(ficnn: klax.nn.FICNN) -> klax.nn.FICNN:
    """Convert a klax.nn.FICNN to use SoftplusPositive instead of NonNegative."""
    # Swap the NonNegative constraint with the Positive unwrappable
    return jax.tree.map(
        lambda leaf: SoftplusPositive(leaf.parameter)
        if isinstance(leaf, klax.NonNegative)
        else leaf,
        ficnn,
        is_leaf=lambda leaf: isinstance(leaf, klax.NonNegative),
    )


### -------------- Laplace_ODESolver ------------------------------


# wrapper for a dynax.NeuralODE includes Laplace utilities for making main scripts more compact
class Laplace_ODESolver():
    map_model: dynax.ODESolver
    t_train: Array
    x0_train: Array
    y_train: Array
    us_train: Array


    def __init__(self, map_model, t_train, x0_train, y_train, us_train=None):
        self.check_model_wrapping(map_model)
        self.map_model = map_model
        self.t_train = t_train
        self.x0_train = x0_train
        self.y_train = y_train
        self.us_train = us_train
        self.is_forced = us_train is not None

        # expects state shape (n_trajectories, n_timesteps, state_size)
        self.state_size = self.y_train.shape[-1]  # output state size (the latent dimension for POD-reduced problems)

        ## ----------- Flatten MAP parameters (shared by both methods) ------------ ##
        self.params_tree, self.static_solver = self.partition_model(self.map_model)
        self.p_map, self.unravel_solver = fu.ravel_pytree(self.params_tree)
        self.D = self.p_map.shape[0]
        print(f"Parameter count D = {self.D}")

        # Isotropic Gaussian prior over the flat parameter vector p_map (see src/priors.py).
        self.prior = priors.IsotropicGaussianPrior(D=self.D)

        ## ----------- Compute GGN (prior-independent, safe to build now) ------------ ##
        print("\nComputing GGN of the linearized ODESolve (differentiating through diffeqsolve)...")
        # The likelihood is Gaussian on the solved trajectory (squared-error loss on the states), so the GGN is
        # J^T J with J the Jacobian of the flattened ODE-solve output with respect to the parameters.
        self.GGN = lp.compute_ggn_vector(self.f_solve_flat, self.p_map, self.train_inputs(), hessian_scale=1.0)

    def _prepare_eigenbasis(self):
        """Eigendecomposition of the GGN (self.eigvals_ggn, self.eigvecs_ggn). It is computed once and
        reused by the marginal-likelihood calibration and by the posterior."""
        if getattr(self, "eigvecs_ggn", None) is not None:
            return
        self.eigvals_ggn, self.eigvecs_ggn = lp.GGN_eigendecompose(self.GGN, THRESHOLD=1e-3)

    ## ============================================================ ##
                ## Helper functions for entire class ##
    ## ============================================================ ##

    def check_model_wrapping(self, model):
        """Overridable pre-partition sanity check. Default: no-op, since a
        plain NeuralODE has no wrapped leaves to lose track of. Subclasses
        whose partition_model() filters on a specific wrapper type (e.g.
        Laplace_sPHNNSolver filtering on NonTrainable/Constraint to exclude
        x* from the posterior) MUST override this to verify that wrapper is
        actually still present -- otherwise a caller who accidentally passes
        an already klax.finalize()'d / klax.unwrap()'d model silently gets a
        partition_model() filter that matches nothing, and things it should
        exclude from the posterior (e.g. a fixed equilibrium point) end up
        wrongly treated as trainable. See Laplace_sPHNNSolver's override for
        the concrete check.
        """
        pass

    def partition_model(self, model):
        """makes model partitioning overwritable for customization in
        NeuralODE vs sPHNN cases.
        """
        return eqx.partition(model, eqx.is_array)

    def rebuild(self, p_flat):
        """makes model rebuilding overwritable for customization in
            NeuralODE vs sPHNN cases.
        """
        return eqx.combine(self.unravel_solver(p_flat), self.static_solver)

    
    def solve(self, solver, t, x0, us=None):
        """helper for adjusting solver depending on forcing"""
        if us is None:
            return solver(t, x0)
        return solver(t, x0, us=us)

    def train_inputs(self):
        """helper to adjust input structure depending on forcing"""
        if self.us_train is None:
            return self.x0_train
        return (self.x0_train, self.us_train)

    def _prepare_mll_inputs(self):
        """Eigendecomposition of the GGN and the per-component quantities (S_theta, theta, N) that
        neg_log_marginal_likelihood / calibrate_mll need (the part of fit_laplace_mll before the Adam loop).

        Returns:
            dict with keys "S_theta", "theta_for_mll", "N", "eigvals_ggn", "prior", i.e. the
            non-hyperparameter arguments of neg_log_marginal_likelihood.
        """
        self._prepare_eigenbasis()
        preds_flat = jax.vmap(lambda inp: self.f_solve_flat(self.p_map, inp))(self.train_inputs())
        preds_train_solve = preds_flat.reshape(self.y_train.shape)
        S_theta_solve = float(jnp.sum((preds_train_solve - self.y_train) ** 2))
        N_solve = self.y_train.size  # n_trajectories * n_timesteps * state_size
        return {
            "S_theta": S_theta_solve,
            "theta_for_mll": self.p_map,
            "N": N_solve,
            "eigvals_ggn": self.eigvals_ggn,
            "prior": self.prior,
        }

    def fit_laplace_mll(self):
        """Calibrates the prior precision (prior_prec_opt) and the observation noise (sigma_noise_opt) by
        maximizing the Laplace marginal likelihood with Adam, and sets the full-spectrum active_idx.
        Call once after instantiation to obtain calibrated uncertainties.

        With S_theta the training SSE at the MAP solve (sum over all trajectories, as used for the GGN),
        N the number of scalar training targets (n_trajectories * n_timesteps * state_size) and
        eigvals_curv the GGN eigenvalues, the negative log marginal likelihood is
        -log p(D | alpha, sigma_n^2) = S_theta / (2 sigma_n^2) + alpha / 2 ||p_map||^2 + N / 2 log(2 pi sigma_n^2)
                                       - D / 2 log(alpha) + 1/2 log det(A).
        """
        mll_inputs = self._prepare_mll_inputs()

        print("\nCalibrating prior_prec via marginal likelihood...")
        self.prior_prec_opt, self.sigma_noise_opt = lp.calibrate_mll(
            mll_inputs["S_theta"], mll_inputs["theta_for_mll"], mll_inputs["N"],
            mll_inputs["eigvals_ggn"], mll_inputs["prior"], n_steps=200, lr=1e-1,
        )
        print(f"  Calibrated prior_prec  = {float(self.prior_prec_opt):.5f}")
        print(f"  Calibrated sigma_noise = {float(self.sigma_noise_opt):.5f}")

        self.active_idx = jnp.arange(self.D)  # full spectrum; GGN_eigendecompose already reports the low-eigenvalue count

    ## ============================================================ ##
    ## Linearize the entire ODESolve (Ott et al. paper Eq. 5-6)
    ## Closed-form Gaussian directly over trajectory outputs -- no
    ## sampling, no repeated integration at inference time.
    ## ============================================================ ##
    
    def f_solve_flat(self, p_flat, input):
        """ODESolve(f_theta, x0) evaluated over the fixed training grid t, output
        flattened to (len(t) * state_size,) since compute_ggn_vector expects a
        single (out_dim,) return per call -- here the whole trajectory over t is
        treated as "out_dim" for one x0 at a time. Used only to build the GGN
        (which needs the Jacobian over the training horizon); evaluation for
        plotting uses the f_solve_flat_general closure below instead.
        """
        solver = self.rebuild(p_flat)

        if self.us_train is None:
            x0 = input
            us = None
        else:
            x0, us = input

        traj = self.solve(solver, self.t_train, x0, us) # trajectory including augmented states if present
        obs = traj[..., :self.state_size] # removes augmented states if present, 
                                          # important for later jacobian computation
        return obs.reshape(-1)  # (len(t) * state_size,)

    def f_solve_flat_general(self, p_flat, t_eval, x0, us=None):
        """ODESolve(f_theta, t_eval, x0) evaluated over a flexible evaluation grid t, output
        flattened to (len(t) * state_size,) since std_linearized expects this for jacobian computation --
        here the whole trajectory over t is treated as "out_dim" for one x0 at a time. 
        Used for computing the linearization jacobian in the pushforward eq.
        """
        m = self.rebuild(p_flat)
        traj = self.solve(m, t_eval, x0, us) # trajectory including augmented states if present
        return traj[..., :self.state_size].reshape(-1)  # removes augmented states if present, 
                                                        # important for later jacobian computation

    def _posterior_directions_and_weights(self):
        """Represent the calibrated posterior as U diag(w) U.T."""
        if getattr(self, "active_idx", None) is None:
            raise RuntimeError("The Laplace posterior is not prepared. Call fit_laplace_mll() first.")
        active, weights = lp.posterior_eigen_weights(
            self.eigvals_ggn,
            self.active_idx,
            self.prior_prec_opt,
            self.prior,
            sigma_noise=self.sigma_noise_opt,
        )
        return self.eigvecs_ggn[:, active], weights

    def _marginal_variance_eigen_jvp_prepared(
            self,
            t_eval,
            x0_eval,
            us_eval,
            directions,
            weights,
            *,
            chunk_size,
            ):
        """Evaluate one trajectory using a prepared posterior basis."""
        def trajectory(p_flat):
            solver = self.rebuild(p_flat)
            # jax.linearize needs forward-mode derivatives; diffrax's default (reverse-mode) adjoint does not support them.
            forward_solver = eqx.tree_at(
                lambda ode_solver: ode_solver.adjoint,
                solver,
                diffrax.ForwardMode(),
            )
            solved = self.solve(
                forward_solver,
                t_eval,
                x0_eval,
                us_eval,
            )
            return solved[..., :self.state_size].reshape(-1)

        variance_flat = lp.chunked_eigen_jvp_marginal_variance(
            trajectory,
            self.p_map,
            directions,
            weights,
            chunk_size=chunk_size,
        )
        return variance_flat.reshape(len(t_eval), self.state_size)

    def marginal_variance_eigen_jvp(
            self,
            t_eval,
            x0_eval,
            us_eval=None,
            *,
            chunk_size=16,
            ):
        """Compute exact linearized marginal variance without forming J."""
        directions, weights = self._posterior_directions_and_weights()
        return self._marginal_variance_eigen_jvp_prepared(
            t_eval,
            x0_eval,
            us_eval,
            directions,
            weights,
            chunk_size=chunk_size,
        )

    def make_marginal_variance_eigen_jvp(
            self,
            t_eval,
            *,
            chunk_size=128,
            ):
        """Build a compiled, reusable exact marginal-variance scorer."""
        directions, weights = self._posterior_directions_and_weights()

        if self.is_forced:
            def scorer(x0_eval, us_eval):
                return self._marginal_variance_eigen_jvp_prepared(
                    t_eval,
                    x0_eval,
                    us_eval,
                    directions,
                    weights,
                    chunk_size=chunk_size,
                )
        else:
            def scorer(x0_eval):
                return self._marginal_variance_eigen_jvp_prepared(
                    t_eval,
                    x0_eval,
                    None,
                    directions,
                    weights,
                    chunk_size=chunk_size,
                )

        return jax.jit(scorer)
    
    def std_linearized(self, t_eval, x0_eval, us_eval=None, cov_type="marginal"):
        """cov_type: "marginal" (default) returns per-timestep/per-state std,
        shape (len(t_eval), state_size). "dense" returns the full
        (len(t_eval)*state_size, len(t_eval)*state_size) covariance
        matrix instead, including cross-covariances between all output
        entries (e.g. across timesteps) -- see
        src.laplace.pushforward_linearized_cov_fullr.
        """
        J_solve_plot = jax.jacobian(lambda p: self.f_solve_flat_general(p, t_eval, x0_eval, us_eval))(self.p_map)  # (len(t_plot)*state_size, D)
        alpha_solve = J_solve_plot @ self.eigvecs_ggn  # (len(t_plot)*state_size, D)
        if cov_type == "dense":
            return lp.pushforward_linearized_cov_fullr(
                alpha_solve, self.eigvals_ggn, self.active_idx,
                self.prior_prec_opt, self.prior, sigma_noise=self.sigma_noise_opt,
            )
        std_solve_flat = lp.pushforward_linearized_std(
            alpha_solve, self.eigvals_ggn, self.active_idx,
            self.prior_prec_opt, self.prior, sigma_noise=self.sigma_noise_opt,
        )
        return std_solve_flat.reshape(len(t_eval), self.state_size)

    def mean_linearized(self, t_eval, x0_eval, us_eval=None):
        """The mean of linearized Laplace is just the MAP model itself.
        Goes through rebuild(p_map) rather than calling self.map_model
        directly -- for sPHNN, self.map_model is intentionally kept in its
        raw, un-klax.unwrap()'d form (see check_model_wrapping), so it is
        not directly callable; rebuild() applies klax.unwrap() internally.
        For NODE, rebuild(p_map) is equivalent to self.map_model itself."""
        traj = self.solve(self.rebuild(self.p_map), t_eval, x0_eval, us_eval)
        return traj[..., :self.state_size]  # removes augmented states if present,
                                            # keeps shape consistent with std_linearized

    ## ============================================================ ##
    ## Vector field of the model (used for the vector-field uncertainty)
    ## ============================================================ ##

    def f_field(self, p_flat, z, u=None):
        """Raw vector field f_theta(z) = NeuralODE(t=None, z) -- no integration,
        matching compute_ggn_vector's existing pointwise-evaluation assumption
        directly (this is exactly the shape it was already designed for)."""
        solver = self.rebuild(p_flat)
        return solver.func(None, z, u=u)  # (state_size,) -- solver.func is the NODE


class Laplace_NeuralODESolver(Laplace_ODESolver):
    """Customization of Laplace_ODESolver to NODE structure.
       Currently no adjustments required.
    """
    pass

class Laplace_sPHNNSolver(Laplace_ODESolver):
    """Customization of Laplace_ODESolver to sPHNN structure."""
    ## ============================================================ ##
      ## Customization overrides of parent class hooks for sPHNN ##
    ## ============================================================ ##

    def check_model_wrapping(self, model):
        """partition_model below filters on (NonTrainable, Constraint) to
        keep x* (dynax.ConvexNormalization's fixed equilibrium, wrapped in
        NonTrainable) out of the trainable/posterior parameters. That
        filter only works if those wrapper instances are still PRESENT in
        `model` -- if the caller passed a model that was already run through
        klax.finalize()/klax.unwrap() (which resolve and remove exactly
        these wrappers), the filter silently matches nothing, and x* gets
        folded into p_map as if it were an ordinary trainable/posterior
        parameter. This raises loudly instead of letting that happen
        quietly. Fix at the call site: pass klax.finalize's raw pre-finalize
        model as `map_model` (i.e. do not finalize/unwrap before handing the
        model to this class) -- rebuild() calls
        klax.unwrap() internally wherever a callable model is needed.
        """
        if not klax.contains_unwrappables(model):
            raise ValueError(
                "Laplace_sPHNNSolver received a model with no remaining "
                "klax.Unwrappable instances (e.g. the NonTrainable wrapping "
                "x*). This model was likely already passed through "
                "klax.finalize() or klax.unwrap() before being handed to "
                "this class -- doing so silently breaks the x*-exclusion "
                "filter in partition_model, since it filters on wrapper "
                "TYPES that no longer exist in an already-finalized model. "
                "Pass the pre-finalize model instead; rebuild() calls "
                "klax.unwrap() internally wherever a callable model is needed."
            )

    def partition_model(self, model):
        """custom sPHNN partitioning -- Contains constraints and NonTrainables."""
        filter_spec = jax.tree_util.tree_map(
            lambda leaf: eqx.is_array(leaf) and not isinstance(leaf, (NonTrainable, Constraint)),
            model,
            is_leaf=lambda leaf: isinstance(leaf, (NonTrainable, Constraint)),
        )
        return eqx.partition(
            model,
            filter_spec,
            is_leaf=lambda leaf: isinstance(leaf, (NonTrainable, Constraint)),
        )

    def rebuild(self, p_flat):
        """custom sPHNN rebuilding."""
        return klax.unwrap(eqx.combine(self.unravel_solver(p_flat), self.static_solver))


### -------------- SWAG_ODESolver ------------------------------


class SWAG_ODESolver:
    """Small ODE wrapper around src.swag's flat-parameter SWAG core.

    The class mirrors Laplace_ODESolver's model hooks, but it does not build
    a GGN or run MLL calibration. It only continues training in flat
    parameter space, collects SWAG snapshots, samples parameters from the
    resulting Gaussian, and evaluates those samples as an ODE ensemble.
    """

    map_model: dynax.ODESolver
    t_train: Array
    x0_train: Array
    y_train: Array
    us_train: Array

    def __init__(self, map_model, t_train, x0_train, y_train, us_train=None):
        self.check_model_wrapping(map_model)
        self.map_model = map_model
        self.t_train = t_train
        self.x0_train = x0_train
        self.y_train = y_train
        self.us_train = us_train
        self.is_forced = us_train is not None
        self.state_size = self.y_train.shape[-1]

        self.params_tree, self.static_solver = self.partition_model(self.map_model)
        self.p_init, self.unravel_solver = fu.ravel_pytree(self.params_tree)
        self.p_current = self.p_init
        self.D = self.p_init.shape[0]
        self.swag_state = None
        self.theta_samples = None
        self.keep_mask = None
        print(f"[SWAG] Parameter count D = {self.D}")

    def check_model_wrapping(self, model):
        """Overridable sanity check; plain NeuralODE needs no special check."""
        pass

    def partition_model(self, model):
        return eqx.partition(model, eqx.is_array)

    def rebuild(self, p_flat):
        return eqx.combine(self.unravel_solver(p_flat), self.static_solver)

    def build_sample_ensemble(self, theta_samples):
        return ens.ensemble_from_flat_samples(
            theta_samples, self.unravel_solver, self.static_solver
        )

    def solve(self, solver, t, x0, us=None):
        if us is None:
            return solver(t, x0)
        return solver(t, x0, us=us)

    def train_inputs(self, batch_idx=None):
        if batch_idx is None:
            x0 = self.x0_train
            us = self.us_train
        else:
            x0 = self.x0_train[batch_idx]
            us = None if self.us_train is None else self.us_train[batch_idx]
        if us is None:
            return x0
        return (x0, us)

    def f_solve_flat(self, p_flat, input):
        solver = self.rebuild(p_flat)
        if self.us_train is None:
            x0, us = input, None
        else:
            x0, us = input
        traj = self.solve(solver, self.t_train, x0, us)
        return traj[..., :self.state_size].reshape(-1)

    def trajectory_loss(self, p_flat, l2=0.0, batch_idx=None):
        y_train = self.y_train if batch_idx is None else self.y_train[batch_idx]
        preds_flat = jax.vmap(
            lambda inp: self.f_solve_flat(p_flat, inp)
        )(self.train_inputs(batch_idx))
        preds = preds_flat.reshape(y_train.shape)
        return jnp.mean((preds - y_train) ** 2) + l2 * jnp.sum(p_flat ** 2)

    def fit_swag(
            self,
            burnin_steps=0,
            swag_steps=1_000,
            burnin_lr=1e-3,
            swag_lr=3e-5,
            swag_freq=20,
            swag_optimizer="sgd",
            l2=0.0,
            reset_opt_every=None,
            batch_size=None,
            key=jrd.key(0),
            clip_grad_norm=None,
            noise_scale=None,
            log_every=200,
            ):
        """Continue training and collect a full-covariance SWAG posterior.

        Args:
            batch_size: optional number of training trajectories per SGD
                step. None keeps the old full-batch behavior. For SWAG,
                mini-batches are the standard source of stochasticity; this
                batches over trajectories, not over timesteps.
            key: PRNG key used for minibatch sampling (if batch_size is not
                None) and/or gradient noise (if noise_scale is set) --
                shared and split internally between the two uses.
            clip_grad_norm: optional global gradient-norm clipping threshold.
                Useful for ODE models where a small parameter step can make
                the solver hit max_steps.
            noise_scale: optional std of isotropic Gaussian noise added to
                the gradient at every step (see src.swag.add_gradient_noise
                for the full rationale) -- None/0 keeps the old behavior
                exactly. Useful when there are very few training
                trajectories: minibatch SGD/Adam then only ever walks along
                as many fixed gradient directions as there are trajectories,
                which can collapse SWAG's covariance onto a single
                eigendirection if those per-trajectory gradients are close
                to antiparallel at the MAP point. Pick noise_scale relative
                to the ACTUAL per-trajectory gradient norms in your problem
                (e.g. from a trial fit_swag run's printed loss/grad
                behavior) -- a value orders of magnitude below the real
                gradient norm has no visible effect, since it is simply
                swamped by the real gradient.
        """
        p = self.p_current
        n_train = self.x0_train.shape[0]
        if reset_opt_every is not None and noise_scale is not None and noise_scale > 0:
            raise NotImplementedError(
                "fit_swag(reset_opt_every=...) is not compatible with noise_scale yet. "
                "The reset path would need to rebuild the full optax chain state, "
                "including the gradient-noise transform."
            )
        if batch_size is not None:
            if not 0 < batch_size <= n_train:
                raise ValueError("batch_size must be in [1, n_train].")
            batch_size = int(batch_size)

        losses_burnin = []
        if burnin_steps > 0:
            opt_burnin = optax.adam(burnin_lr)
            state_burnin = opt_burnin.init(p)

            @jax.jit
            def step_burnin(p_flat, opt_state):
                loss, grads = jax.value_and_grad(
                    lambda q: self.trajectory_loss(q, l2=l2)
                )(p_flat)
                updates, opt_state = opt_burnin.update(grads, opt_state, p_flat)
                return optax.apply_updates(p_flat, updates), opt_state, loss

            @jax.jit
            def step_burnin_batch(p_flat, opt_state, batch_idx):
                loss, grads = jax.value_and_grad(
                    lambda q: self.trajectory_loss(q, l2=l2, batch_idx=batch_idx)
                )(p_flat)
                updates, opt_state = opt_burnin.update(grads, opt_state, p_flat)
                return optax.apply_updates(p_flat, updates), opt_state, loss

            print(f"[SWAG] Burn-in for {burnin_steps} steps...")
            for step in range(burnin_steps):
                if batch_size is None:
                    p, state_burnin, loss = step_burnin(p, state_burnin)
                else:
                    key, step_key = jrd.split(key)
                    batch_idx = jrd.choice(step_key, n_train, shape=(batch_size,), replace=False)
                    p, state_burnin, loss = step_burnin_batch(p, state_burnin, batch_idx)
                losses_burnin.append(float(loss))
                if log_every and step % log_every == 0:
                    print(f"  burn-in step {step:6d}  loss={float(loss):.6g}")

        if swag_optimizer == "sgd":
            base_optimizer = optax.sgd(swag_lr)
        elif swag_optimizer == "adam":
            base_optimizer = optax.adam(swag_lr)
        else:
            raise ValueError("swag_optimizer must be either 'sgd' or 'adam'.")

        swag_transform = sw.swag_full(freq=swag_freq)
        transforms = []
        if noise_scale is not None and noise_scale > 0:
            # BEFORE clip_grad_norm (not after) -- noise must be part of what
            # gets clipped, otherwise clip_grad_norm bounds the real gradient
            # to norm clip_grad_norm and then noise_scale*N(0,I) is added on
            # top completely unbounded (norm ~= noise_scale*sqrt(D)), which
            # can dwarf the real gradient by 1-2 orders of magnitude and blow
            # up the loss instead of producing a bounded random-walk step.
            key, noise_key = jrd.split(key)
            transforms.append(sw.add_gradient_noise(noise_scale, noise_key))
        if clip_grad_norm is not None:
            transforms.append(optax.clip_by_global_norm(clip_grad_norm))
        transforms.extend([base_optimizer, swag_transform])
        opt_swag = optax.chain(*transforms)
        state_swag = opt_swag.init(p)

        @jax.jit
        def step_swag(p_flat, opt_state):
            loss, grads = jax.value_and_grad(
                lambda q: self.trajectory_loss(q, l2=l2)
            )(p_flat)
            updates, opt_state = opt_swag.update(grads, opt_state, p_flat)
            return optax.apply_updates(p_flat, updates), opt_state, loss

        @jax.jit
        def step_swag_batch(p_flat, opt_state, batch_idx):
            loss, grads = jax.value_and_grad(
                lambda q: self.trajectory_loss(q, l2=l2, batch_idx=batch_idx)
            )(p_flat)
            updates, opt_state = opt_swag.update(grads, opt_state, p_flat)
            return optax.apply_updates(p_flat, updates), opt_state, loss

        print(f"[SWAG] Collection phase for {swag_steps} steps with {swag_optimizer}...")
        losses_swag = []
        for step in range(swag_steps):
            if reset_opt_every is not None and step % reset_opt_every == 0:
                if clip_grad_norm is None:
                    state_swag = (base_optimizer.init(p), state_swag[-1])
                else:
                    state_swag = (state_swag[0], base_optimizer.init(p), state_swag[-1])
            if batch_size is None:
                p, state_swag, loss = step_swag(p, state_swag)
            else:
                key, step_key = jrd.split(key)
                batch_idx = jrd.choice(step_key, n_train, shape=(batch_size,), replace=False)
                p, state_swag, loss = step_swag_batch(p, state_swag, batch_idx)
            losses_swag.append(float(loss))
            if log_every and step % log_every == 0:
                print(f"  SWAG step    {step:6d}  loss={float(loss):.6g}")

        self.p_current = p
        self.swag_state = state_swag[-1]
        self.n_snapshots = int(self.swag_state.n)
        self.history = {
            "burnin_loss": losses_burnin,
            "swag_loss": losses_swag,
            "n_snapshots": self.n_snapshots,
            "swag_optimizer": swag_optimizer,
            "batch_size": batch_size,
            "clip_grad_norm": clip_grad_norm,
        }
        print(f"[SWAG] Snapshots collected: {self.n_snapshots}")
        if self.n_snapshots < 2:
            print("[SWAG] Warning: fewer than 2 snapshots, covariance sampling is not meaningful yet.")
        return self.history

    def mean(self, t_eval, x0_eval, us_eval=None):
        """SWA mean prediction. Falls back to the current parameters before fit_swag."""
        if self.swag_state is None or int(self.swag_state.n) == 0:
            p_mean = self.p_current
        else:
            p_mean = self.swag_state.mean
        traj = self.solve(self.rebuild(p_mean), t_eval, x0_eval, us_eval)
        return traj[..., :self.state_size]

    def sample_params(self, key, n_samples=64, scale=1.0, eps=1e-6):
        if self.swag_state is None:
            raise RuntimeError("fit_swag() must be called before sampling.")
        if int(self.swag_state.n) < 2:
            raise RuntimeError("SWAG needs at least 2 collected snapshots before sampling.")
        keys = jrd.split(key, n_samples)
        self.theta_samples = jax.vmap(
            lambda k: sw.sample_swag_full(k, self.swag_state, scale=scale, eps=eps)
        )(keys)
        return self.theta_samples

    def predict(
            self,
            t_eval,
            x0_eval,
            us_eval=None,
            n_samples=64,
            key=jrd.key(0),
            scale=1.0,
            eps=1e-6,
            y_ref=None,
            max_abs_value=1e6,
            ):
        """Monte-Carlo predictive mean/std from sampled SWAG ODE solvers."""
        theta_samples = self.sample_params(key, n_samples=n_samples, scale=scale, eps=eps)
        solver_ensemble = self.build_sample_ensemble(theta_samples)

        try:
            preds_all = ens.evaluate_ensemble_trajectory(
                solver_ensemble, t_eval, x0_eval, us=us_eval
            )
        except Exception as exc:
            print(f"[SWAG] Batched integration failed ({type(exc).__name__}); falling back to per-sample evaluation...")
            preds = []
            for i in range(n_samples):
                try:
                    traj = self.solve(self.rebuild(theta_samples[i]), t_eval, x0_eval, us_eval)
                    preds.append(traj[..., :self.state_size])
                except Exception:
                    preds.append(jnp.full((len(t_eval), self.state_size), jnp.nan))
            preds_all = jnp.stack(preds, axis=0)

        preds_all = preds_all[..., :self.state_size]
        reduce_axes = tuple(range(1, preds_all.ndim))
        finite_mask = jnp.all(jnp.isfinite(preds_all), axis=reduce_axes)

        if y_ref is not None:
            scale_ref = jnp.max(jnp.abs(y_ref)) + 1e-8
            magnitude_ok = jnp.max(jnp.abs(preds_all), axis=reduce_axes) < 1e3 * scale_ref
        else:
            magnitude_ok = jnp.max(jnp.abs(preds_all), axis=reduce_axes) < max_abs_value

        self.keep_mask = finite_mask & magnitude_ok
        n_survived = int(self.keep_mask.sum())
        print(f"[SWAG] {n_survived}/{n_samples} sampled ODE solvers produced finite, bounded trajectories.")
        if n_survived == 0:
            raise RuntimeError("SWAG: no sampled parameter vector produced a usable trajectory.")

        preds_keep = preds_all[self.keep_mask]
        mean = jnp.mean(preds_keep, axis=0)
        std = jnp.std(preds_keep, axis=0)
        return mean, std


class SWAG_NeuralODESolver(SWAG_ODESolver):
    """SWAG wrapper for a plain dynax.NeuralODE inside dynax.ODESolver."""
    pass


class SWAG_sPHNNSolver(SWAG_ODESolver):
    """SWAG wrapper for sPHNN ODESolvers, matching Laplace_sPHNNSolver hooks."""

    def check_model_wrapping(self, model):
        if not klax.contains_unwrappables(model):
            raise ValueError(
                "SWAG_sPHNNSolver received a model with no remaining "
                "klax.Unwrappable instances. Pass the pre-finalize model; "
                "rebuild() unwraps internally for training and prediction."
            )

    def partition_model(self, model):
        filter_spec = jax.tree_util.tree_map(
            lambda leaf: eqx.is_array(leaf) and not isinstance(leaf, (NonTrainable, Constraint)),
            model,
            is_leaf=lambda leaf: isinstance(leaf, (NonTrainable, Constraint)),
        )
        return eqx.partition(
            model,
            filter_spec,
            is_leaf=lambda leaf: isinstance(leaf, (NonTrainable, Constraint)),
        )

    def rebuild(self, p_flat):
        return klax.unwrap(eqx.combine(self.unravel_solver(p_flat), self.static_solver))

    def build_sample_ensemble(self, theta_samples):
        params_trees = jax.vmap(self.unravel_solver)(theta_samples)
        return eqx.filter_vmap(
            lambda params_tree: klax.unwrap(eqx.combine(params_tree, self.static_solver)),
            in_axes=eqx.if_array(0),
        )(params_trees)
