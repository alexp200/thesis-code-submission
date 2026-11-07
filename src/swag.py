"""Stochastic Weight Averaging - Gaussian (SWAG), full covariance variant.

Maddox et al. (2019), "A Simple Baseline for Bayesian Uncertainty in Deep
Learning" (NeurIPS 2019). The paper's own SWAG uses a diagonal + low-rank
covariance approximation for memory reasons (Section 3.4), needed for
million-parameter vision models. For the small networks used throughout this
repo (hundreds to low thousands of parameters), the FULL DxD sample
covariance is cheap to store and compute directly, so that is all this
module implements -- no diagonal-only or low-rank+diagonal variant.

Design, mirroring src/laplace.py's conventions:
  - Operates on a FLAT parameter vector (D,), like src/laplace.py's p_flat --
    not a PyTree. Flatten your model once with jax.flatten_util.ravel_pytree
    before use, same as every other script in this repo already does.
  - swag_full is the optax.GradientTransformation that collects the running
    mean/covariance during training, sample_swag_full draws one posterior
    sample from the fitted Gaussian, add_gradient_noise perturbs the
    gradients (used when there are very few training trajectories). No
    config objects, no diagonal/low-rank switches -- if you need those, use
    github.com/activatedgeek/optax-swag directly instead of extending this
    module.

How it plugs into training: an optax.GradientTransformation is just a pair
of pure functions (init, update) -- exactly how optax.adam itself is built.
swag_full's update_fn passes the incoming `updates` straight through
UNCHANGED (it never modifies gradients or parameters); it only inspects the
post-update `params` every `freq` steps to update its own running
mean/covariance. This makes it a pure observer you chain alongside your real
optimizer, e.g.:

    opt = optax.chain(optax.adam(1e-3), swag_full(freq=100))
    opt_state = opt.init(p_flat)

    @jax.jit
    def step(p, state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, state = opt.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    ...train as usual for N steps...

    swag_state = opt_state[1]  # index matches position in optax.chain(...)
    mean, cov = swag_state.mean, swag_state.cov

No other change to an existing training loop is needed.

Mean/covariance are updated via Welford's online algorithm (numerically
stabler than the paper's own two-moment-difference form
Sigma = E[theta^2] - E[theta]^2, which can lose precision by subtracting two
large nearly-equal numbers when the weights don't move much between
collected snapshots).

NOTE on BatchNorm / other non-parameter state (see Maddox et al., Section
3.4 and github.com/activatedgeek/optax-swag's README): if a model has
stateful layers whose running statistics are updated by forward passes
rather than by gradients (e.g. eqx.nn.BatchNorm's running_mean/running_var,
tracked via a separate eqx.nn.State pytree, NOT via `params`), those
statistics are tied to the specific weights active during training and can
go stale for a freshly SWAG-sampled theta. None of the models currently used
in this repo (klax.nn.MLP, the FICNN in src/models.py, dynax.NeuralODE) have
such layers, so this does not affect swag_full/sample_swag_full as
implemented. If a future NODE/sPHNN architecture adds BatchNorm or another
eqx.nn.State-based layer, that state would need to be recomputed (e.g. a
forward pass over training data) after each sample_swag_full draw, before
using the sampled parameters for prediction -- this module does not handle
that case.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class SWAGFullState(NamedTuple):
    """State for SWAG's full-covariance Gaussian fit.

    mean: running mean of the collected flat parameter vectors, shape (D,).
    cov: running (biased) covariance of the collected flat parameter
        vectors, shape (D, D).
    step: total optimizer steps seen so far (used to decide every `freq`-th
        step to collect a snapshot).
    n: number of snapshots collected so far (i.e. Welford's sample count).
    """
    mean: jax.Array
    cov: jax.Array
    step: jax.Array
    n: jax.Array


def swag_full(freq: int) -> optax.GradientTransformation:
    """SWAG with the FULL DxD sample covariance (no diagonal/low-rank
    approximation) -- see module docstring.

    Args:
        freq: collect a parameter snapshot into the running mean/covariance
            every `freq` optimizer steps. The paper collects once per epoch,
            after an initial "burn-in" phase where SGD has already reached a
            good solution -- e.g. don't start SWAG updates from step 0 of
            training, chain it in (or otherwise gate it) only once the loss
            has plateaued, or increase `freq` enough to skip the early,
            fast-descending part of training.

    Returns:
        An optax.GradientTransformation. Its `update_fn` returns the
        incoming updates UNCHANGED; it only mutates its own SWAGFullState.
    """
    assert freq > 0, "freq must be a positive integer."
    collect_fn = lambda next_step: (next_step % freq) == 0

    def init_fn(params: jax.Array) -> SWAGFullState:
        d = params.shape[0]
        return SWAGFullState(
            mean=jnp.zeros_like(params),
            cov=jnp.zeros((d, d), dtype=params.dtype),
            step=jnp.zeros([], jnp.int32),
            n=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        updates: jax.Array, state: SWAGFullState, params: jax.Array
    ) -> tuple[jax.Array, SWAGFullState]:
        next_step = state.step + 1
        do_collect = collect_fn(next_step)

        # theta AFTER this optimizer step, i.e. what the update actually
        # produces -- same "collect the post-update params" convention as
        # optax-swag's swa/swag_diag/swag.
        theta = params + updates

        # Welford's online mean/covariance update, applied only on collection
        # steps (n unchanged, mean/cov unchanged otherwise via jnp.where).
        n_new = state.n + 1
        delta = theta - state.mean
        mean_new = state.mean + delta / n_new
        delta2 = theta - mean_new
        cov_new = state.cov + jnp.outer(delta, delta2)

        n = jnp.where(do_collect, n_new, state.n)
        mean = jnp.where(do_collect, mean_new, state.mean)
        cov = jnp.where(do_collect, cov_new, state.cov)

        return updates, SWAGFullState(mean=mean, cov=cov, step=next_step, n=n)

    return optax.GradientTransformation(init_fn, update_fn)


def sample_swag_full(
    key: jax.Array, state: SWAGFullState, scale: float = 1.0, eps: float = 1e-6
) -> jax.Array:
    """Draw one sample theta ~ N(mean, scale^2 * Sigma) from the fitted
    SWAG posterior.

    Args:
        key: PRNG key for one sample. vmap over a batch of keys for multiple
            samples, e.g. jax.vmap(sample_swag_full, in_axes=(0, None))(keys, state).
        state: a SWAGFullState with at least 2 collected snapshots (state.n >= 2)
            -- the covariance estimate is undefined/zero otherwise.
        scale: optional global scale on the covariance (e.g. Maddox et al.
            use an extra 1/2 factor combining diagonal + low-rank terms;
            here there is only the one full-covariance term, so scale=1.0
            reproduces the plain sample covariance -- override only if you
            want to inflate/shrink the posterior for calibration).
        eps: small ridge added to the covariance diagonal before the
            Cholesky factorization, for numerical stability (the running
            covariance can be exactly singular/near-singular early in
            training or if fewer snapshots than parameters were collected).

    Returns:
        theta_sample: flat parameter vector, shape (D,).
    """
    cov = state.cov / jnp.maximum(state.n - 1, 1)
    d = cov.shape[0]
    L = jnp.linalg.cholesky(cov + eps * jnp.eye(d))
    z = jax.random.normal(key, state.mean.shape, dtype=state.mean.dtype)
    return state.mean + scale * (L @ z)


def add_gradient_noise(
    noise_scale: float, key: jax.Array
) -> optax.GradientTransformation:
    """Add isotropic Gaussian noise to the gradient at each step, before the
    base optimizer transform -- a simplified SGLD-style perturbation.

    Motivation: with very few training trajectories (e.g. nD=2), minibatch
    SGD/Adam alternates between as many FIXED update directions as there are
    trajectories (one gradient per trajectory). If those per-trajectory
    gradients are close to antiparallel at the MAP point (as they tend to be
    once training has actually converged there -- verified diagnostically:
    cos(grad_traj0, grad_traj1) ~ -0.999 on this repo's chicken-dataset
    nD=2 checkpoints once genuinely converged), the resulting "random walk"
    SWAG relies on collapses to a walk back and forth along ONE axis, not
    real multi-directional exploration -- observed as SWAG's covariance
    concentrating >=95% of its variance in a single eigendirection
    regardless of learning rate. Injecting noise on the gradient perturbs
    the update away from that fixed axis at every step, giving SWAG a
    chance to explore other directions too.

    Verified empirically on such a case:
    noise_scale comparable to the actual per-trajectory gradient norm (not
    small relative to it) is needed for a visible effect -- a noise_scale
    orders of magnitude below the gradient norm changes nothing, since it is
    just swamped by the real gradient. Pick noise_scale relative to the
    gradient magnitudes you actually see in your problem (e.g. from a
    trial run's grad_norm diagnostics), not as a small constant by default.

    Args:
        noise_scale: standard deviation of the per-parameter Gaussian noise
            added to the gradient at every step (same std for every one of
            the D parameters, i.e. isotropic in the flat parameter space).
        key: PRNG key. A single key is consumed once at init_fn time and
            advanced (via jax.random.split) internally on every update_fn
            call, so the SAME key given here produces a fresh, independent
            noise draw at each step -- do not need to pass a new key
            per-step yourself.

    Returns:
        An optax.GradientTransformation. Its `update_fn` returns
        `updates + noise_scale * N(0, I)`, so chain it BEFORE the base
        optimizer (e.g. optax.chain(add_gradient_noise(...), optax.sgd(lr),
        swag_full(...))) so the noise is subject to the same
        clip_grad_norm/optimizer transform as the real gradient, rather
        than being added downstream of them.
    """
    def init_fn(params: jax.Array) -> jax.Array:
        del params
        return key

    def update_fn(
        updates: jax.Array, state: jax.Array, params: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        del params
        noise_key, next_state = jax.random.split(state)
        noisy_updates = updates + noise_scale * jax.random.normal(
            noise_key, updates.shape, dtype=updates.dtype
        )
        return noisy_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)
