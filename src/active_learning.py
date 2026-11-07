"""Acquisition criteria for active learning on dynamical systems.

A criterion maps the predictive marginal variances of a candidate trajectory to one score; the candidate with the largest score is acquired.
sum_marginal (MGE): sum of log marginal variances; sum_variance (VAR): sum of marginal variances.
"""
import jax.numpy as jnp


# ------------- Acquisitions -------------

VARIANCE_FLOOR = 1e-12

# vanilla ALM (Active Learning MacKay) - uncertainty max
def score_sum_marginal(var: jnp.ndarray,
                       sigma_noise: float = 0.0,
                       variance_floor: float = VARIANCE_FLOOR,
                       ) -> float:
    """Sum log marginal predictive variances with a numerical floor.

    `variance_floor` is in the same squared output units as `var`. It only
    prevents degenerate variances from producing -inf/nan acquisition scores.
    """
    safe_var = jnp.maximum(var + sigma_noise**2, variance_floor)
    return jnp.sum(jnp.log(safe_var))


# Lundby-style sum-of-marginal-variances (no log) - see Lundby et al. 2023,
# "Deep active learning for nonlinear system identification", eq. (13)/(25):
# their acquisition sums raw empirical ensemble variances (optionally scaled
# by 1/std per state channel), never a log-variance/entropy surrogate. This
# is NOT equivalent to score_sum_marginal after aggregation: log is strictly
# concave, so log(sum(var)) != sum(log(var)), and ranking by sum(var) can
# disagree with ranking by sum(log(var)) whenever the variance vectors being
# compared are not simple positive scalar multiples of one another (e.g. one
# candidate has one dominant high-variance coordinate while another has many
# moderately-uncertain coordinates -- the log-sum score compresses the first
# candidate's outlier coordinate relative to the sum-of-variances score).
def score_sum_variance(var: jnp.ndarray,
                       sigma_noise: float = 0.0,
                       variance_floor: float = VARIANCE_FLOOR,
                       ) -> float:
    """Sum of raw (non-log) marginal predictive variances.

    Unlike `score_sum_marginal`, there is no logarithm to blow up near zero,
    so `variance_floor` here only clips genuinely tiny negative numerical
    noise (e.g. GGN round-off on a coordinate whose true variance is exactly
    zero) up to zero, within `[-variance_floor, 0)`. A coordinate more
    negative than `-variance_floor` is not silently clipped -- it indicates
    a materially broken variance estimate (a bug upstream, not floating-
    point noise) and is deliberately propagated as NaN so it contaminates
    this candidate's score and is visible to the caller (`jnp.isfinite`),
    rather than being hidden by clipping.
    """
    raw_var = var + sigma_noise**2
    tiny_negative = (raw_var < 0.0) & (raw_var >= -variance_floor)
    safe_var = jnp.where(tiny_negative, 0.0, raw_var)
    return jnp.sum(safe_var)
