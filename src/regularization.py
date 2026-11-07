"""Regularization helpers aligned with the Laplace parameter partition."""

import equinox as eqx
import jax
import jax.numpy as jnp
import klax


def active_parameter_tree(model, model_class: str):
    """Return exactly the parameter subtree used by the Laplace classes."""
    if model_class == "NODE":
        return eqx.filter(model, eqx.is_array)
    if model_class != "sPHNN":
        raise ValueError(f"Unknown model class: {model_class!r}")

    wrapper_types = (klax.NonTrainable, klax.Constraint)
    is_wrapper = lambda leaf: isinstance(leaf, wrapper_types)
    filter_spec = jax.tree.map(
        lambda leaf: eqx.is_array(leaf) and not is_wrapper(leaf),
        model,
        is_leaf=is_wrapper,
    )
    params, _ = eqx.partition(model, filter_spec, is_leaf=is_wrapper)
    return params


def active_parameter_squared_norm(model, model_class: str) -> jax.Array:
    """Squared L2 norm of the parameters entering the Laplace posterior."""
    leaves = [
        leaf
        for leaf in jax.tree.leaves(active_parameter_tree(model, model_class))
        if eqx.is_array(leaf)
    ]
    return sum((jnp.sum(leaf**2) for leaf in leaves), start=jnp.asarray(0.0))


def active_parameter_count(model, model_class: str) -> int:
    """Number of scalar parameters entering the Laplace posterior."""
    return sum(
        leaf.size
        for leaf in jax.tree.leaves(active_parameter_tree(model, model_class))
        if eqx.is_array(leaf)
    )
