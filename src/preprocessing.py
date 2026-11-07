"""Proper orthogonal decomposition (POD) of high-dimensional field data: fit a basis on training snapshots, encode to and decode from the latent space."""
import jax.numpy as jnp


##================= POD (proper orthogonal decomposition) =================##
# Linear dimensionality reduction for high-dimensional field data (e.g. FE
# nodal snapshots), following Roth et al. 2025 ("Stable Port-Hamiltonian
# Neural Networks"), Appendix F.1. Each field (state, input, ...) gets its
# own POD basis, fit from TRAINING snapshots only.
#
# Fit once with fit_pod(), then use encode_pod()/decode_pod() to move
# between the full-order field space (N,) and the latent space (n,). For a
# state field with a known physical equilibrium (e.g. ambient temperature),
# shift_equilibrium_pod() re-centers the latent space so that equilibrium
# maps to the origin -- required for sPHNN's x* placement.


def fit_pod(snapshots: jnp.ndarray, n_modes: int) -> jnp.ndarray:
    """Fit a POD basis from a matrix of snapshots via SVD (Eq. F.1-F.2).

    Args:
        snapshots: (n_snapshots, N) matrix of flattened field snapshots,
            e.g. all (trajectory, timestep) pairs from the training
            trajectories stacked along axis 0, N = number of mesh nodes
            (or nodes * components, for a vector-valued field).
        n_modes: number of POD modes to keep (n << N).

    Returns:
        modes: (N, n_modes) POD basis matrix M = c * V_tilde, where c is
            the standard deviation of U_tilde @ Sigma_tilde -- this scaling
            ensures the resulting latent trajectories have unit variance
            (Appendix F.1), consistent with the per-channel normalization
            convention used elsewhere in this codebase (e.g.
            chicken_data.py). Since V_tilde's columns are orthonormal,
            M^T @ M = c^2 * I -- encode_pod() relies on this to invert
            decode_pod() in closed form. Pass modes directly to
            encode_pod()/decode_pod().
    """
    U, S, Vt = jnp.linalg.svd(snapshots, full_matrices=False)
    U_trunc, S_trunc, Vt_trunc = U[:, :n_modes], S[:n_modes], Vt[:n_modes]
    c = jnp.std(U_trunc * S_trunc)
    modes = c * Vt_trunc.T  # (N, n_modes)
    return modes


def encode_pod(field: jnp.ndarray, modes: jnp.ndarray) -> jnp.ndarray:
    """Map full-order field(s) to POD latent coordinates (Eq. F.3), the
    least-squares projection x = argmin_x' ||X - M @ x'||, closed-form
    x = (1/c^2) M^T @ X since M^T @ M = c^2 * I (see fit_pod's docstring).

    Args:
        field: (..., N) full-order field snapshot(s), any number of leading
            batch/time dimensions.
        modes: (N, n_modes) POD basis from fit_pod().

    Returns:
        latent: (..., n_modes) latent coordinates.
    """
    c2 = jnp.sum(modes[:, 0] ** 2)  # = c^2, since column 0 of modes has unit-norm direction scaled by c
    return field @ modes / c2


def decode_pod(latent: jnp.ndarray, modes: jnp.ndarray) -> jnp.ndarray:
    """Map POD latent coordinates back to the full-order field (Eq. F.2),
    X = M @ x.

    Args:
        latent: (..., n_modes) latent coordinates.
        modes: (N, n_modes) POD basis from fit_pod().

    Returns:
        field: (..., N) reconstructed full-order field.
    """
    return latent @ modes.T


def shift_equilibrium_pod(field_eq: jnp.ndarray, modes: jnp.ndarray) -> jnp.ndarray:
    """Encode a known full-order equilibrium field into latent coordinates
    (Eq. F.6's psi_T(T_eq) term), for use as the constant offset that
    re-centers the latent space so the equilibrium maps to the origin:
        x = encode_pod(field) - x_eq
        field ~= decode_pod(x + x_eq, modes)

    Args:
        field_eq: (N,) full-order equilibrium field (e.g. a uniform ambient
            temperature field).
        modes: (N, n_modes) POD basis from fit_pod().

    Returns:
        x_eq: (n_modes,) latent-space offset.
    """
    return encode_pod(field_eq, modes)
