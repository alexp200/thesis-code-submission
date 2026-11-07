"""Acquisition criteria on feature matrices of the rollout covariance (pure NumPy, float64).

Every UQ method is written as Sigma_c = Phi_c Phi_c^T with Phi_c in R^{m x r} (m = 2 probes x 280 samples = 560 outputs of one candidate trajectory c, normalised output units as in the AL runs):
  Laplace (LA, one member)  Phi = J_c U diag(w)^{1/2}      (J_c rollout Jacobian, U / w the posterior eigen-directions / variances)
  Deep ensemble (DE)        Phi = (mu_k - mean_k mu_k)_k / sqrt(K)                          (K member rollouts, population covariance)
  MoLA                      Phi = [Phi_1/sqrt(K), ..., Phi_K/sqrt(K), Phi_DE]               (Sigma = mean_k Sigma_k + Cov_k(mu_k))
Criteria (a larger score is picked), identical to src.active_learning with sigma_noise = 0:
  sum_variance   ||Phi_c||_F^2 = tr Sigma_c                              (VAR)
  sum_marginal   sum_i log max(diag(Sigma_c)_i, 1e-12)                   (MGE)
"""
import numpy as np

VARIANCE_FLOOR = 1e-12


def sum_variance(Phi):
    """VAR score: trace of the rollout covariance."""
    return float(np.sum(Phi ** 2))


def sum_marginal(Phi):
    """MGE score: sum of the log marginal variances of all outputs."""
    return float(np.sum(np.log(np.maximum(np.sum(Phi ** 2, axis=1), VARIANCE_FLOOR))))


def de_features(mu):
    """Features of the deep ensemble; mu: (K, m) member rollouts."""
    return (mu - mu.mean(0)).T / np.sqrt(len(mu))


def mola_features(laplace_features, mu):
    """Features of MoLA; laplace_features: list of K matrices (m x r), mu: (K, m) member rollouts -> m x (K r + K)."""
    K = len(laplace_features)
    return np.concatenate([P / np.sqrt(K) for P in laplace_features] + [(mu - mu.mean(0)).T / np.sqrt(K)], axis=1)
