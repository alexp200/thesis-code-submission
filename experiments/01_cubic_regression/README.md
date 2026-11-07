# Cubic regression (Section 4.1.1, Appendix B.1)

Cubic regression is linear in its four coefficients, so Bayesian linear regression (BLR) gives the exact posterior and marginal likelihood. This is used to check the Laplace approximation (LA) and its marginal-likelihood calibration, and as a small test bed for a tanh network (49 parameters) under LA, MoLA, DE, SWAG and MultiSWAG.

| script | content |
|---|---|
| `verify_analytic_cubic_regression_mll.py` | BLR versus LA at fixed hyperparameters, marginal likelihood at several (alpha, sigma) pairs, Adam calibration against the exact optima. Writes `summary.json`, Fig. 4.1, Tables B.1, B.3, B.4 |
| `ffnn_uq_comparison.py` | ten network members on the same data, all UQ methods against BLR (about 3 min on one CPU). Writes Fig. 4.2, Table B.2 |
| `common/` | export paths (`CR_EXPORT_ROOT`, `CR_AUDIT_ROOT` override them), table export, shared plot elements, network settings |

Data: 11 noisy points, `theta_true = (1, 1, 2, 4)`, seed 2026, noise standard deviation 1.5. Outputs go to `thesis_figures/final/` (figures and tables); the numbers behind them are in `summary.json` (verification) and `thesis_figures/final_audit/ffnn_summary.json` (network).
