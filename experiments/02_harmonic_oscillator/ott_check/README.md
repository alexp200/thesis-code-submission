# Ott-style check of the Laplace approximation (Appendix B.2.1)

Qualitative check of the NODE + full Laplace + linearised ODE-solve pipeline on the harmonic oscillator, following the naive NODE experiment of Ott et al. (2023).

* `make_ott_figure.py`: trains the 2-256-2 NODE for both data sets, fits and calibrates the LA, compares the scalar results with the saved summary and draws Fig. B.1 (about one minute).
* `runs/primary_investigations/`: summary of the original run, used by the comparison.
