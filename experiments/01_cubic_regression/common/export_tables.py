"""Compact settings tables of the cubic-regression experiment, tabular fragments without floats, captions or labels.
table1_dataset_configuration.tex   data, models and NN training in one table (blocks 1 and 2)      -> verify_analytic_cubic_regression_mll.py::write_tables
tableA2_ffnn_setup.tex             inference / numerics (block 3)                                   -> ffnn_uq_comparison.py::export
Values come from VerificationConfig (data) and common/ffnn_settings.py (NN); the fragments are identical to the revised thesis tables (tables/appendix/exp01/). Each table is written by exactly one exporter,
so the two exporters can run in either order."""
import ffnn_settings as F


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")


def _pow10(x: float) -> str:
    import math
    return f"10^{{{int(round(math.log10(x)))}}}"


def dataset_model_training_table(cfg, d_theta_nn: int = 49):
    L = ["\\begin{tabularx}{\\linewidth}{@{}p{0.25\\linewidth}X@{}}", "\\toprule", "Setting & Value \\\\", "\\midrule",
         "\\multicolumn{2}{@{}l}{\\emph{Shared data}} \\\\",
         f"Training inputs & $n_D={cfg.n_train}$ equidistant points on $[{cfg.x_train_min:g},{cfg.x_train_max:g}]$ \\\\",
         f"Observation noise & Independent zero-mean Gaussian noise, $\\sigma_{{\\mathrm{{gen}}}}={cfg.sigma_generation:g}$ \\\\",
         f"Evaluation grid & {cfg.n_eval} equidistant points on $[{cfg.x_eval_min:g},{cfg.x_eval_max:g}]$ \\\\",
         f"Data seed & {cfg.seed} \\\\", "\\addlinespace",
         "\\multicolumn{2}{@{}l}{\\emph{Models}} \\\\",
         "Cubic model & Polynomial features from \\cref{eq:cubic-model}, $d_\\theta=4$ \\\\",
         f"Neural network & \\ac{{MLP}} $1\\to{F.H}\\to1$, tanh hidden units, linear output, $d_\\theta={d_theta_nn}$ \\\\",
         "NN initialisation & Independent zero-mean Gaussian input weights, hidden biases,",
         f"  and output weights with standard deviations ${F.INIT['w1']:.1f}$, ${F.INIT['b1']:.1f}$, and ${F.INIT['w2']:.1f}$, respectively.",
         "  Output bias zero. \\\\", "\\addlinespace",
         "\\multicolumn{2}{@{}l}{\\emph{NN training}} \\\\",
         "Objective & $\\mathrm{MSE}+\\frac{\\lambda}{2}\\|\\boldsymbol\\theta\\|^2$,",
         f"  $\\lambda={_pow10(F.L2)}$ \\\\",
         f"Optimisation & Full-batch Adam, {_thousands(F.TRAIN_STEPS)} steps, cosine learning rate starting at ${_pow10(F.TRAIN_LR)}$ \\\\",
         f"Ensemble & $n_{{\\mathrm{{ens}}}}={F.K}$, initialisation seeds $0,\\ldots,{F.K - 1}$ \\\\",
         "\\bottomrule", "\\end{tabularx}"]
    return L


def ffnn_inference_table(cfg):
    L = ["\\begin{tabularx}{\\linewidth}{@{}p{0.25\\linewidth}X@{}}", "\\toprule", "Setting & Value \\\\", "\\midrule",
         f"BLR reference & $\\alpha={cfg.fixed_prior_precision:g}$, $\\sigma_{{\\mathrm n}}={cfg.fixed_sigma:g}$,",
         "  analytical expansion point $\\boldsymbol\\theta^*=\\boldsymbol\\mu_{\\mathrm{BLR}}$ \\\\",
         "Members & \\ac{LA} and \\ac{SWAG}: member 0.",
         "  \\Ac{DE}, \\ac{MoLA}, and \\ac{MultiSWAG}: all retained members with equal weights. \\\\",
         "NN Laplace & Full-network \\ac{GGN}, isotropic Gaussian prior,", "  linearised prediction \\\\",
         "NN calibration & Per-member \\ac{MLL} optimisation of", "  $(\\log\\alpha,\\log\\sigma_{\\mathrm n})$ with trained weights fixed.",
         f"  Adam, {F.CAL_STEPS} steps, learning rate ${F.CAL_LR:g}$, initial $\\alpha=\\sigma_{{\\mathrm n}}=1$. \\\\",
         f"\\ac{{SWAG}} continuation & \\ac{{SGD}}, {F.SWAG_STEPS} steps, batch size {F.SWAG_BATCH},",
         f"  constant learning rate ${F.SWAG_LR:g}$, gradient-norm clipping at {F.SWAG_CLIP:g} \\\\",
         f"Snapshots & Every {F.SWAG_FREQ} continuation steps, $n_{{\\mathrm{{snap}}}}={F.SWAG_STEPS // F.SWAG_FREQ}$ \\\\",
         "\\ac{SWAG} covariance & Full empirical covariance, normalised by", "  $n_{\\mathrm{snap}}-1$, with ridge",
         "  $\\delta_{\\mathrm{ridge}}\\boldsymbol I_{d_\\theta}$,", f"  $\\delta_{{\\mathrm{{ridge}}}}={_pow10(F.SWAG_RIDGE)}$ \\\\",
         f"Parameter samples & $n_{{\\mathrm{{MC}}}}={F.N_SAMPLES}$ per \\ac{{SWAG}} component \\\\",
         f"Continuation / sampling seeds & $k$ / ${F.SAMPLE_KEY_OFFSET}+k$ for member $k$ \\\\",
         "Predictive moments & Population normalisation, epistemic uncertainty only \\\\",
         "Arithmetic & \\texttt{float64} \\\\", "\\bottomrule", "\\end{tabularx}"]
    return L
