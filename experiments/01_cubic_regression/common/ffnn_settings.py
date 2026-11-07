"""Settings of the cubic-regression NN illustration that are reported in the thesis tables (single definition, used by ffnn_uq_comparison.py and by the table export of
verify_analytic_cubic_regression_mll.py: table1_dataset_configuration.tex, tableA2_ffnn_setup.tex). Values are those of the runs behind the exported figures (unchanged)."""
H, K = 16, 10                                           # hidden width, ensemble size (initialisation seeds 0..K-1)
INIT = dict(w1=2.0, b1=1.5, w2=0.5)                      # Std of the Gaussian input weights, hidden biases, output weights (output bias zero)
L2 = 1e-3                                               # MAP objective: mean r^2 + (L2/2)||theta||^2
TRAIN_STEPS, TRAIN_LR = 30000, 1e-2                      # full-batch Adam, cosine decay of the learning rate from TRAIN_LR to 0 over TRAIN_STEPS (optax.cosine_decay_schedule)
CAL_STEPS, CAL_LR = 200, 1e-1                            # per-member Laplace marginal-likelihood calibration (src/laplace.py::calibrate_mll; initial alpha = sigma_n = 1 by its defaults)
SWAG_BATCH, SWAG_STEPS, SWAG_FREQ, SWAG_LR = 1, 2000, 10, 0.5
SWAG_CLIP, SWAG_RIDGE, N_SAMPLES = 1.0, 1e-14, 64        # gradient-norm clipping, covariance ridge, parameter draws per SWAG component
SAMPLE_KEY_OFFSET = 30000                                # sampling key of member k: 30000 + k (continuation key: k)
