# Design of the library (`src/`)

The uncertainty-quantification (UQ) methods are implemented as backends that know nothing about the model. For every model class a thin wrapper runs the workflow of a method on that model, so that every script uses a method in the same way for the neural ODE (NODE) and the stable port-Hamiltonian network (sPHNN). The wrappers are in `src/models.py`.

| layer | files | what it does |
|---|---|---|
| backends | `laplace.py`, `swag.py`, `priors.py` | a UQ method on a **flat parameter vector** and a function `parameters -> prediction`: GGN and its eigendecomposition, calibration of the prior precision and the noise by the marginal likelihood, linearised pushforward (Laplace); snapshots and sampling of the full covariance (SWAG) |
| wrappers | `models.py` | `Laplace_ODESolver` and `SWAG_ODESolver` carry out the workflow for an ODE solver (flatten the trainable parameters, build the trajectory function, call the backend, return mean, variance or samples); `Laplace_NeuralODESolver`, `Laplace_sPHNNSolver` and the two SWAG subclasses only supply the model-specific part |
| combination | `mola.py`, `ensembles.py` | MoLA is a mixture of K Laplace wrappers (mixture moments); the deep ensemble needs no backend, `ensembles.py` holds its helpers (training-RMSE filter of the members) |
| acquisition | `active_learning.py` | scores (MGE, VAR) of a candidate trajectory from its predictive marginal variances, independent of the UQ method |

**The model-specific part** of a wrapper is three methods: `check_model_wrapping`, `partition_model` and `rebuild`. For a NODE the trainable parameters are all arrays of the network. For an sPHNN they are the raw coordinates before the softplus reparametrisation of the constrained weights; fixed quantities such as the equilibrium x\* are not part of the posterior, and the model has to be unwrapped before it is called. This handling is in one place. A method then runs the same for both model classes, e.g. `fit_laplace_mll()` followed by `mean_linearized` and `std_linearized`.

**Use in the experiments.** The Laplace wrappers are used by the harmonic-oscillator, thermal and DED code, the SWAG wrappers by the harmonic-oscillator code. The experiments form MoLA and DE themselves from these pieces: MoLA as the mean of the Laplace variances of the members plus the variance of their linearised means, DE as the variance of the member rollouts. `MoLA_ODESolver` is the packaged form of the same formula (it is tested and used in `examples/`).

**Not provided:** a single abstract UQ interface. Each method has its own entry points; what they share is the flat-parameter backend and the model-class wrapper.

Two small programs show the library in use: [`examples/`](../examples/README.md) (UQ workflow and a short AL loop on the harmonic oscillator). The experiment code does the same with more organisation, see [AL_WORKFLOW.md](AL_WORKFLOW.md).
