# Examples

Two short programs that show the library on the harmonic oscillator of the thesis (undamped, state (q, p); data, models and training are those of `experiments/02_harmonic_oscillator`). They are meant as a readable entry to the code; the experiments do the same with more organisation (see [`docs/DESIGN.md`](../docs/DESIGN.md) and [`docs/AL_WORKFLOW.md`](../docs/AL_WORKFLOW.md)).

```bash
python examples/01_uq_workflow.py          # about 1 minute
python examples/02_active_learning.py      # about 8 minutes
```

Both take `--model sPHNN` (default, the conservative sPHNN of the thesis) or `--model NODE` and `--steps` (training steps per model; the experiments use 5000 and more). The figures are written to `examples/output/`.

| file | content |
|---|---|
| `01_uq_workflow.py` | train K = 3 members; predict with DE, LA (Laplace approximation of one member) and MoLA (one Laplace approximation per member); plot of the mean and the ±2 standard deviation band far beyond the training window |
| `02_active_learning.py` | the loop of Figure 3.2: a pool of 12 candidate initial states, a start trajectory, then train, evaluate, score the pool with DE, LA or MoLA (score MGE or VAR), acquire; with a Random baseline |
| `_oscillator.py` | shared setup: imports of the oscillator code, ensemble training, Laplace fit without the progress output |

The results are those of a toy problem with small training budgets. A single AL path is noisy, so the output of `02_active_learning.py` shows how the loop works and is not a comparison of the methods.
