"""Fixed dataset roles for the final DED active-learning experiment.

The dataset contains only 25 simulations on a rigid 5x5 factorial grid.
No subset is treated as a representative holdout:

* batch 0 is the single warm-start trajectory;
* every other batch is available for active-learning selection;
* every iteration is evaluated on the same complete set of 25 trajectories.

The resulting full-dataset RMSE measures accuracy over this finite benchmark
design space. It intentionally includes trajectories already selected for
training and therefore must not be interpreted as an independent held-out
generalization estimate.
"""

N_TRAJECTORIES = 25
WARMSTART_ID = 0  # v=10, Q=300 -- one of the paper's two training corners

EVALUATION_IDS = list(range(N_TRAJECTORIES))
POOL_IDS = [batch_id for batch_id in EVALUATION_IDS if batch_id != WARMSTART_ID]

assert len(EVALUATION_IDS) == 25
assert len(POOL_IDS) == 24
assert WARMSTART_ID in EVALUATION_IDS
assert WARMSTART_ID not in POOL_IDS
assert set([WARMSTART_ID] + POOL_IDS) == set(EVALUATION_IDS)
