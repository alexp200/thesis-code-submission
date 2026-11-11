# Actual AL picks versus separately evaluated acquisition scores (first selection, n_D = 1 -> 2)

177 of 180 first selections agree (2 architectures x 15 starts x 6 strategies). Differences:

- NODE/DE/sum_marginal start 873: actual 874, diagnostic 861
- NODE/DE/sum_variance start 873: actual 874, diagnostic 861
- NODE/DE/sum_marginal start 905: actual 795, diagnostic 874

The first-acquisition figures use the actual AL-path picks (`actual_AL_pick`, from the stored paths); the error rankings and the seed-sensitivity analysis use the diagnostic data (separately evaluated acquisition scores, exhaustive errors). The diagnostic models were evaluated locally while the AL paths ran on the cluster; equal keys do not imply bitwise identical fits across machines.

`pick_audit.csv`: all comparisons.
