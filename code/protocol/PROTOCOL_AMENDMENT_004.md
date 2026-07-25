# Protocol amendment 004: operational details for direction averaging and controls

**Time:** 2026-07-16, before inspection of E4-E6 scientific outcome tables.  
**Reason:** The frozen plan required bidirectional pair averaging and four negative/positive controls, but did not specify the permutation count or the computational subset for the semisynthetic controls.

The following implementation details are frozen:

1. Every E6 biological pair is run in both source-to-target and target-to-source directions. Directions are averaged inside the biological pair before any cross-pair summary; they are never counted as independent pairs.
2. E6 label-shuffle and spatially restricted within-block permutation controls use 100 deterministic repetitions per `pair x direction x method x witness x score` unit.
3. Circular shift uses a deterministic shift of one third of the source units. The leakage positive control uses the observed row loss itself as the risk score and is used only to verify that the evaluation pipeline can recover an oracle ranking.
4. The E5 semisynthetic permutation controls are evaluated on the pre-existing robustness subset: both technologies, one metadata-selected base per technology, the `crop_missing` and `combined` scenarios, seeds `0, 4, 9`, and all five required methods. This choice is made for compute control and does not depend on observed effects.
5. Confound-adjusted E6 associations use rank-standardized audit score plus boundary proximity, local sparsity, log library size, and log region size. Binary label mismatch uses a binomial GLM with HC3 covariance; continuous held-out loss uses OLS with HC3 covariance.

This amendment does not change a scientific endpoint, direction, threshold, method, selected pair, or exclusion rule.
