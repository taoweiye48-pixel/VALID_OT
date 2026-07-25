# Protocol amendment 007: v1.1 full restart at epsilon 0.25

**Authorization:** explicit user instruction, 2026-07-16.  
**New protocol ID:** `valid-ot-v1.1-epsilon025-2026-07-16`  
**Scope:** complete confirmatory restart; no v1.0 scientific outcome table was inspected for parameter selection.

## Change

The shared entropic regularization parameter for row-softmax, balanced OT, and UOT is changed from `epsilon=0.10` to `epsilon=0.25`. The `1e-9` convergence Gate, evidence interventions, cost normalization, masses, exact-response definition, methods, selected data, witnesses, seeds, and scientific thresholds remain unchanged.

## Numerical justification

- At `epsilon=0.10`, the Stereo-seq balanced problem failed the frozen convergence Gate repeatedly after three solver-repair rounds.
- Continuing a failed solve from 4,000 to 8,000 iterations changed the plan by relative L1 `1.32e-6`; it could not be relabelled as exact.
- A preregistered probe at `epsilon=0.25` converged under the original criterion in 2,605 iterations, whereas `0.10`, `0.15`, and `0.20` did not converge within 4,000 alternating iterations.
- The change is motivated by solver validity rather than audit performance.

## Restart and separation rules

1. All 729 completed v1.0 E4-E5 checkpoints are archived with their original status files and are excluded from v1.1 aggregation and inference.
2. E1 and the runtime probe are rerun under v1.1.
3. E4-E9 are run from empty v1.1 checkpoint directories.
4. `epsilon=0.10` is reported only as a numerical feasibility failure/stress setting; it cannot be mixed with v1.1 scientific results.
5. The paper must state that conclusions apply to the frozen moderate-entropy setting and must not claim invariance to all regularization strengths.

This amendment changes a model parameter and therefore creates a new confirmatory protocol version rather than patching the old result table.
