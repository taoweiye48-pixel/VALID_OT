# Protocol amendment 011: post-review correction and v1.3 analysis freeze

**Time:** 2026-07-16, after completion of v1.2 and an independent code/evidence review.  
**Status:** corrective amendment, not a retroactive preregistration.

Version 1.3 preserves every v1.2 table, threshold, and decision in the
`15_v1_3_correction/00_v1_2_snapshot` directory.  It addresses implementation
and estimand defects identified during review.  It must not silently replace or
reinterpret the registered v1.2 result.

## Frozen corrections

1. Bidirectional real-slice results are averaged within biological pair before
   cross-pair summaries or signed-rank tests, as already required by amendment
   004.
2. Risk-coverage and top-fraction overlap use deterministic fractional tie
   blocks.  Input row order cannot break score ties.
3. Endpoint response is named **one-sided finite-difference sensitivity**.  It
   is a local response linearization requiring additional solver evaluations,
   not a static or cost-free proxy.  Step sizes 0.005, 0.01, 0.02, and 0.05 are
   used for sensitivity analysis; 0.01 remains the original registered value.
4. Label mismatch is split into a shared-label closed-set witness and a
   source-only open-set/unmatched-support witness.  Coverage is always reported.
5. The primary deployable QC comparator is frozen as source boundary proximity.
   Per-task best QC is retained only as a hindsight oracle envelope.
6. Balanced OT and UOT are the confirmatory OT methods.  Row-softmax is a
   non-OT stress-test comparator.  PASTE/PASTE2 results produced with the v1.2
   structural weighting and cross-coordinate local proxies are excluded from
   confirmatory Gates until objective-consistent reruns are available.
7. Robustness direction is defined relative to random ranking as
   `sign(1 - NEX-AURC)` and is accompanied by absolute delta and ranking
   summaries.
8. External utility is matched to the same intervention and witness used for
   internal fidelity.  Combined max scores are reported separately.
9. Cross-stage Stereo-seq comparisons are one independent developmental-series
   unit.  Bootstrap intervals are descriptive and are supplemented by
   leave-one-independent-unit/base-slice-out analyses.

## Post-hoc hypothesis boundary

The observed UOT expression-intervention signal was discovered after inspecting
the registered combined endpoint.  It is explicitly exploratory in v1.3 and
requires a new, independent, prospectively frozen replication before it may be
called confirmatory.

## FGW implementation correction

Future square-loss FGW runs scale both within-slice structure matrices by the
square root of the requested spatial evidence weight, so the structural
objective changes linearly.  Local FGW costs use an incumbent-plan GW
objective/gradient contribution.  A finite plan is not described as a globally
exact solve; only numerical stationarity of the frozen non-convex solver is
reported.
