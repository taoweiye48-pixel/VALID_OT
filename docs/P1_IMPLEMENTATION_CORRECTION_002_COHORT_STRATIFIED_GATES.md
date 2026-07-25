# P1 implementation correction 002: cohort-stratified gates

## Problem

The v2 sample-expansion amendment explicitly prohibits pooling unlike cohort
types as one biological-effect estimate.  The common P1 finalizer nevertheless
produced a backward-compatible gate table after combining all 21 independent
units.  Because the expanded design contains a primary spatialDLPFC cohort, a
controlled HER2ST manual-truth cohort, and a legacy replication cohort, the
pooled technology count and pooled effect are not a valid primary inferential
unit.

## Correction

No solver output, endpoint, witness, parameter grid, gate threshold, or frozen
registered result is changed.  The v2 wrapper now applies the unchanged gate
function separately within each frozen `cohort_role` and writes
`p1_cohort_gate_summary.csv`.  The original pooled `p1_gate_summary.csv` is
retained only for backward-compatible descriptive auditing.

The summary JSON explicitly records that cohort-stratified gates are the
primary basis and that pooled-gate inference is not allowed.

## Scientific consequence

This correction prevents technologies from different evidence roles from
artificially satisfying the minimum-technology gate.  It may reduce the number
of gate passes; that reduction is intentional and reflects the amended
independence and transportability claims.
