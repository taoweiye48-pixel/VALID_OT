# P1 implementation correction 001: Gate-summary field mapping

Date: 2026-07-18

Detected after all 750 P1 v2 solver checkpoints were complete and before any
result table, Gate summary, figure or scientific interpretation was produced.

## Symptom

The family-level internal-fidelity summary stored reporting fields as
`median_spearman`, `median_top_decile_overlap` and `median_nmae`, then passed
that reporting record to `add_gate()`, which expects canonical fields
`spearman`, `top_decile_overlap` and `nmae`.  Pandas therefore raised
`KeyError: nmae` during finalization.

## Correction

Construct an explicit one-row canonical metric view from the three median
fields before applying the unchanged Gate thresholds.  No solver, cost,
intervention, score, witness, threshold, aggregation rule or checkpoint is
changed.  A regression test asserts that the canonical mapping produces the
expected Gate result.

## Consequence

The 750 completed checkpoints remain valid and are reused.  Only deterministic
post-processing is rerun.  The pre-correction and post-correction source hashes
are retained through the versioned code manifests and repository history.
