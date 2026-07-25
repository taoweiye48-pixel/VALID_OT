# P1 Amendment 002: incomplete manual labels in the expansion cohort

Status: frozen after metadata audit and before inspection of any P1 v2 outcome  
Date: 2026-07-18  
Scope: sampling and interpretation of label-based witnesses in the spatialDLPFC expansion cohort

## Metadata finding

The official 30-section object contains 113,927 spots, but `manual_layer_label` is non-missing for only 11,991 spots and therefore does not cover the ten-donor expansion. Treating the remaining sections as manually annotated would be false. This finding was made while validating metadata, before any P1 v2 solver output or external-utility result existed.

## Frozen corrective rule

1. The ten-donor primary cohort remains eligible for internal finite-intervention fidelity and held-out-expression utility analyses.
2. Spot sampling is changed from label-stratified to deterministic label-agnostic sampling (up to 1,500 spots per section; fixed seeds 20260718 and 20260719 for anterior and middle positions).
3. `BayesSpace_harmony_07`, available for every spot, is retained only as a seven-domain auxiliary pseudo-label so that label-conditioned diagnostics can execute.
4. Any result using `label_error_shared_closed_set` or `source_only_open_set` in the expanded cohort must be labelled “auxiliary pseudo-label witness”; it is not an independent histological ground truth and cannot support a claim of real alignment-error identification.
5. The held-out-expression witness is the only external witness in the ten-donor cohort that may contribute to the primary expanded-cohort interpretation.
6. Legacy cohorts retain their existing label semantics and remain a separate replication stratum.

## Consequences for interpretation

- The sample expansion strengthens precision and heterogeneity assessment for internal fidelity, scale invariance, solver behaviour, and held-out expression utility.
- It does not create ten independently annotated ground-truth alignments.
- The label-based external-utility claim remains limited; future manually annotated donors would be required to resolve it decisively.
- No epsilon/tau grid, solver setting, score, threshold, arm, or outcome-based selection rule is changed.

