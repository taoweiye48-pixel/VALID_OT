# P1 Amendment 003: independent pathologist-labelled truth cohort

Status: frozen after source-file audit and before any P1 v2 solver outcome  
Date: 2026-07-18  
Scope: add a manually annotated validation cohort without misrepresenting semi-synthetic targets as real cross-sections

## Motivation and source audit

Public DLPFC sections with expert cortical-layer labels do not provide 8--12 independent donors: the standard 12-slice resource contains only three donors, while the 30-section spatialDLPFC cohort has manual labels for only 11,991 of 113,927 spots. To obtain at least eight independent expert-labelled biological units, P1 v2 adds HER2ST.

The official HER2ST repository contains one pathologist-labelled spatial-transcriptomics section for each of eight patients: `A1`, `B1`, `C1`, `D1`, `E1`, `F1`, `G2`, and `H1`. The labels are breast-tumour pathology regions (for example invasive cancer, connective tissue, immune infiltrate), not cortical layers. Count matrices, grid coordinates, and the pathologist labels were verified to join by rounded array coordinates. Spots labelled `undetermined` are excluded.

## Frozen controlled-pair construction

Each patient is one independent unit and contributes exactly one controlled pair:

1. Use every count-matrix spot with a non-missing, non-`undetermined` pathologist label.
2. Select 500 cost genes and a disjoint set of 100 held-out genes by variance within that patient's real section.
3. Generate the target from the same real section with the pre-existing `combined` generator: deterministic rigid transform, fixed-amplitude non-rigid warp, expression batch noise/dropout, and 20% spatial crop.
4. Use seeds `20260718` through `20260725` in the fixed section order above.
5. Preserve the exact source-to-target correspondence and missingness table produced by the crop.
6. Evaluate both directions, but count the patient only once.

This yields `n=8` independent patient units with genuine pathologist source labels and controlled target truth. It is a real-data-based semi-synthetic validation cohort, not a real cross-section alignment cohort.

## Reporting rules

- Report this cohort separately from the ten-donor real cross-section spatialDLPFC cohort and the legacy replication cohort.
- Label-based error and correspondence recovery can be treated as ground-truth validation only within this controlled cohort.
- Do not claim that the generated target is a second measured tumour section.
- Do not pool the three cohort types into a single biological-effect estimate; pooled tables are bookkeeping/sensitivity summaries only.
- No P1 arm, parameter grid, solver setting, fidelity threshold, score, or registered external Gate result is changed.

