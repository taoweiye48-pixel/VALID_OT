# Supplementary figure legends

## Supplementary Figure S1 | Multi-step numerical convergence

Adjacent-step relative L1 differences for five nested step pairs under the
four Arm-by-channel families. Pair-directions were first aggregated within 21
biological independent units. Lines show medians and ribbons show
interquartile ranges. Solid circles denote Arm R; dashed triangles denote Arm
N. Opaque and lighter lines denote expression and spatial interventions,
respectively. All 1,260 unit-family-step records were estimable.

## Supplementary Figure S2 | Balanced-OT implicit differentiation validation

Plan-level relative L1 error, row-level relative L1 error and absolute
direction-cosine error between implicit and finite-difference derivatives.
Direction records were aggregated within nine independent units before
plotting. Points are independent units and black bars are medians. Logarithmic
axes are used for numerical errors.

## Supplementary Figure S3 | Complete endpoint transportability grid

Median independent-unit Spearman correlation and relative mean absolute error
between the finite response at `h=0.01` and the exact endpoint response for all
methods, arms and deleted channels. Biological pairs were aggregated within 21
independent units before the displayed medians were calculated.

## Supplementary Figure S4 | UOT mass-shape decomposition

Median independent-unit rank and magnitude transportability of transported
mass and conditional row shape from the local to endpoint response in 21
units. The two quantities diagnose distinct components of the UOT row-plan
response and are not pooled into a biological-causality claim.

## Supplementary Figure S5 | Coordinate-frame robustness

Spatial-cost correlation with the baseline construction, local-fidelity
Spearman correlation of `s(0.01)` and held-out-expression NEX-AURC across the
baseline, label-free rigid and HER2 oracle coordinate variants. Lines and
ribbons show medians and interquartile ranges. Baseline and label-free
summaries use the 21 main units; the HER2 oracle variant is available only for
the eight controlled HER2ST patients. Method points are separated only within
the categorical x positions.

## Supplementary Figure S6 | Seven-gene-split robustness

Held-out-expression NEX-AURC of `s(0.01)` across the historical split, five
frozen random splits and a source-only split. Lines and ribbons show medians
and interquartile ranges across 21 independent units. The red dashed line at 1
denotes random selective ranking; lower values are better.

## Supplementary Figure S7 | Complete practical-score audit

Median internal-fidelity Spearman correlation with the high-accuracy local
reference and median held-out-expression NEX-AURC for all eight registered
practical scores across the 21 main independent units. Labels denote assigned
raw cost (cost), barycentric displacement (bary.), conditional entropy
(entropy), finite response at `h=0.01` [`s(0.01)`], one minus maximum row
probability (low-p), probability-margin risk (margin), source-boundary
proximity (boundary) and transported-mass deficit (mass). Lower NEX-AURC is
better; the two heatmaps are descriptive and are not pooled into one score.

## Supplementary Figure S8 | WP11 two-dimensional response surfaces

Median endpoint response over the frozen 3-by-5 grid of deleted-channel
coefficient `u` and retained-channel coefficient `v`, shown for fixed and
co-regularized regimes and all three methods. Every cell contains the same 21
independent units. The surfaces decompose frozen model-objective responses and
do not identify biological causal mechanisms.

## Supplementary Figure S9 | Representative tissue and manual-layer context

Source and target H&E images, manual cortical layers, exact-response priority,
source-boundary-proximity fixed-QC priority, retained manual-layer mismatch
and the corresponding selective-ranking curve for one adjacent spatialLIBD
section pair. Risk percentiles are display-only. This descriptive example does
not alter the three-donor analysis or provide exact spot-to-spot truth.

## Supplementary Figure S10 | Historical parameter sensitivity and Arm-S control

Median expression-channel NMAE across the historical epsilon grid, UOT tau
grid and the 372 Arm-S objective-scaling implementation comparisons. Circles
and solid lines denote Arm R; triangles and dashed lines denote Arm N. The red
dashed line marks the frozen NMAE gate of 0.75. Arm S produced maximum plan
difference, normalized plan L1 and response NMAE of 0, and minimum response
Spearman correlation of 1 in all 372 comparisons.
