# Figure 2 contract — local-reference validation

## Core conclusion

Multi-step convergence and three independent derivative checks establish that the
finite-step score at $h=0.01$ is a numerically reliable local response reference
for the tested methods, intervention arms and cost channels. This figure supports
local numerical fidelity only; it does not establish endpoint transportability,
external utility or biological correspondence.

## Figure archetype

Quantitative multi-panel validation figure: convergence profiles, independent
derivative cross-validation and unit-level local-fidelity summaries.

## Target and export

- Target: Bioinformatics main text
- Width: 177.8 mm (double column)
- Backend: Python / matplotlib
- Exports: editable SVG and PDF; 300-dpi PNG; 600-dpi LZW TIFF
- Typeface: Arial/Helvetica/sans-serif fallback

## Panel map

### a — Multi-step convergence

- Data: 3,000 adjacent-step comparisons from WP1
- Structure: 3 methods × 2 arms × 2 channels × 5 adjacent-step pairs × 50
  direction-level numerical conditions
- Marks: median lines and interquartile ribbons
- Horizontal reference: the frozen median-error threshold of 0.05, evaluated
  at the smallest adjacent step only
- Role: demonstrate convergence towards a stable local reference as the step
  size decreases

### b — Independent derivative checks

- Row-softmax: 200 analytic-derivative conditions
- Balanced OT: 104 converged implicit-differentiation conditions from 9 fixed
  independent units
- UOT: 48 conditions from four fixed units, three $(\epsilon,\tau)$ settings
  and four Arm--channel combinations. An independent convex-dual optimizer and
  UOT first-order implicit derivative were implemented without calling the
  production log-Sinkhorn solver or response-metric helpers.
- Marks: all numerical conditions and group medians
- Row-softmax frozen gates: median relative L1 ≤ 0.01 and q90 ≤ 0.05
- Balanced-OT frozen gates: global relative L1 ≤ 0.01, row median relative L1
  ≤ 0.01, row q90 relative L1 ≤ 0.05 and median row cosine ≥ 0.999
- All three checks use numerical conditions for validation rather than
  biological replication.

### c — Twelve local-fidelity families at $h=0.01$

- Observations: 252 unit-family rows = 21 independent units × 12 families
- Families: 3 methods × 2 arms × 2 cost channels
- Plotted low-is-better metrics: row-plan relative L1, scalar rMAE and
  neighbourhood error
- Marks: all independent units plus family-median diamonds
- Additional high-is-better metrics reported in text: Spearman $\rho$,
  top-decile overlap and direction cosine
- Result: 252/252 unit-family gates and 12/12 family gates pass

## Frozen WP3 thresholds

- Spearman $\rho$ ≥ 0.95
- Top-decile overlap ≥ 0.90
- Scalar rMAE ≤ 0.10
- Row-plan relative L1 ≤ 0.10
- Direction cosine ≥ 0.99
- Neighbourhood error ≤ 0.15

## Source data

- `data/wp1_full_step_convergence_unit.tsv`
- `data/wp1_full_local_reference_summary.tsv`
- `data/wp2_derivative_cross_validation.tsv`
- `data/uot_independent_derivative_conditions.tsv`
- `data/wp3_local_fidelity_unit.tsv`
- `data/wp3_local_fidelity_family.tsv`

## Reviewer risks and controls

- Numerical conditions in panels a and b are validation conditions, not
  biological replicates and not units for biological inference.
- Panel c shows all 21 independent-unit observations in every family.
- The 0.05 line in panel a is not a gate for every displayed step; it is the
  preregistered criterion for the leftmost, smallest adjacent-step comparison.
- The UOT cross-check uses deterministic 96-by-96 subsamples and is reported as
  numerical validation rather than additional biological evidence.
- The figure validates a local reference at $h=0.01$ and does not by itself
  justify extrapolation from the local score to the deletion endpoint.
