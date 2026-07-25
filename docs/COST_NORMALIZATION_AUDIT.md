# VALID-OT cost-normalization audit (P0, read-only)

## Scope and authority

This audit documents the copied v1.3 implementation under `code/` without changing it or executing a scientific analysis. The confirmatory path is `validot.external_data.prepare_pair_from_h5ad` → `validot.solvers.cost_components` → `validot.benchmark.run_audit`/`_solve_common`.

The public copy normalizes machine-specific roots to repository-relative paths. The original frozen source snapshot remains unchanged in the local project archive; scientific parameters, pair manifests, hashes and result files are retained here.

## 1. Relevant files and functions

- `code/validot/external_data.py`
  - `_normalize`: conditional library-size normalization to 10,000 counts and `log1p` for count-like inputs.
  - `_row_normalize`: per-row L2 normalization of expression vectors.
  - `_standardize_xy`: per-slice centering and division by RMS radial distance.
  - `prepare_pair_from_h5ad`: HVG/held-out selection and construction of expression/coordinate inputs.
- `code/validot/solvers.py`
  - `cost_components`: constructs and independently rescales expression, spatial-cross, source-structure, and target-structure costs.
  - `log_sinkhorn` and `row_softmax`: consume the final mixed cost through `cost / epsilon`.
- `code/validot/benchmark.py`
  - `_solve_common`: mixes cost components and selects balanced OT, UOT, row-softmax, or legacy FGW solvers.
  - `run_audit`: fixes the baseline weights and implements the registered deletion and endpoint interventions.
- `code/protocol/frozen_config.json`
  - supplies `epsilon`, `uot_tau`, iteration limits, tolerance, and the frozen protocol identifier.

## 2. Expression cost, CE

For real data, count-like matrices are normalized per observation to a library size of 10,000 and transformed by `log1p`. The selected HVG vectors are then normalized row-wise:

\[
\tilde{x}_i = x_i / \max(\lVert x_i\rVert_2,10^{-12}).
\]

The unscaled pairwise expression cost is squared Euclidean distance:

\[
C^{raw}_{E,ij}=\lVert\tilde{x}^{(s)}_i-\tilde{x}^{(t)}_j\rVert_2^2.
\]

`cost_components` divides the complete source-by-target matrix by the median of its strictly positive entries:

\[
C_E=C_E^{raw}/\max\{\operatorname{median}(C_E^{raw}[C_E^{raw}>0]),10^{-12}\}.
\]

Thus CE normalization is pair/direction/matrix-specific, not a single global scale shared across the dataset.

## 3. Spatial cost, CS

Each slice's coordinates are centered independently and divided by its RMS radius:

\[
\tilde{z}_i=(z_i-\bar z)/\max\left\{\sqrt{n^{-1}\sum_k\lVert z_k-\bar z\rVert_2^2},10^{-12}\right\}.
\]

The cross-slice spatial cost is squared Euclidean distance,

\[
C^{raw}_{S,ij}=\lVert\tilde{z}^{(s)}_i-\tilde{z}^{(t)}_j\rVert_2^2,
\]

and is independently divided by the median of its positive entries using the same `scale` helper. Therefore CS is also pair/direction/matrix-specific. Source- and target-internal structure matrices used only by legacy FGW paths are ordinary Euclidean distances and are each median-positive scaled separately.

## 4. Baseline mixing and solver parameters

For balanced OT, UOT, and row-softmax, `_solve_common` constructs

\[
C=w_E C_E+w_S C_S.
\]

The frozen baseline is `wE = 0.5`, `wS = 0.5`, so

\[
C_0=0.5C_E+0.5C_S.
\]

The frozen configuration supplies:

- entropic regularization `epsilon = 0.25`;
- UOT marginal relaxation `tau_a = tau_b = 2.0`;
- maximum Sinkhorn iterations `4000`;
- Sinkhorn tolerance `1e-8`.

Balanced OT, UOT, and row-softmax receive the same normalized mixed cost and the same epsilon. UOT additionally uses tau. No second matrix normalization is applied after the weighted sum in this confirmatory path.

## 5. Current interventions

`run_audit` implements the registered finite deletion endpoints as:

- `I_EXPR`: `wE = 0.0`, `wS = 0.5`, hence `C = 0.5 CS`;
- `I_SPATIAL`: `wE = 0.5`, `wS = 0.0`, hence `C = 0.5 CE`.

The one-sided finite-difference endpoint with step `h` uses:

- expression: `wE = 0.5(1-h)`, `wS = 0.5`;
- spatial: `wE = 0.5`, `wS = 0.5(1-h)`.

Epsilon and tau remain unchanged for the baseline, deletion endpoints, and finite-difference endpoints. Consequently the deletion endpoints change both the evidence composition and the total cost scale relative to fixed epsilon. P0 records this property but does not alter it.

## 6. Local-score scaling and possible second weighting

For balanced OT/UOT/row-softmax, local proxy construction stores `model_expression_cost = 0.5 CE`, `model_spatial_cost = 0.5 CS`, and their sum. This is the baseline weighting used to define local scores, not a second scaling of the solver's already mixed cost.

Legacy PASTE/PASTE2 branches use a different FGW-specific mapping: spatial structures are multiplied by `sqrt(spatial_weight)` so the square-loss structural objective receives a linear spatial weight, while expression cost receives `expression_weight` and additional `fgw_alpha` factors enter the solver/proxy definitions. The manuscript explicitly excludes those historical FGW outputs from confirmatory Gates because objective consistency was not established. P0 made no change to these paths.

## 7. Agreement with the manuscript

The Methods statement `C = wE CE + wS CS`, baseline `0.5/0.5`, `I_EXPR = (0,0.5)`, `I_SPATIAL = (0.5,0)`, `epsilon = 0.25`, and `tau = 2.0` agrees with the inspected confirmatory code.

The Methods does not currently spell out the positive-median cost scaling, the preceding expression row-L2 normalization, or per-slice coordinate RMS normalization. Those omissions are scientific-method reporting issues for a later authorized manuscript revision; P0 only records them here.

## 8. P1 questions deliberately left unresolved

P0 did not run or implement any of the following:

1. weight-renormalized deletions (`CE` or `CS` with remaining weight 1);
2. effective-scale-compensated deletions (for example changing epsilon alongside a retained 0.5 block);
3. a full weight path between baseline and deletion;
4. comparisons of Gate outcomes under these alternative interventions.

No `--audit-cost-scales` mode was added because the copied workspace is not wired to a self-contained processed-data root and adding a pipeline option without executing a representative data load could not be validated safely. The formulas and exact implementation locations above provide the read-only P0 audit trail without risking pipeline behavior.
