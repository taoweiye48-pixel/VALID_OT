# Protocol Amendment 010: exploratory M5 3d-OT transport head

- Frozen before M5 outcome inspection: 2026-07-16
- Status: post-hoc exploratory add-on; it is not part of the registered M0-M4 confirmation.
- Official repository commit: `39a7cb02748d83299cd471f172f3b972896e61d8`
- Official implementation: `lib_3d_OT.ottools.ot.sinkhorn`
- Environment: isolated `.runtime-3dot`, Python 3.10, PyTorch 2.2.0+cu121.

## Scope

Run the official 3d-OT transport head on the same seven frozen real pairs in both directions
(14 runs). This add-on evaluates the transport head only. It must not be described as a full
retraining or reproduction of the complete deep 3d-OT framework.

## Frozen head settings

- Input features: the 500 frozen cost features already used by M0-M4.
- Coordinates: min-max normalized independently within each slice, matching the official model.
- `epsilon=1`, `gamma=1`, `max_iter=100`, `sim_k=5`, `dist_k=min(200,n_target)`.
- Device: CUDA when the pre-run GPU validation passes.

## Evidence interventions

- Base: the official two-stage support, retaining the top five feature-similar targets among the
  200 spatially nearest candidates.
- `I_EXPR`: replace both feature arrays by constant unit features and set
  `sim_k=dist_k=min(200,n_target)`. This removes feature ranking/cost evidence while retaining the
  spatial candidate set.
- `I_SPATIAL`: set `dist_k=n_target`, retaining the top five feature-similar targets without spatial
  candidate filtering.

These are finite, exact re-solves of the official transport head under specified evidence removal.
Because its geometry enters through a discrete top-k support, no continuous spatial endpoint
gradient is defined. M5 therefore does not fabricate endpoint-gradient results.

## Outcomes

- Primary exploratory score: `exact_combined=max(response_I_EXPR,response_I_SPATIAL)`.
- Real witnesses: public-label mismatch and held-out-expression loss.
- Frozen QC comparators: source boundary proximity, source sparsity and matched-target sparsity.
- Report normalized excess AURC, binary AUROC/AUPRC where defined, 100-repeat negative controls,
  confound-adjusted association and direction-averaged pair gains.
- M5 cannot change the registered M0-M4 Gate decision; it is disclosed separately.

