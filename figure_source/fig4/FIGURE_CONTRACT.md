# Figure 4 contract

## Scientific conclusion

Arm R and Arm N do not differ through one interchangeable mechanism. On the
frozen WP11 objective grid, expression-channel deletion is dominated by
retained spatial-channel compensation, whereas spatial-channel deletion is
dominated by the removal term. Fixed regularization additionally introduces a
strong scale contribution at the pure-spatial endpoint; co-scaling all
applicable regularization terms removes that contribution.

## Evidence logic

- **Panel a — response surface:** median endpoint mean response across 21
  independent units on the fixed-regularization grid. The common colour scale
  permits direct comparison among Balanced OT, UOT and Row-softmax. Markers
  identify the baseline and the Arm R/Arm N endpoints.
- **Panel b — factorial contrasts:** all 21 unit-level values, interquartile
  ranges and medians for removal, compensation, interaction and joint effects.
  The identity is
  `joint = removal + compensation + interaction`.
- **Panel c — scale control:** paired, within-unit differences between the
  fixed-regularization and co-regularized responses at the pure-expression and
  pure-spatial endpoints. A zero difference denotes no relative-regularization
  scale contribution; all differences return to zero at the shared `u=1`
  endpoint.

## Frozen data and estimand

- Independent units: 21.
- Methods: Balanced OT, UOT and Row-softmax.
- Main response: `endpoint_response_mean`.
- Grid coordinates: total cost scale `u` and spatial-cost fraction `v`, with
  expression coefficient `u(1-v)` and spatial coefficient `uv`.
- Heatmaps use fixed regularization only; the factorial panel uses the frozen
  fixed-regularization contrasts; the scale-control panel displays both fixed
  and co-regularized regimes.
- Summary statistics are descriptive medians and interquartile ranges across
  independent units. No spots or directions are pooled as independent
  replicates.

## Claim boundary

This figure decomposes responses of the frozen model objective. It does not
identify a unique biological or causal mechanism, establish biological
correspondence truth, or rank the three methods overall. The sampled `u`–`v`
grid is sparse and should not be described as a continuous analytic surface.
