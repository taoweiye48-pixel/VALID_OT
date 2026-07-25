# Figure 5 contract

## Scientific conclusion

Local-response fidelity and witness-specific diagnostic utility are distinct
properties. The finite response at `h=0.01` reproduces the high-accuracy local
reference, yet its held-out-expression utility remains method-dependent; a
practical score can also be externally useful while being poorly aligned with
the local model response.

## Evidence logic

- **Panel a — same response family:** cohort-stratified held-out-expression
  gain over the frozen fixed-QC comparator. All independent-unit values are
  shown for the finite response, together with its median and interval; the
  local-reference and endpoint-response medians are overlaid. The finite and
  local-reference external results are nearly identical, but their signs and
  magnitudes differ among methods.
- **Panel b — dual-axis audit:** the eight registered primary practical scores
  are positioned by their median Spearman correlation with the local reference
  and their median held-out-expression NEX-AURC. The axes are deliberately not
  combined into a single score. NEX-AURC = 1 is the random-ranking reference.

## Data contract

- Panel a: 21 main-analysis independent units plus 3 manual-layer donors. The
  latter are displayed as a separate cohort and are never pooled with the 21
  main units.
- Panel b: 21 main-analysis independent units for each of 3 methods and 8
  registered primary scores (`21 × 3 × 8 = 504` unit-score observations).
- Panel-a quantity:
  `ΔNEX = NEX(source_boundary_proximity) − NEX(score)`; positive values mean
  better selective ranking than fixed-QC.
- Panel-b internal axis: unit-level Spearman correlation with the frozen local
  reference; higher is better.
- Panel-b external axis: held-out-expression normalized excess AURC; lower is
  better and 1 denotes random ranking.
- Centers are medians. Panel-a intervals are IQRs for `n≥8` and full ranges for
  `n=3`. No pooled spot-level inference or population-level P value is used.

## Archetype and hierarchy

Quantitative grid. Panel a is the cohort-level external-utility summary; Panel
b is the mechanistic claim-boundary panel that demonstrates the non-equivalence
of internal fidelity and external utility.

## Claim boundary

Held-out expression is a cost-feature-external but model-conditioned auxiliary
witness, not biological correspondence truth. The figure does not rank methods
overall, prove transfer to anatomical labels, or imply that low local fidelity
precludes task-specific diagnostic utility.

