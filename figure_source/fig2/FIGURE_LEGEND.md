# Figure 2 legend

**Multi-step and independent derivative checks validate the local-response
reference at $h=0.01$.** *Expr* and *spatial* denote expression-derived and
coordinate-derived cost channels. Arm R lowers the intervened coefficient from
0.5 to 0 while holding the retained coefficient at 0.5; Arm N pairs this
reduction with a linear retained-coefficient increase from 0.5 to 1. Row-plan
relative L1 is the per-row L1 deviation between the finite-step response vector
and its high-accuracy local reference, normalized by the reference L1 norm and
summarized across rows. Relative mean absolute error (rMAE) is mean absolute
scalar-response error divided by mean absolute reference response. A *gate* is
a prespecified numerical qualification threshold, not a hypothesis-test cutoff.

**a**, Adjacent-step convergence for Balanced OT, unbalanced OT (UOT) and
Row-softmax across the two arms and cost channels. Lines show medians and
ribbons show interquartile ranges across 50 direction-level numerical
conditions per family and step. The dashed 0.05 gate applies only to the
smallest adjacent-step comparison ($6.25\times10^{-4}$ versus
$1.25\times10^{-3}$).

**b**, Independent derivative cross-validation. Points are conditions; black
bars are medians. Row-softmax finite differences agree with analytic
derivatives in 200 conditions; Balanced-OT finite differences agree with
implicit differentiation in 104 converged conditions from nine fixed units.
For UOT, production log-Sinkhorn finite differences agree with an independently
implemented convex-dual optimizer and first-order implicit derivative in 48
conditions spanning four fixed units, three $(\epsilon,\tau)$ settings and four
Arm--channel combinations. The prespecified $10^{-2}$ gate is off-scale.

**c**, Local fidelity at $h=0.01$ for 252 unit-family observations
(21 independent units × 3 methods × 2 arms × 2 channels). Circles show units
and diamonds family medians; blue, teal and orange denote Balanced OT, UOT and
Row-softmax. Dashed lines are prespecified gates for row-plan relative L1 (0.10),
rMAE (0.10) and neighbourhood error (0.15). All
252 unit-family tests and all 12 family gates pass; family medians have
Spearman $\rho\geq0.999844$, top-decile overlap $\geq0.9933$ and direction
cosine $\geq0.999986$.

## Statistical and data notes

- Independent biological units in panel c: 21.
- Numerical validation conditions: 50 per family and adjacent-step comparison
  in panel a; 200 analytic, 104 Balanced-OT implicit and 48 UOT independent
  conditions in panel b.
- Center statistic: median; spread in panel a: interquartile range.
- No hypothesis test, confidence interval or pooled biological inference is
  claimed in this numerical-validation figure.
