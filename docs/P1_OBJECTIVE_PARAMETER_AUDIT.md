# P1 objective and parameter audit

Analysis version: `valid-ot-p1-scale-regularization-sensitivity-v1`  
Claim status: **post-review sensitivity analysis**

## Audited implementation

The active implementation is `code/validot/solvers.py` as called by
`code/validot/benchmark.py`. Expression and cross-slice spatial costs are each
divided once by their own strictly-positive median in `cost_components` before
the weighted sum is formed. No solver performs a second cost normalization.

## Balanced entropic OT

For source masses \(a\), target masses \(b\), cost \(C\), and plan \(P\), the
implemented solver minimizes the standard entropy-regularized balanced problem

\[
\min_{P\ge 0,\;P\mathbf 1=a,\;P^\top\mathbf 1=b}
\langle P,C\rangle+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1).
\]

`epsilon` is used directly in `exp(-C/epsilon)` and in the identical entropic
dual optimized by the Newton refinement. It is not transformed before use.
Consequently, \(C\mapsto\lambda C\) and
\(\varepsilon\mapsto\lambda\varepsilon\) multiply the full objective by
\(\lambda>0\) and preserve the minimizer.

## KL-unbalanced entropic OT

With two finite penalties, the implemented fixed-point updates use

\[
q_a=\frac{\tau_a}{\tau_a+\varepsilon},\qquad
q_b=\frac{\tau_b}{\tau_b+\varepsilon},
\]

and the kernel `exp(-C/epsilon)`. These are the standard updates for

\[
\min_{P\ge0}\;\langle P,C\rangle+
\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)
+\tau_a\,\mathrm{KL}(P\mathbf1\|a)
+\tau_b\,\mathrm{KL}(P^\top\mathbf1\|b).
\]

The public `uot_tau` setting is passed directly and separately as both
`tau_a` and `tau_b`; it is therefore a direct KL marginal-penalty coefficient,
not a transported-mass target and not an internally transformed user parameter.
The only derived quantities are the two fixed-point exponents above. P1 keeps
the penalties symmetric. Objective-preserving scaling therefore requires
simultaneous scaling of \(C\), \(\varepsilon\), \(\tau_a\), and \(\tau_b\).

## Row-softmax comparator

For each source row, the comparator computes

\[
P_{ij}=a_i\,\mathrm{softmax}_j(-C_{ij}/T),
\]

where the implementation names the temperature argument `epsilon`. It is a
temperature, not a balanced-OT regularizer. Scaling \(C\) and \(T\) by the
same positive constant leaves every row probability unchanged.

## Parameter transformation used in Arm S

P1 fixes \(\lambda=0.5\):

- balanced OT: `C *= 0.5`, `epsilon *= 0.5`;
- UOT: `C *= 0.5`, `epsilon *= 0.5`, `tau_a *= 0.5`, `tau_b *= 0.5`;
- row-softmax: `C *= 0.5`, `temperature *= 0.5`.

Arm S is an implementation invariant and not an independent biological result.
Failure of the plan-level equivalence tolerance is retained as a blocker and
precludes any claim that the control was objective preserving.

## Scope boundary

This audit does not establish primal-dual/KKT certificates (P4), add witnesses,
alter the v1.2 intervention, alter Gate thresholds, or replace any v1.2/v1.3
frozen result.
