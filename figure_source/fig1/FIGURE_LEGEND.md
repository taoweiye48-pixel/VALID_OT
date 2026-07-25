# Figure 1 legend

**Fig. 1 | VALID-OT separates intervention semantics, two local-fidelity
diagnostics and external utility.** **a**, Paired spatial sections define
independently scaled expression and spatial cost channels. For a specified
intervention \(I^{A,k}(t)\), the frozen model is completely reoptimized at the
baseline, local path positions and the endpoint. Row-wise change is measured
by

\[
D_i(P,Q)=\frac{\sum_j|P_{ij}-Q_{ij}|}
{\sum_jP_{ij}+\sum_jQ_{ij}+10^{-12}}.
\]

Writing \(R_i^{A,k}(t)=D_i(P^0,P^{A,k}(t))\), the finite-step score is
\(s_i(0.01)=R_i^{A,k}(0.01)/0.01\); the high-accuracy local reference is
\(r_i^{L}=\partial_tR_i^{A,k}(0^+)\), established by multistep convergence
and, where available, analytic or implicit differentiation; and the complete
finite endpoint response is \(r_i^{E}=R_i^{A,k}(1)\). For each audit instance
\(\mathcal A=(I,r^\ast,s,w)\), the reference role \(r^\ast\) is matched to the
question being tested, while witness \(w\) is specified and
dependency-classed separately. **b**, Arm R reduces the intervened channel
from 0.5 to 0 while retaining the other channel at 0.5. Arm N reduces the
intervened channel to 0 while increasing the retained channel from 0.5 to 1.
Arm S is an objective-scaling control: the Arm-N endpoint cost and all
applicable linear regularization scales are multiplied by \(\lambda=0.5\); it
is not a third finite-difference path. **c**, Finite-intervention local
fidelity is resolved into local-response fidelity,
\(s(0.01)\leftrightarrow r^{L}\), and local-to-endpoint transportability,
\(r^{L}\leftrightarrow r^{E}\). The direct comparison
\(s(0.01)\leftrightarrow r^{E}\) is retained as an end-to-end readout and is
not an algebraic combination of the two diagnostic modules. External utility
separately evaluates whether a response score ranks the specified witness
\(w\). All response references are model-relative; none alone establishes
biological correspondence truth.
