# Figure 6 legend

**Figure 6 | Controlled diagnostic utility changes candidate-pair selection but
remains witness-specific.** **a,** Patient-level gain in retained candidate-pair
precision when top-two probability margin replaces maximum coupling
probability at fixed coverage in eight controlled HER2ST patients. Forward and
reverse directions were averaged within patient. Small symbols are patients;
large symbols and thick segments are medians and deterministic patient
bootstrap 95% intervals. Filled circles denote the prespecified 80% primary
coverage and open triangles the 90% sensitivity analysis. At 80% coverage,
median gains were 0.127, 0.110 and 0.063 for Balanced OT, UOT and Row-softmax;
all three improved in 8/8 patients (two-sided exact Wilcoxon signed-rank tests,
Holm-adjusted P=0.023). The corresponding 90% gains were 0.059, 0.049 and
0.031. **b,** Median patient-level AUROC for top-1 mismatch and crop-missingness
detection using eight registered scores. Black outlines mark the highest median
within each method-task row; they are descriptive rather than newly selected
confirmatory endpoints. Labels denote assigned raw cost (cost), barycentric
displacement (bary.), conditional entropy (entropy), the finite response at
`h=0.01` [`s(0.01)`], one minus maximum row probability (low-p), probability-
margin risk (margin), source-boundary proximity (boundary) and transported-mass
deficit (mass). The shared colour scale spans AUROC 0--1; AUROC=0.5 denotes
random discrimination. **c,** Gain over the frozen fixed-QC comparator for held-out-
expression and manual-layer witnesses in the same three spatialLIBD donors,
using the exact Arm-R expression response at the default setting
(`epsilon=0.25`; UOT `tau=2`). Thin grey lines connect witnesses within donor;
coloured diamonds show donor medians. Positive gain denotes lower NEX-AURC than
source-boundary-proximity fixed-QC. HER2ST comprises controlled within-section
targets rather than independently sampled adjacent sections; retained-pair
precision does not establish improved representation learning, and manual
layers do not provide exact spot-to-spot correspondence truth. Source data are
provided with the figure source files.
