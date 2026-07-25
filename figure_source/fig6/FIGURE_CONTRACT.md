# Figure 6 contract

Core conclusion: Under controlled HER2ST correspondence truth, top-two
probability margin improves retained candidate-positive-pair precision relative
to maximum coupling probability at the same coverage, while external utility
remains diagnostic-task- and witness-specific.

Figure archetype: asymmetric quantitative validation figure with one hero
comparison and two supporting boundary panels.

Target/output: two-column figure, 7.0 inches wide; editable SVG/PDF, 300 dpi PNG
and 600 dpi LZW-compressed TIFF.

Backend: Python/matplotlib only.

Panel map:

- a (hero): Patient-level precision gain of top-two probability margin over
  maximum coupling probability at 80% primary and 90% sensitivity coverage.
  Small marks are eight patients; large marks and thick horizontal segments are
  patient medians and frozen bootstrap 95% intervals.
- b: Median patient-level AUROC for eight registered scores in top-1 mismatch
  and crop-missingness detection. Black outlines mark descriptive row maxima.
- c: Paired donor-level gain over fixed-QC for held-out-expression and
  manual-layer witnesses using the same exact Arm-R expression response.

Evidence hierarchy:

- Hero evidence: fixed-budget candidate-pair precision consequence.
- Validation evidence: task-specific controlled diagnostic AUROC.
- Boundary evidence: failure of utility to transfer to manual-layer mismatch.

Independent-unit boundaries:

- Panels a-b: eight controlled HER2ST patients. Forward/reverse directions are
  averaged within patient before all summaries and tests.
- Panel c: three spatialLIBD donors. Directions/sections are not treated as
  independent observations.
- The cohorts are never pooled.

Reviewer risks:

- HER2ST targets are controlled within-section correspondences, not independent
  adjacent-section biological replicates.
- Higher retained-pair precision is an intermediate selection result and does
  not establish improved representation learning or biological downstream
  performance.
- A high AUROC for a task-specific score is not general correspondence truth.
- Manual layers are an anatomical witness, not exact spot-to-spot truth.
- Do not claim universal manual-layer negativity: Row-softmax is positive in
  Br5595, although its three-donor median is negative.
