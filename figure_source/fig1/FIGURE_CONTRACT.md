# Figure 1 contract — revised local-fidelity framework

## Core conclusion

VALID-OT fixes the intervention semantics before auditing finite-step local
scores, separates local-response fidelity from local-to-endpoint
transportability, retains end-to-end agreement as an overall readout, and
evaluates external utility on a separate evidence axis.

## Figure archetype

Schematic-led composite with three panels.

## Target and export

- Main-text, full-width figure for a computational genomics methods paper.
- Final width: approximately 178 mm.
- White background and restrained blue–teal–orange palette.
- Editable SVG and PDF; 600 dpi LZW TIFF; 300 dpi PNG preview.

## Panel map

- **a — Audit instance and response quantities:** paired spatial sections
  define frozen expression and spatial costs. A specified cost-channel path is
  completely reoptimized at baseline, local steps and endpoint. The displayed
  quantities distinguish the finite-step score \(s(0.01)\), high-accuracy
  local reference \(r^{L}\), finite endpoint response \(r^{E}\), and
  externally specified witness \(w\). The reference slot \(r^\ast\) is matched
  to the audit question.
- **b — Intervention semantics:** Arm R fixes the retained channel at 0.5;
  Arm N increases it to 1; Arm S synchronously scales the Arm-N endpoint
  objective by 0.5 and is not a third finite-difference path.
- **c — Evidence hierarchy:** the umbrella concept of finite-intervention
  local fidelity contains two diagnostic modules,
  \(s(0.01)\leftrightarrow r^{L}\) and
  \(r^{L}\leftrightarrow r^{E}\). The direct
  \(s(0.01)\leftrightarrow r^{E}\) comparison is shown as an end-to-end
  readout, not as an algebraic composition. External utility is a separate
  axis comparing a response score with witness \(w\).

## Evidence hierarchy

- Hero evidence: panel c, because it resolves the old single-comparison
  ambiguity.
- Structural evidence: panel a fixes the four-tuple and response roles.
- Control evidence: panel b fixes the Arm R/N estimands and Arm S invariant.

## Reviewer-risk controls

- Do not call \(r^{L}\) or \(r^{E}\) biological truth.
- Do not depict Arm S as a finite-difference path.
- Do not imply that the two diagnostics algebraically determine the end-to-end
  metric.
- Do not describe every external witness as statistically independent; its
  dependency class is declared per task.
- Preserve the exact Arm-R, Arm-N and Arm-S coefficients.
