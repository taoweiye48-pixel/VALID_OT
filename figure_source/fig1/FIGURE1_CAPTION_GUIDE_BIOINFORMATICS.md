# Figure 1 caption and notation guide

## Current manuscript version

Use fig1_audit_design_local_fidelity.* for the revised manuscript. Earlier
compact and white-modular exports remain archived visual references and must
not replace the revised evidence hierarchy.

## Compact response notation

Define the row-response curve once:

\[
R_i^{A,k}(t)=D_i(P^0,P^{A,k}(t)).
\]

Then use:

\[
s_i(h)=\frac{R_i^{A,k}(h)}{h},\qquad
r_i^L=\partial_tR_i^{A,k}(0^+),\qquad
r_i^E=R_i^{A,k}(1).
\]

Here \(L\) denotes the high-accuracy local reference and \(E\) denotes the
fully reoptimized endpoint response. These short superscripts replace the
longer working labels \(r^{\mathrm{loc}}\) and \(r^{\mathrm{end}}\).

## Recommended English legend

Use the complete text in FIGURE_LEGEND.md. Its required content is:

- panel a defines \(D_i\), \(s_i(0.01)\), \(r_i^L\), \(r_i^E\), \(w_i\), and
  the question-matched reference slot \(r^\ast\);
- panel b gives the exact Arm R/N paths and states that Arm S is a synchronous
  objective-scale control rather than a third finite-difference path;
- panel c distinguishes
  \(s(0.01)\leftrightarrow r^L\),
  \(r^L\leftrightarrow r^E\), and
  \(s(0.01)\leftrightarrow r^E\), while keeping
  response score \(\rightarrow w\) on a separate evidence axis;
- the closing sentence states that all response references are model-relative
  and do not independently establish biological correspondence.

## Required terminology boundaries

- Do not call \(r^L\) or \(r^E\) biological ground truth.
- Do not describe Arm S as a finite-difference path.
- Do not imply that the end-to-end readout is an algebraic product or sum of
  the two diagnostic modules.
- Do not call every witness statistically independent. Report its dependency
  class in Methods.
- Use “external utility” for score-to-witness ranking and “local fidelity” for
  the two model-response diagnostic modules.

## Layout and export

- Final width: approximately 178 mm.
- Editable PDF and SVG are the production formats.
- TIFF is 600 dpi with LZW compression; PNG is a 300 dpi preview.
- Use the 100 dpi final-width preview for readability checks.
