# Figure 1 QA notes

## Scientific checks

- Arm R endpoint: deleted channel \(0\), retained channel \(0.5\).
- Arm N endpoint: deleted channel \(0\), retained channel \(1.0\).
- Arm S: \(\lambda=0.5\) simultaneous scaling of the Arm-N endpoint cost and
  applicable regularization scales; explicitly not a finite-difference path.
- Local score: normalized row-plan response after a complete re-solve at
  \(h=0.01\), divided by \(h\).
- Exact response: normalized row-plan response after a complete endpoint
  re-solve.
- Reoptimization is labelled a model-response reference rather than biological
  correspondence truth.
- Internal fidelity and external witness utility are displayed as separate
  evidence axes.

## Visual and export checks

- Full-width target: 182.9 mm.
- Minimum explicit text size: 5 pt.
- Lowercase bold panel labels.
- Sans-serif font stack with editable SVG text and PDF TrueType fonts.
- Muted, print-safe palette with labels in addition to colour.
- SVG, PDF, 300 dpi PNG and 600 dpi LZW-compressed TIFF exported.
- Automated source preflight: 14 passed, 0 warnings, 0 failures.
- Visual inspection completed at full resolution and at a 720-pixel
  publication-width preview.
- LaTeX manuscript compiled without unresolved references, box overflow or
  underflow warnings; Figure 1 appears on page 3.

## Layout revision

- Increased vertical clearance between the Arm-R/Arm-N axis labels and their
  coefficient-path equations.
- Enlarged the audit-tuple container and redistributed the four tuple elements
  with equal vertical spacing; the witness connector now terminates at the
  container boundary.
- Shortened and centred the Arm-S title so it remains inside the control panel.
- Increased the gutter and internal padding between the internal-fidelity and
  external-utility cards; long headings now use controlled two-line wrapping.
- Offset all Arm-R and Arm-N direct labels away from their coefficient paths,
  so no label intersects a plotted line at full or final publication width.
- Reflowed the internal-fidelity metrics onto three lines to match the lateral
  padding of the external-utility card.
- Reduced the vertical gap between each intervention-path axis title and its
  coefficient-path equation while preserving a clear typographic hierarchy.

## Data and image integrity

This is a vector method schematic. The slice dots and matrix tiles are fixed
diagram elements and do not represent experimental observations. No biological
image, simulated result or quantitative source-data row is shown.
