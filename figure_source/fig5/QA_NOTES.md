# Figure 5 QA notes

## Data integrity

- Frozen WP5 unit table: 324 rows before witness selection.
- Panel a selects held-out-expression rows only and pivots to 72 unit-method
  rows from 24 independent units.
- Frozen cohort counts are 10 spatialDLPFC, 8 HER2ST controlled, 3 Legacy
  replication and 3 separately reported manual-layer donors per method.
- The maximum paired difference between finite-response and local-reference
  `ΔNEX` is below 0.004.
- Panel b merges the frozen internal and external WP9 tables one-to-one by
  independent unit, method and score.
- The eight scores shown in Panel b are exactly the scores marked `primary` in
  the frozen registry. Endpoint response and local reference are non-primary
  audit references and therefore appear in Panel a but not the practical-score
  panel.
- Panel b contains 504 unit-score rows (`21 × 3 × 8`) and plots medians across
  the 21 independent main-analysis units.
- No observation is sampled or removed for aesthetics.

## Frozen source hashes

- WP5 same-score unit table:
  `0b86a0cccf5c07f225b7fe51512ac0dfd4b5a1d0bcbecad00e24c674ffa2df71`
- WP9 internal-audit unit table:
  `8ad2a501151cc5a09cc75010d9a13190159a52064ed21ddabba8884a754c8ef4`
- WP9 external-audit unit table:
  `b9e4267572a79585016c1012af392f6e0ebcb108a629999d2ceb0db77ede9e27`
- WP9 score registry:
  `31f8d38f1b1470b4d34b97d0078e3264551f4a911bc04dc79ed19733e0420705`

## Visual and export QA

- White background and the same method palette as Figures 2–4.
- Panel a preserves independent-unit variation; Panel b uses a compact shared
  score key instead of point-by-point annotations and leader lines.
- The zero line in Panel a and random-ranking line in Panel b have
  estimand-defined meanings; no arbitrary gate is plotted.
- Editable PDF and SVG, 300-dpi PNG and 600-dpi LZW TIFF are exported from the
  same Python source.

## Reviewer-risk boundary

Do not describe positive `ΔNEX` as a proportional biological-error reduction.
Do not infer anatomical or spot-to-spot correspondence validity from held-out
expression. Full unit-level practical-score distributions belong in the
supplementary audit figure; the main dual-axis panel reports registered-score
medians for legibility.
