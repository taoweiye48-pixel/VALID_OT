# Figure 4 QA notes

## Data integrity checks embedded in the plotting script

- Condition registry: 34 rows.
- Unit-level response-surface table: 2,142 rows (`21 × 3 × 34`).
- Unit-level factorial table: 2,016 rows.
- Fixed/coregularized grid validated at `u={0.50,0.75,1.00}` and
  `v={0,0.25,0.50,0.75,1.00}`.
- Each plotted factorial family contains 21 independent units.
- Co-regularized response is invariant to `u` within each unit, method and
  selected endpoint to a maximum deviation below `8.8e-16`.
- Panel c computes `fixed − co-regularized` after pairing by independent-unit
  identifier; it is not a difference of separately calculated medians.
- All 126 paired Panel-c observations at `u=1` are equal within `1e-12`.

## Visual and export checks

- White background; no decorative framing.
- Method colours match Figures 2 and 3.
- One shared sequential colour scale is used for all heatmaps.
- Panel letters and headings use a consistent hierarchy.
- Unit points are subordinate to medians and interquartile ranges.
- Panel c uses three paired-difference trajectories instead of six overlapping
  absolute-response trajectories; the zero line has an estimand-defined role.
- Natural zero references are shown for factorial contrasts and paired scale
  differences; no arbitrary gate threshold is added.
- Editable PDF and SVG, 300-dpi PNG and 600-dpi LZW TIFF are exported from the
  same source script.

## Reviewer-risk boundary

Use “response grid” or “sparse response surface”; do not claim a continuous
analytic response surface. Interpret Panel c as an objective-scale control,
not proof that one mechanism uniquely causes the Arm R/Arm N difference.
