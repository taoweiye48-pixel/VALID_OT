# Figure 3 QA notes

## Data audit

- Frozen input rows: 252.
- Independent units: 21.
- Families: 12.
- Rows per family: 21.
- Missing plotted values: 0.
- Source SHA-256:
  `CEC77FCF183A453178C2BA9BEBF7E6BAB9848160E18788397BE1502CF6EBAE21`.

## Visual audit

- White canvas and near-white plotting areas.
- Four panel titles share two aligned baselines.
- Left-column condition labels are shared by the corresponding right panel.
- Condition labels and ordering match Figure 2 exactly.
- No numerical coordinate jitter is used.
- All unit observations, medians and IQRs are visible at final width.
- Method colours are consistent with Figure 2.
- No arbitrary transportability gate is introduced; only the mathematical
  references $\eta=1$ and $\kappa=1$ are shown.
- The $\kappa$ axis is logarithmic only after the source-data gate verifies
  that every plotted $\kappa$ value is strictly positive; no clipping or
  pseudocount is applied.

## Export audit

- PDF and SVG are editable vector outputs.
- PNG is exported at 300 dpi.
- TIFF is exported at 600 dpi with LZW compression.
- Formal basename remains `fig3_scale_equivalence` to overwrite the manuscript's
  existing Figure 3 asset without creating parallel figure versions; the
  scientific content and legend have been replaced.
