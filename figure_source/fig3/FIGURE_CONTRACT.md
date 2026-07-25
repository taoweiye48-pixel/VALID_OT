# Figure 3 contract — local-to-endpoint transportability

## Core conclusion

After Figure 2 establishes that the finite-step score accurately recovers the
high-accuracy local response, Figure 3 asks whether that local response can be
transported to the complete finite-intervention endpoint. Transportability is
high for Arm-R expression deletion, lower for Arm-N expression deletion and
limited for spatial-channel deletion. The endpoint mismatch is associated
mainly with condition-specific path-speed changes rather than large path
reversals.

## Evidence identity

- Frozen source: `data/wp4_path_geometry_unit.tsv`.
- Main-analysis independent units: 21.
- Families: three methods × two arms × two cost channels = 12.
- Each family contains exactly 21 independent-unit observations.
- All summaries are descriptive medians and interquartile ranges; no spot-level
  or pooled population inference is made.

## Panel map

- **a — Endpoint rank transportability:** Spearman correlation between the
  high-accuracy local response $r^L$ and the complete endpoint response $r^E$.
- **b — Endpoint magnitude error:** relative mean absolute error (rMAE) between
  $r^L$ and $r^E$.
- **c — Path directness:** median row-level directness $\eta$, where 1 denotes a
  path whose accumulated length equals its endpoint displacement.
- **d — Path-speed change:** median row-level late/early speed ratio $\kappa$;
  values below 1 indicate deceleration and values above 1 acceleration.

## Visual encoding

- Small translucent points: independent biological units.
- Thick coloured segment: interquartile range.
- Large coloured point: median.
- Method colours follow the shared project palette.
- The categorical y-axis follows Figure 2 exactly: `R — expr`, `R — spatial`,
  `N — expr`, `N — spatial`. No quantitative coordinate is jittered.
- Dashed grey reference lines in panels c and d mark $\eta=1$ and $\kappa=1$.

## Reviewer boundaries

- High $\eta$ rules out large aggregate detours but does not prove path
  linearity in every response component.
- $\kappa$ describes temporal change along the frozen intervention path; it
  does not uniquely identify channel removal, compensation or regularization
  as a causal mechanism. The WP11 response surface addresses those factors.
- The figure evaluates model-relative endpoint transportability, not
  biological correspondence truth.
