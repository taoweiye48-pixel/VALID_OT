# Figure 2 QA notes

## Data-integrity guards executed by the plotting script

- WP1 adjacent-step rows: 3,000/3,000.
- WP1 local-reference rows: 600/600.
- Row-softmax analytic checks: 200/200; all pass median and q90 error gates.
- Balanced-OT implicit checks: 104/104; all converged and all pass four frozen
  agreement gates.
- Independent UOT convex-dual/implicit checks: 48/48 across four fixed units,
  three parameter settings and four Arm--channel conditions; all pass.
- UOT baseline-plan relative L1 is at most 2.94e-7 and global derivative
  relative L1 is at most 6.12e-7; no row is non-estimable.
- WP3 unit-family rows: 252/252 from 21 independent units.
- WP3 family rows: 12/12; every family and every unit-family row passes.
- No missing rows are silently dropped after the declared filters.

## Scientific boundaries

- Panel a's 0.05 threshold applies to the smallest adjacent-step comparison
  only; the full line is a visual reference.
- Panel b does not infer biological variation from numerical conditions.
- Panel b is restricted to the observed $10^{-9}$--$10^{-6}$ range; its
  frozen $10^{-2}$ gate is explicitly labelled off-scale rather than being
  drawn at the cost of four orders of magnitude of empty plotting space.
- The independent UOT check uses fixed 96-by-96 deterministic subsamples and
  is a numerical implementation check rather than an additional biological
  replicate.
- Passing Figure 2 supports local-response fidelity at $h=0.01$; endpoint
  transportability and external utility are tested elsewhere.

## Visual encoding

- Method identity is redundantly encoded by facet title and colour.
- Arm and channel identity are redundantly encoded by line style and marker.
- Near-coincident Arm R and Arm N curves are separated by luminance, dash and
  marker-fill hierarchy without jittering or changing either coordinate.
- Panel c displays all unit-level points and separately marks family medians.
- Panel c uses one compact method-colour key plus matching group bands, avoiding
  repeated method names beside all 12 rows.
- Frozen thresholds are labelled directly and are not encoded by colour alone.
- Encoding definitions and validation-condition counts are kept in the figure
  legend rather than repeated as small text inside the plotting area.

## Export and preflight

- Final size: 177.8 mm wide, double-column layout.
- Minimum source text size: 5 pt.
- Editable vectors: SVG and PDF.
- Raster exports: 300-dpi PNG and 600-dpi LZW TIFF.
- Final-width PNG inspected for clipping, overlap and legibility.
- Automated source preflight: 14 PASS, 0 WARN, 0 FAIL.
