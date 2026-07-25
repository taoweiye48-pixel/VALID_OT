# P1 Amendment 004: spatialLIBD manual cortical-layer extension

Date: 2026-07-18

Status: frozen before any solver output for this extension

## Rationale

The 30-section spatialDLPFC expansion contains spots in every section, but its
manual layer field is not complete across the ten-donor cohort.  Data-driven
spatial-domain labels cannot be presented as manual cortical-layer truth.  A
separate official spatialLIBD cohort is therefore added to evaluate the
label-based witnesses against manually assigned L1--L6/white-matter labels.

## Frozen design

- Data source: official spatialLIBD Human DLPFC Visium processed object.
- Sections: 12.
- Independent donors: 3 (`Br5292`, `Br5595`, `Br8100`).
- Biological pairs: 6 adjacent-replicate pairs, two per donor.
- Directions: forward and reverse, averaged before pair and donor aggregation.
- Spots: at most 1,500 manually labelled spots per section, selected with fixed
  seeds without using the layer label.
- Labels: `layer_guess_reordered`, restricted to L1--L6 and WM.
- Methods, intervention arms, epsilon/tau grid, scores, controls, gates, solver
  tolerance and failure policy: unchanged from the P1 v2 sensitivity design.
- Outputs: separate versioned analysis, results and checkpoint directories.

## Interpretation constraints

1. Twelve sections and six pairs are not independent sample sizes; inferential
   summaries use three equal-weight donor units.
2. The labels are manual layer truth, but adjacent sections do not provide an
   exact spot-to-spot correspondence truth.
3. This extension can strengthen or weaken label-witness evidence, but it does
   not retroactively change the immutable registered external gate.
4. Results are reported separately from the ten-donor real cross-position
   cohort, the eight-patient HER2ST controlled cohort and legacy cohorts.
