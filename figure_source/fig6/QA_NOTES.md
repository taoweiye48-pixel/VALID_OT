# Figure 6 QA notes

## Frozen inputs

- `positive_pair_quality_unit.tsv` — SHA256
  `F3A21F207647CB212EE525A35A667DA45609ADD45D550BC7075DE86BE8E6E578`.
- `positive_pair_quality_summary.tsv` — SHA256
  `7B3DFF870FC5A73FAE6C724FDBBB414EAA2F45F2EBB893521B59E57F4A55F66A`.
- `wp10_her2st_correspondence_truth_direction.tsv` — SHA256
  `37D09B3CDEA89EBFF5A84A19B7E4705B4DBDD015E3743A5FCF5C5CEA074467BA`.
- `wp10_crop_missingness_utility_direction.tsv` — SHA256
  `B4929C72B9015CE6747698EB77737E0FFBB387744412ADD95D90BA7EE0FB8C0C`.
- `p1_manual_layer_validation.csv` — SHA256
  `0BA3C0B5AB781385795E66E2BC0EB5B522E7190B9308A3A7523FF4A30536A111`.

No plotted value is transcribed manually. The script asserts independent-unit
counts, primary medians, bootstrap intervals, adjusted P values and locked
witness medians before plotting.

## Aggregation and statistics

- Panel a: 8 HER2ST patients x 3 methods. Directions were averaged within
  patient upstream. The 80% endpoint is primary and 90% is sensitivity.
  Horizontal intervals are frozen deterministic patient-bootstrap 95%
  intervals for the median. P values use two-sided exact Wilcoxon signed-rank
  tests across patients with Holm correction across three methods.
- Panel b: 8 patients x 3 methods x 8 scores per task after within-patient
  direction aggregation; cells show patient medians.
- Panel c: 3 donors x 3 methods x 2 witnesses = 18 donor-level rows.
- No cohort pooling or spot-level pseudoreplication is used.

## Visual encoding

- Method colours match Figures 2-5.
- Panel a small symbols are patients; large symbols and thick segments are
  medians and bootstrap intervals. Filled circles denote 80% and open triangles
  90% coverage.
- Panel b uses one explicit shared 0-1 AUROC colour scale centred at random
  performance (0.5).
- Panel c uses donor marker shapes and paired lines; zero denotes equality to
  fixed-QC, not equality to random ranking.

## Claim boundary

- Controlled candidate-pair precision and diagnostic performance are task-
  specific.
- Higher retained-pair precision is not evidence of improved representation
  learning or a complete downstream pipeline.
- The outlined heat-map cells identify row-wise descriptive maxima.
- The manual-layer panel is a three-donor transfer boundary and does not
  establish biological correspondence truth.
