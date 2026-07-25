# v1.2 → v1.3 configuration and estimand changes

Version 1.3 is a corrective analysis after external review. It does not alter the immutable v1.2 Gate decision and must not be described as preregistered.

| Area | v1.2 | v1.3 corrective treatment |
|---|---|---|
| Independent unit | Bidirectional slices could enter cross-pair summaries separately | Average directions within biological pair; cross-stage Stereo-seq is one developmental-series unit |
| Ties | Row-order-sensitive ranking possible | Deterministic fractional tie blocks |
| Endpoint | “Endpoint gradient” | One-sided finite-difference sensitivity; extra solver evaluations required |
| Error witness | Single label mismatch | Shared-label closed-set, source-only open-set, and held-out expression witnesses with coverage |
| QC baseline | Per-task best QC could appear primary | Source boundary proximity is fixed primary; best QC is hindsight oracle envelope |
| External utility | Combined/max scores emphasized | Match score to intervention and witness; combined scores remain separate |
| Robustness | Direction ambiguity | Define direction relative to random ranking and report absolute changes |
| FGW | Historical scaling/proxy inconsistency | Historical PASTE/PASTE2 excluded from confirmatory Gates; future code uses objective-consistent scaling/components |
| Claim | General explanation/algorithm framing | Audit protocol and failure-boundary analysis; no new-solver claim |

The post-hoc UOT expression-intervention signal is exploratory and requires an independent, prospectively frozen replication.
