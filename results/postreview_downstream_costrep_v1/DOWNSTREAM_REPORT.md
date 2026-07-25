# VALID-OT downstream positive-pair and representation analysis

## Analysis boundary

This package evaluates whether score choice changes pseudo-positive purity and a controlled
positive-pair-conditioned representation-transfer task. It is not a full AlignDG reproduction
and does not establish general real-section biological utility.

## Independent unit

HER2ST patient is the independent unit (n=8 in the full run). Directions, spatial folds and
random-gate repeats are aggregated within patient before inference.

## Representation transfer: primary endpoint

| method | n_independent_units | median_difference | q25_difference | q75_difference | bootstrap_median_ci95_low | bootstrap_median_ci95_high | units_improved | units_worsened | units_tied | wilcoxon_two_sided_p_raw | metric | wilcoxon_p_holm | primary_endpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_ot | 8 | 0.00225431 | 0.00109649 | 0.00356032 | -0.000877193 | 0.00380328 | 6 | 2 | 0 | 0.0390625 | top1 | 0.117188 | True |
| row_softmax | 8 | 0.000438596 | -0.000384615 | 0.00152916 | -0.00153846 | 0.00243902 | 4 | 2 | 2 | 0.84375 | top1 | 1 | True |
| uot | 8 | 0 | 0 | 0.00033102 | 0 | 0.00132408 | 2 | 1 | 5 | 0.5 | top1 | 1 | True |

## Run status

```json
{
  "state": "COMPLETED",
  "started_local_epoch": 1784713184.0889852,
  "config_path": "C:\\Users\\Administrator\\Desktop\\MHAgent_Auditing finite-intervention m_20260717_163616\\workspace\\configs\\postreview_downstream_costrep_v1.json",
  "config_sha256": "f1af35af8e3113c0da088cc28bdf89434c91442e404fa2d4ae76aa79c1e7e05a",
  "smoke": false,
  "python": "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.22631-SP0",
  "representation_transfer": {
    "n_fold_strategy_rows": 2400,
    "n_independent_units": 8,
    "n_methods": 3,
    "n_directions": 16,
    "all_solvers_converged": true,
    "primary_comparisons": [
      {
        "method": "balanced_ot",
        "n_independent_units": 8,
        "median_difference": 0.002254305977710233,
        "q25_difference": 0.0010964912280701754,
        "q75_difference": 0.003560315430520034,
        "bootstrap_median_ci95_low": -0.0008771929824561405,
        "bootstrap_median_ci95_high": 0.003803282182438193,
        "units_improved": 6,
        "units_worsened": 2,
        "units_tied": 0,
        "wilcoxon_two_sided_p_raw": 0.0390625,
        "metric": "top1",
        "wilcoxon_p_holm": 0.1171875,
        "primary_endpoint": true
      },
      {
        "method": "row_softmax",
        "n_independent_units": 8,
        "median_difference": 0.00043859649122807,
        "q25_difference": -0.0003846153846153846,
        "q75_difference": 0.0015291643747058258,
        "bootstrap_median_ci95_low": -0.0015384615384615385,
        "bootstrap_median_ci95_high": 0.0024390243902439024,
        "units_improved": 4,
        "units_worsened": 2,
        "units_tied": 2,
        "wilcoxon_two_sided_p_raw": 0.84375,
        "metric": "top1",
        "wilcoxon_p_holm": 1.0,
        "primary_endpoint": true
      },
      {
        "method": "uot",
        "n_independent_units": 8,
        "median_difference": 0.0,
        "q25_difference": 0.0,
        "q75_difference": 0.00033102049193268275,
        "bootstrap_median_ci95_low": 0.0,
        "bootstrap_median_ci95_high": 0.001324081967730731,
        "units_improved": 2,
        "units_worsened": 1,
        "units_tied": 5,
        "wilcoxon_two_sided_p_raw": 0.5,
        "metric": "top1",
        "wilcoxon_p_holm": 1.0,
        "primary_endpoint": true
      }
    ]
  },
  "completed_local_epoch": 1784713211.5569482,
  "elapsed_seconds": 27.467962980270386
}
```
