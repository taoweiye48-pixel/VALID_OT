# VALID-OT downstream positive-pair and representation analysis

## Analysis boundary

This package evaluates whether score choice changes pseudo-positive purity and a controlled
positive-pair-conditioned representation-transfer task. It is not a full AlignDG reproduction
and does not establish general real-section biological utility.

## Independent unit

HER2ST patient is the independent unit (n=8 in the full run). Directions, spatial folds and
random-gate repeats are aggregated within patient before inference.

## Positive-pair quality

| coverage | method | n_independent_units | median_difference | q25_difference | q75_difference | bootstrap_median_ci95_low | bootstrap_median_ci95_high | units_improved | units_worsened | units_tied | wilcoxon_two_sided_p_raw | wilcoxon_p_holm | max_probability_median_precision | max_probability_q25_precision | max_probability_q75_precision | top2_margin_median_precision | top2_margin_q25_precision | top2_margin_q75_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | balanced_ot | 8 | 0.127317 | 0.104297 | 0.149537 | 0.0945122 | 0.150474 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.660108 | 0.626822 | 0.766593 | 0.800812 | 0.728198 | 0.922632 |
| 0.8 | row_softmax | 8 | 0.063056 | 0.0476974 | 0.0985319 | 0.0380117 | 0.120155 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.857229 | 0.7953 | 0.940878 | 0.958664 | 0.896849 | 0.990086 |
| 0.8 | uot | 8 | 0.110434 | 0.0851759 | 0.124442 | 0.0804094 | 0.127907 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.741056 | 0.707634 | 0.906014 | 0.870115 | 0.844565 | 0.990685 |
| 0.9 | balanced_ot | 8 | 0.0592836 | 0.0377213 | 0.0688648 | 0.0271739 | 0.0689655 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.673696 | 0.632487 | 0.777967 | 0.738667 | 0.663177 | 0.847221 |
| 0.9 | row_softmax | 8 | 0.0306521 | 0.0237644 | 0.052947 | 0.0157895 | 0.0568966 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.864031 | 0.804777 | 0.93333 | 0.922848 | 0.841505 | 0.965672 |
| 0.9 | uot | 8 | 0.0492779 | 0.0470301 | 0.0624335 | 0.0428571 | 0.0691057 | 8 | 0 | 0 | 0.0078125 | 0.0234375 | 0.753391 | 0.724929 | 0.902333 | 0.818482 | 0.780082 | 0.955765 |

## Representation transfer: primary endpoint

| method | n_independent_units | median_difference | q25_difference | q75_difference | bootstrap_median_ci95_low | bootstrap_median_ci95_high | units_improved | units_worsened | units_tied | wilcoxon_two_sided_p_raw | metric | wilcoxon_p_holm | primary_endpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_ot | 8 | 0.0475535 | 0.00800728 | 0.0770293 | 0 | 0.118567 | 6 | 1 | 1 | 0.03125 | top1 | 0.09375 | True |
| row_softmax | 8 | 0.0134395 | 0 | 0.120763 | 0 | 0.221597 | 4 | 0 | 4 | 0.125 | top1 | 0.125 | True |
| uot | 8 | 0.00978252 | 0 | 0.11709 | 0 | 0.226352 | 5 | 0 | 3 | 0.0625 | top1 | 0.125 | True |

## Run status

```json
{
  "state": "COMPLETED",
  "started_local_epoch": 1784712916.6376472,
  "config_path": "C:\\Users\\Administrator\\Desktop\\MHAgent_Auditing finite-intervention m_20260717_163616\\workspace\\configs\\postreview_downstream_v1.json",
  "config_sha256": "be2abd5e1bb282b5410ceff2126c8bfbea75ccc0acebadc76a1797a67658e6a4",
  "smoke": false,
  "python": "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.22631-SP0",
  "positive_pair_quality": {
    "source": "C:\\Users\\Administrator\\Desktop\\MHAgent_Auditing finite-intervention m_20260717_163616\\workspace\\results\\postreview_wp1_wp10_v1\\wp10\\wp10_her2st_correspondence_truth_direction.tsv",
    "source_sha256": "37d09b3cdea89ebff5a84a19b7e4705b4dbdd015e3743a5fcf5c5cea074467ba",
    "n_direction_method_rows": 48,
    "n_independent_units": 8,
    "n_methods": 3
  },
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
        "median_difference": 0.047553529628617086,
        "q25_difference": 0.008007276081164173,
        "q75_difference": 0.07702933407676378,
        "bootstrap_median_ci95_low": 0.0,
        "bootstrap_median_ci95_high": 0.11856731426251466,
        "units_improved": 6,
        "units_worsened": 1,
        "units_tied": 1,
        "wilcoxon_two_sided_p_raw": 0.03125,
        "metric": "top1",
        "wilcoxon_p_holm": 0.09375,
        "primary_endpoint": true
      },
      {
        "method": "row_softmax",
        "n_independent_units": 8,
        "median_difference": 0.013439464039787935,
        "q25_difference": 0.0,
        "q75_difference": 0.12076306361910732,
        "bootstrap_median_ci95_low": 0.0,
        "bootstrap_median_ci95_high": 0.2215971559365918,
        "units_improved": 4,
        "units_worsened": 0,
        "units_tied": 4,
        "wilcoxon_two_sided_p_raw": 0.125,
        "metric": "top1",
        "wilcoxon_p_holm": 0.125,
        "primary_endpoint": true
      },
      {
        "method": "uot",
        "n_independent_units": 8,
        "median_difference": 0.009782521565899392,
        "q25_difference": 0.0,
        "q75_difference": 0.1170902214445575,
        "bootstrap_median_ci95_low": 0.0,
        "bootstrap_median_ci95_high": 0.22635172019140343,
        "units_improved": 5,
        "units_worsened": 0,
        "units_tied": 3,
        "wilcoxon_two_sided_p_raw": 0.0625,
        "metric": "top1",
        "wilcoxon_p_holm": 0.125,
        "primary_endpoint": true
      }
    ]
  },
  "completed_local_epoch": 1784712935.2590444,
  "elapsed_seconds": 18.621397256851196
}
```
