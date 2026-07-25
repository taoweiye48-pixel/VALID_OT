# P1 Amendment 001: expansion to independent donor units

Status: frozen before inspection of P1 v2 outcomes  
Date: 2026-07-18  
Scope: P1 cost-scale/regularization sensitivity only

Note: the label and sampling clauses below are superseded by Amendment 002 after the full object's incomplete manual-label coverage was audited. The donor/pair selection remains unchanged.

## Reason for amendment

The original P1 plan limited the real-data analysis to three independent units already present in the frozen VALID-OT benchmark. The author subsequently required a materially larger sample and explicitly withdrew the restrictions “use existing real data only”, “three real independent units”, and “do not add new data”. No P1 v1 solver task had completed when that instruction was received; the checkpoint directory contained zero completed tasks. P1 v1 files remain preserved as an aborted historical design and are not overwritten.

## New sampling design

The primary expansion cohort is the public `spatialDLPFC` resource: 30 Visium sections from ten neurotypical adult donors, sampled at anterior, middle, and posterior DLPFC positions. The unit of independence is the donor, not the section or analysis direction.

Before inspecting any P1 v2 outcome, the following deterministic rule is fixed:

1. Include every donor with both an anterior and a middle section.
2. Use exactly one anterior-to-middle biological pair per donor.
3. Evaluate both directions, then average directions before any donor-level aggregation.
4. Do not substitute posterior sections based on alignment performance.
5. Use manual cortical layer labels and exclude missing/`None` labels before deterministic label-stratified sampling.
6. Sample at most 1,500 spots per section using seed 20260718.
7. Select 500 cost genes and 100 non-overlapping held-out genes by pooled variance within each donor pair. Held-out genes are never used to construct the OT cost.

The ten planned donors are `Br2720`, `Br2743`, `Br3942`, `Br6423`, `Br6432`, `Br6471`, `Br6522`, `Br8325`, `Br8492`, and `Br8667`. Thus the primary expanded analysis has `n=10` independent donor units and 20 directional evaluations.

The three independent units from the legacy benchmark are retained as a separately identified replication stratum. The pooled sensitivity summary therefore contains 13 independent units, but results must also be reported for the ten-donor primary cohort and the three-unit legacy cohort separately. Repeated developmental pairs remain one independent legacy unit.

## What changes and what does not

Changed:

- permitted data sources;
- number and composition of real independent units;
- preprocessing seed for the new cohort;
- analysis/output version and paths;
- cohort-stratified reporting.

Unchanged:

- P1 scientific question;
- R, N, and S intervention definitions;
- epsilon/tau/temperature grid;
- solver tolerance and failure policy;
- internal fidelity metrics and thresholds;
- external witnesses, controls, and thresholds;
- pair-first and equal-independent-unit aggregation;
- the registered v1.2 external Gate result (zero passes);
- the status of P1 as a post-review sensitivity analysis rather than confirmatory evidence.

## Interpretation guardrails

- A section, direction, or multiple pair from one donor is not an additional independent sample.
- The nominal primary sample size is ten donors; the pooled sample size is thirteen independent units.
- Statistical summaries must expose donor counts, cohort counts, failures, and missing values.
- The expanded cohort cannot retroactively convert registered failure into success.
- No result-dependent donor exclusion, grid expansion, threshold change, or witness selection is permitted.
- If the full public object cannot be parsed reproducibly, P1 v2 stops and records a data-access blocker; the design must not silently fall back to treating the 12 legacy DLPFC sections as independent donors.
