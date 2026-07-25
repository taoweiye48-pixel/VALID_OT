# Data and provenance

## Public source data

The analyses reuse public spatial-omics resources rather than generating a
new human cohort:

- spatialDLPFC;
- HER2ST;
- Stereo-seq;
- STARmap PLUS;
- spatialLIBD manual cortical-layer annotations.

The exact source objects, section identifiers, preparation rules, source
commits and checksums are recorded in `data/provenance/` and in the frozen
configuration files under `configs/`. Obtain each source from its official
repository or data record and retain the original licence and citation.

## Local data layout for a rerun

Place downloaded inputs in the following repository-relative locations:

```text
data/raw/
├── p1_v2_spatialDLPFC/
├── p1_v2_her2st/github_master/
└── p1_v2_dlpfc_manual_layers/

data/processed/
├── external_pairs/
├── p1_v2_expanded_pairs/
└── p1_v2_manual_layer_pairs/
```

The preparation scripts accept environment variables for alternative locations:

- `VALIDOT_SPATIALDLPFC_ROOT`;
- `VALIDOT_HER2ST_ROOT`;
- `VALIDOT_MANUAL_LAYER_RAW_ROOT`;
- `VALIDOT_LEGACY_PAIRS_ROOT`;
- `VALIDOT_EXPANDED_PAIRS_ROOT`;
- `VALIDOT_MANUAL_LAYER_PAIRS_ROOT`.

Raw and processed data are excluded by `.gitignore`. They are not deleted from
the original local project by the release-staging process.

## Dataset-to-result mapping

The result directories under `results/` contain the numerical outputs used by
the manuscript. Figure-level source tables are stored under
`figure_source/*/data/`; the corresponding scripts and figure contracts are
in the same directories. SHA-256 manifests are retained where available.
