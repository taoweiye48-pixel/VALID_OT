# VALID-OT: code and data release

This repository contains the code, configurations, provenance records,
manuscript source-data tables and reproducibility materials for VALID-OT.
The manuscript source files and rendered paper figures are intentionally kept
outside this code-and-data release.

## Included

- `code/validot/`: core solvers, intervention definitions, metrics and witness
  evaluation;
- `code/run_*.py` and `code/prepare_*.py`: analysis and preparation entry
  points, including WP1--WP11;
- `configs/`: versioned analysis configurations;
- `code/tests/`: numerical and regression tests;
- `results/`: result tables and manifests used by the manuscript;
- `figure_source/`: figure-generation scripts and panel-level source data;
- `data/`: public-data provenance, pair registries and checksum manifests;
- `docs/`: reproduction, third-party software and data/code availability
  guidance;
- `code/reproducibility/`: environment and execution instructions.

## Installation

Use Python 3.10 for the frozen environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code\requirements-v1.3-core.txt
python -m pip install -e .
```

## Verification

```powershell
python -m pytest -q code\tests
```

The test requiring the separately archived legacy reference installation is
skipped unless `VALIDOT_LEGACY_ROOT` is set. Figure scripts regenerate outputs
under the ignored `figures/` directory from `figure_source/` and `results/`.

## Reproduction

Read [code/reproducibility/README_REPRODUCE.md](code/reproducibility/README_REPRODUCE.md)
and [data/README.md](data/README.md) first. Download the public source datasets
from their official records, record their versions and licences in
`data/provenance/`, prepare the repository-relative input directories, then
run the frozen configurations in `configs/`.

Raw public datasets and local processed caches are not mirrored here. Their
source objects, section identifiers, preparation contracts and checksums are
recorded in `data/provenance/`. The optional PASTE/PASTE2/3d-OT wrappers and
licence notes are documented in `docs/THIRD_PARTY.md`.

## Public release checklist

Before making the repository public:

1. choose and add an OSI-compatible software licence;
2. fill `CITATION.cff.template` and save the final `CITATION.cff`;
3. create an immutable release tag and archive it with a DOI;
4. replace the URL/DOI placeholders in the manuscript's Data and Code
   Availability statements.
