# Reproducing VALID-OT

## Scope

This directory supports the current frozen P1 and WP1--WP11 analyses. The
repository contains the code, configurations, published result tables and
figure source data. Solver-heavy reruns require the public source datasets,
which are described in `../../data/README.md`.

## Fast verification

From the project root with Python 3.10 and the v1.3 core requirements installed:

```powershell
python -m pytest -q code\tests
python code\run_postreview_pipeline.py --help
python code\run_postreview_wp11.py --help
```

The result directories under `../../results/` contain the published numerical
outputs used by the manuscript. Their manifests and the files under
`../../data/provenance/` provide the audit trail.

Run the unit tests from the project root with:

## Rebuild figures from source data

```powershell
python figure_source\fig2\make_figure2.py
python figure_source\fig3\make_figure3.py
python figure_source\fig4\make_figure4.py
python figure_source\fig5\make_figure5.py
python figure_source\fig6\make_figure6.py
python figure_source\supplementary\make_supplementary_figures.py
```

## Rerun the solver-heavy analyses

After preparing the input pairs under `data/processed/`, run the stages in this
order:

```powershell
python code\run_postreview_pipeline.py
python code\run_postreview_wp11.py --run
python code\run_uot_independent_validation.py
```

## Container

Build from the project root:

```powershell
docker build -f code\Dockerfile.v1.3 -t valid-ot:release .
docker run --rm valid-ot:release
```

The container smoke test does not contain the raw public datasets. Mount or
prepare the documented data directories before a solver rerun.
