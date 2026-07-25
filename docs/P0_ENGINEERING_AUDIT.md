# VALID-OT P0 engineering audit

## Scope

P0 was limited to project inspection, manuscript compilation, citation/cross-reference repair, author-metadata state, Figure 9 attribution, cost-normalization documentation, automated QA, and pre-reanalysis freezing. No P1–P4 experiment was run or implemented, and no frozen scientific result was changed.

## Project and build entry

- Workspace: the repository root used for the public release.
- Main manuscript: `paper/main.tex`, identified by `documentclass`, `begin{document}`, and active `input` statements for the four files under `paper/sections/`.
- Dormant/template entries: `user_data/main.tex`, `_user_templates/main.tex`, and standalone TikZ sources are not the compiled paper entry.
- Build system before P0: no Makefile or latexmkrc. `latexmk` is installed but cannot run because MiKTeX cannot find Perl. P0 added a minimal Makefile with `manuscript`, `manuscript-check`, `manuscript-clean`, and `freeze` targets.
- Bibliography system: BibTeX (`bibliography{references}` + `bibliographystyle{unsrt}`), not biblatex/biber.
- Reproducible P0 build: `scripts/build_manuscript.py`, which runs `pdflatex → bibtex → pdflatex → pdflatex` under the independent job name `manuscript_p0_clean` and copies outputs to `build/p0/final/`.
- PowerShell launcher: `scripts/run_p0.ps1` locates the frozen Python runtime or accepts `VALIDOT_PYTHON`.

## Baseline findings

The first direct build could not overwrite `paper/main.pdf` because that file was open in another process. P0 therefore used an independent job name and did not force-close or overwrite the user's viewer file.

The successful baseline build had no undefined citation/reference warnings, duplicate-label warnings, missing files, or fatal errors. Nevertheless, rendered citations were empty (`[]`, `[, , ]`). The cause was deterministic: the OUP class v1.5 notes identify explicit `numbered` as superfluous, and the class implementation maps that option to a natbib mode incompatible with the `unsrt` entries used here. Removing only that option restored numeric citations for all 14 cited bibliography keys.

Baseline artifacts are stored under `build/p0/baseline/`, including the PDF, log, command record, warning summary, and pre-edit scientific-artifact hashes.

## Citation audit and repair

- Active citation commands inspected: all `citep` calls in the four active section files.
- Unique cited keys: 14.
- BibTeX entries: 14.
- TeX-only missing keys: 0.
- Duplicate BibTeX keys: 0.
- Bib-only unused entries: 0.
- Citation content changes: none.
- Repair: removed the redundant OUP `numbered` class option; no citation key or bibliography entry was invented or edited.
- Final PDF: numeric citations render correctly; no empty citation brackets remain.

Some existing BibTeX records contain author-side `[VERIFY]` comments in source metadata. P0 did not replace those fields from memory; they remain a later bibliographic-verification task, not an undefined-citation blocker.

## Labels, references, and Figure 9

- Active labels: unique.
- Active refs without labels: 0.
- LaTeX duplicate-label warnings: 0.
- Missing figure/table files: 0.
- `paper/figures/latex_includes.tex` contains duplicated generated figure environments but is not in the active dependency closure and produces no compiled duplicate labels.

Figure 9 is defined inside the main Results source and is referenced by the main text. The active `paper/` source has no separate supplementary compilation entry. P0 therefore applied rule A: retained it as main-text Figure 9 and removed “Supplementary” from the Results sentence, caption heading, and alt text. No data, number, or figure file was changed.

## Author metadata and template fields

No verified author names, affiliations, corresponding email, or author ORCIDs were found. P0 centralized the existing explicit placeholders in `paper/metadata.tex` and documented every required author-supplied field in `docs/P0_UNRESOLVED_METADATA.md`. No anonymous status was assumed and no identity was invented.

`To be assigned during production`, volume 0, issue 0, and the production copyright wording were retained as OUP production/template fields. Funding and CRediT text remain explicit submission placeholders rather than fabricated facts.

## Cost-normalization audit summary

The confirmatory code separately constructs squared-Euclidean CE and CS matrices, divides each by its own positive-entry median, and mixes them as `wE*CE + wS*CS`. Expression inputs are conditionally library/log normalized and row-L2 normalized; coordinates are centered and RMS-radius standardized per slice. The baseline is `(wE,wS)=(0.5,0.5)`, `epsilon=0.25`, and UOT `tau=2.0`. Registered endpoints are `I_EXPR=(0,0.5)` and `I_SPATIAL=(0.5,0)` with epsilon/tau unchanged. Balanced OT, UOT, and row-softmax share the same normalized mixed cost; legacy FGW branches have separate objective-specific weighting and remain excluded from confirmatory Gates.

The complete file/function/formula audit is in `docs/COST_NORMALIZATION_AUDIT.md`. P0 did not renormalize weights, compensate epsilon, add a weight path, or add a data-executing audit mode.

## Frozen scientific-content protection

Nineteen critical scientific artifacts were SHA-256 hashed before manuscript edits. Final QA recomputed all hashes and found zero changes. This covers root `results.json`, the v1.3 frozen config, verification outputs, both copies of figure result JSON, and `user_data/results.json`.

P0 did not edit the abstract, Table 1, Table 2, figure data, Gate thresholds, NEX-AURC, Spearman, overlap, NMAE, independent-unit count, protocol versions, registered/corrected/exploratory hierarchy, verifier expectations, or any result JSON/TSV/CSV.

## Final verification

- Four-step manuscript build: PASS, exit codes `[0,0,0,0]`.
- Undefined citations: 0.
- Undefined references: 0.
- Duplicate active labels: 0.
- Missing active inputs/graphics: 0.
- PDF `??`: 0.
- PDF empty citations: 0.
- Figure 9 attribution: main-text Figure 9, consistent.
- Manuscript QA: 22 PASS, 1 WARN, 0 FAIL. The warning is unresolved author metadata.
- Existing unit tests: 10 passed.
- Frozen v1.3 verifier record: `COMPLETED`, 26 checks, 0 failures. P0 did not re-execute `verify_v13.py` because this copied workspace omits the full `15_v1_3_correction/` result tree expected by that script and the script writes verification outputs; instead, both frozen verifier files were protected by pre/post SHA-256 checks.
- `pdfinfo`: PASS; 8 pages, PDF 1.5, unencrypted, no page rotation.
- `pdftotext`: PASS.
- `qpdf`: not run because qpdf is not installed.
- Visual inspection: first page shows resolved numeric citations; page 7 shows Figure 9 without a Supplementary caption marker.

Non-critical class/layout warnings remain: 7 overfull hboxes, 10 underfull hboxes, 8 overfull vboxes, 5 font warnings, and one balance-package warning. The large hbox warnings arise during OUP output/crop handling and do not visibly cross the page boundary in the rendered inspection. P0 did not alter figure data or scientific text to suppress cosmetic warnings.

## Remaining blockers

1. The corresponding author must provide and confirm the metadata listed in `docs/P0_UNRESOLVED_METADATA.md`.
2. The supplied workspace has no `.git` directory; commit, branch, and dirty-state provenance cannot be reconstructed. SHA-256 freezing is used instead.
3. The standalone v1.2 config is referenced by the copied protocol but is external to this workspace; its path and verified historical SHA-256 are recorded in the freeze.
4. `latexmk` requires Perl and `qpdf` is unavailable. The provided fallback build and PDF checks are reproducible without them.

## Final artifacts

- Clean PDF: `build/p0/final/manuscript_p0_clean.pdf`.
- Final log: `build/p0/final/manuscript_p0_clean.log`.
- Warning extract: `build/p0/final/manuscript_p0_warnings.txt`.
- QA report: `build/p0/final/manuscript_p0_qa.txt`.
- Analysis freeze: `analysis_freeze/p0-pre-reanalysis/`.
