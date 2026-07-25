# Protocol amendment 005: explicit 10% crop and duplicate-motif diagnostics

**Time:** 2026-07-16, before inspection of E4-E6 scientific outcome tables.  
**Reason:** The primary semisynthetic grid instantiated `crop_missing` at 25%, while the frozen plan required both 10% and 25% crop levels. The primary utility table also did not separately expose the registered duplicate-motif identity diagnostics.

The required missing analyses are added without replacing any primary run:

1. Add a 10% crop diagnostic for both technologies, both metadata-selected bases, all ten frozen seeds, and all five required methods at `n=800`.
2. Retain the existing 25% `crop_missing` primary grid and 20% crop inside `combined`; no observed result is overwritten.
3. For every duplicate-motif primary configuration, report strict identity error, dual-solution probability asymmetry, whether the top two targets contain both designed equivalents, entropy AUROC/AUPRC, and top-two-margin AUROC/AUPRC.
4. The entropy endpoint remains the registered ambiguity diagnostic. Top-two margin is reported as the prespecified supporting comparison and cannot replace entropy after outcome inspection.

This amendment corrects coverage of already registered experiments and does not alter selection, thresholds, methods, or conclusions.
