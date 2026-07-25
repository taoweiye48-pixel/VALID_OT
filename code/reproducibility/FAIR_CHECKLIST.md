# FAIR and release checklist

## Findable

- [x] Stable project title and v1.3 protocol identifier.
- [x] Machine-readable artifact manifest with SHA-256 hashes.
- [ ] Mint a Zenodo DOI after author approval and insert it into the manuscript and metadata.

## Accessible

- [x] Public-source record IDs, checksums, and acquisition metadata retained.
- [x] Data availability text distinguishes redistributable derived files from upstream-controlled raw data.
- [ ] Confirm the upstream licences before redistributing raw archives.

## Interoperable

- [x] Tab-separated result tables, JSON decisions, NPZ response arrays, PNG/PDF figures.
- [x] Column-level semantics described by report and figure notes.
- [ ] Add a formal schema/version field to every future table if the project is extended.

## Reusable

- [x] Frozen core environment, Dockerfile, CLI orchestration, tests, and verification checks.
- [x] Historical v1.2 snapshot and explicit v1.3 correction chain.
- [x] PASTE/PASTE2 exclusion and post-hoc hypothesis boundary documented.
- [ ] Replace author/affiliation placeholders and select an open-source licence before public release.
- [ ] Execute a clean-machine/container reproduction and archive its log before submission.
