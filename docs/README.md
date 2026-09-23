# docs

Two kinds of document live here, and they are numbered differently on purpose.

## Architecture decision records — `ADRnnn.md`

A record of a decision: the question, the options, the decision, the evidence for it, and what it
commits the codebase to. Numbered **sequentially in this repository**, starting at 001, and never
renumbered once merged — a number that has been cited somewhere is a permanent address.

| | Decision | Status |
|---|---|---|
| [ADR001](ADR001.md) | Dependency management tooling — `pip` and `venv` for Python, `npm` for the frontend | Decided, amended (the original `uv` choice is withdrawn; the cluster has no `uv`) |
| [ADR002](ADR002.md) | Probability map interface — three grids not one; box anchored at arrival; cell size a required parameter; binning at N = 10⁶; coverage swept on the particles; cos φ recomputed per particle per step | **Proposed** — six rows await agreement from the search half |

**These numbers are not the vault's D-numbers.** The project vault keeps its own decision series,
`decisions/Dnnn`, covering decisions that are not about code — the case study region, the study
period, the data products, the cluster. The two series are independent and an ADR states in its
header which vault decision, if any, it corresponds to. ADR002 is the engineering form of vault
D007; ADR001 has no vault counterpart.

## Module documentation — named after what it documents

Not decision records. These describe what a module does, how to run it, and how to read its output.
They may cite an ADR or a vault decision for *why*, and should not restate the reasoning.

| | Covers |
|---|---|
| [data_pipeline_architecture.md](data_pipeline_architecture.md) | the acquisition, storage, publication and visualisation system as built — start here |
| [current-fetch.md](current-fetch.md) | `sar.fetch.current` — HYCOM surface current retrieval (issue #4) |
| [viz-export.md](viz-export.md) | `sar.viz` — the published archive the website reads |
| [resultant-vector.md](resultant-vector.md) | `sar.model.interpolate` — current and wind at any position and time (issue #10) |
| [probability-grid.md](probability-grid.md) | `sar.model.grid` — particle positions to cells, and back (issue #7) |
| [linear_time_interpolation.md](linear_time_interpolation.md) | `sar.utils.interpolation` — why the drift model needs it |
| [e2e-testing.md](e2e-testing.md) | `npm run test:e2e` and `npm run shots` — browser tests, and looking at the page (issue #21) |
| [issue-template.md](issue-template.md) | the structure every new issue follows, and what "done" means |

## Adding one

An ADR takes the next free number and follows ADR001's shape: `Question`, `Decision`,
`Consequences`, with the evidence written where the claim is made rather than summarised. Module
documentation takes the module's name. If a document explains *why* a choice was made, it is an ADR;
if it explains *how* something works, it is module documentation.
