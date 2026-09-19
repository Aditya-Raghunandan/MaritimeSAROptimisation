# Maritime Search and Rescue: Drift Prediction

A software simulation of an individual adrift at sea, used as a test bed for comparing
traditional search patterns against modern, data-driven alternatives.

Environmental forcing (HYCOM ocean currents and ERA5 wind fields) and observed drift
(NOAA GDP drifter records) are used to inform a leeway and drift model. Monte Carlo
sampling is employed to propagate positional uncertainty forward in time. The backend
performs the modelling; the frontend visualises the resulting probability map.

## Structure

```
src/sar/            Importable Python package; imported as `sar.fetch`, etc.
  fetch/            HYCOM, ERA5 and GDP drifter retrieval
  model/            Drift equation, leeway parameterisation and Monte Carlo sampling
  pipeline/         Normalisation of raw data to Parquet and Zarr
  utils/            geo (coordinate conventions), data_io (the read half),
                    interpolation, shutdown
  viz/              fields (matplotlib renderers), export (one window as a flat
                    bundle), archive (the whole archive as multi-resolution Zarr),
                    drifters (trajectories as Parquet)
frontend/           JavaScript application, with its own package.json
scripts/            Helper utilities, Slurm job scripts and experiments; not imported
  experiments/
tests/              Pytest suite, organised to mirror src/sar
data/               raw/ and processed/ subdirectories; tracked, though their contents
                    are not (see below)
figures/
  report/           Tracked; deliverable figures, indexed in FIGURES.md
  reference/        Not tracked; reference imagery not owned by the project
docs/               In-repository documentation (distinct from the decision vault; see
                    below)
.github/workflows/  Continuous integration configuration
```

## Where the data is

Nothing large is in this repository, by construction rather than by discipline.

| | Lives at |
|---|---|
| Raw archive | the cluster, `/home/26p67/data/raw/` — regenerable, never committed |
| Published copy | `huggingface.co/datasets/AdityaRugs/MaritimeSARoperations` |
| The site | GitHub Pages, code only |

The published copy is a **derived, downsampled** copy for the website; the raw
NetCDF on the cluster remains what analysis reads. Schemas and conventions are in
[`docs/viz-export.md`](docs/viz-export.md); the architecture that produced them, and why
each part of it is shaped the way it is, is in
[`docs/data_pipeline_architecture.md`](docs/data_pipeline_architecture.md).

## Documentation

| | |
|---|---|
| [**`docs/data_pipeline_architecture.md`**](docs/data_pipeline_architecture.md) | **Start here.** The whole system as an architecture — diagrams, data flows, storage tiers, the client, the Slurm topology, and the measurement or failure behind every design choice |
| [`docs/current-fetch.md`](docs/current-fetch.md) | HYCOM retrieval, the D020 raw tier, and the OPeNDAP request-size ceiling |
| [`docs/viz-export.md`](docs/viz-export.md) | what the website reads: Zarr layout, Parquet trajectories, conventions |
| [`docs/linear_time_interpolation.md`](docs/linear_time_interpolation.md) | time interpolation and the `ForcingProvider` interface |
| [`docs/issue-template.md`](docs/issue-template.md) | the Context / Scope / Acceptance criteria shape issues take here |
| [`docs/ADR001.md`](docs/ADR001.md) | architecture decision record |

Decisions, concepts and the limitations register live in the separate vault,
which is not version controlled and not public.

## Setup

**Backend (Python, via `venv` and `pip`)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # add ".[dev,publish]" to upload to Hugging Face
```

This installs the `sar` package in editable mode, such that `scripts/`, `tests/` and, in
due course, the frontend's API layer may import it directly. Dependency versions are
specified in `pyproject.toml`. The rationale for this approach, including why `uv` was
considered and set aside, is set out in
[`decisions/D0XX-dependency-management-tooling.md`](decisions/D0XX-dependency-management-tooling.md).

**Frontend (JavaScript, via npm)**

```bash
cd frontend
npm install
npm run dev
```

**Cluster environment:** the same procedure applies over SSH, using whichever `python3`
is available on the cluster. No installation with elevated privileges is required.
Refer to `env/README.md` for two further cluster-specific considerations: the
shared-account lockout policy and the eduroam access restriction.

## Data

No data is committed to this repository. Raw retrievals and derived Parquet and Zarr
outputs reside on the cluster (`/data1/26p67/raw/`) and on local disk, and are never
committed to version control. GitHub rejects any file exceeding 100 MB, and smaller
files that are committed in error remain in the repository history after deletion; this
policy is therefore enforced through `.gitignore` rather than convention alone. Material
that may be committed includes source code, environment and dependency configuration,
and small deliverable figures within `figures/report/`.

## Tests

```bash
pytest
```

The test suite is organised to mirror `src/sar/`: `tests/fetch`, `tests/model` and
`tests/pipeline`.

## Continuous integration

`.github/workflows/tests.yml` runs the backend and frontend test suites on every push
and pull request directed at `main`. It has been established in advance of the test
suite itself, and tolerates an absence of test files for the time being; both
placeholder conditions noted within the workflow file should be removed once tests have
been written.

## Decisions

Design decisions (study period, drift model formulation, dependency tooling, and so
forth) are recorded in the shared vault rather than in this repository. Refer to the
vault's decision series for the complete record.
