# Maritime Search and Rescue: Drift Prediction

A software simulation of an individual adrift at sea, used as a test bed for comparing
traditional search patterns against modern, data-driven alternatives.

Environmental forcing (HYCOM ocean currents and ERA5 wind fields) and observed drift
(NOAA GDP drifter records) are used to inform a leeway and drift model. Monte Carlo
sampling is employed to propagate positional uncertainty forward in time. The backend
performs the modelling; the frontend visualises the resulting probability map.

## Structure

```
src/sar/            Importable Python package; imported as `sar.model`, etc.
  fetch/            HYCOM, ERA5 and GDP drifter retrieval code
  model/            Drift equation, leeway parameterisation and Monte Carlo sampling
  pipeline/         Normalisation of raw data to Parquet and Zarr formats
  utils/
frontend/           JavaScript application, with its own package.json
scripts/            Helper utilities and experiments; not imported and not tested
  experiments/
tests/              Pytest suite, organised to mirror src/sar
data/               raw/ and processed/ subdirectories; tracked, though their contents
                    are not (see below)
figures/
  report/           Tracked; deliverable figures, indexed in FIGURES.md
  reference/        Not tracked; reference imagery not owned by the project
env/                Setup notes for local and cluster environments
docs/               In-repository documentation (distinct from the decision vault; see
                    below)
.github/workflows/  Continuous integration configuration
```

## Setup

**Backend (Python, via `venv` and `pip`)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
