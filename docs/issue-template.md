# Issue template and standards

Use this as the starting structure for every new GitHub issue on this repository, and as
the checklist for what "done" means before an issue is closed. `#4 (Fetch current data
(HYCOM))` and its resulting code (`src/sar/fetch/current.py`, `docs/current-fetch.md`)
are a worked example of everything below in practice.

## Issue body structure

Every issue should be written with these three headings, in this order.

### Context

Why this work is needed and what decision or constraint it comes from. If it depends on
a decision recorded in the vault (a `Dxxx` file), name it and summarise the relevant
part rather than assuming the reader has it open. If it corrects a previous wrong
assumption, say so explicitly, since that is often the most important sentence in the
issue. IMPORTANT: Corrections must be added to vault AND handoffs.

### Scope

The specific file(s), function(s) or script(s) to be added or changed, and their
signatures where that is already decided (arguments in, what is returned). Scope should
be narrow enough that it maps to one pull request. If the work naturally splits into
independent pieces, open separate issues rather than growing one issue to cover all of
them.

### Acceptance criteria

A checklist of concrete, verifiable conditions, phrased so that each one either clearly
passes or clearly fails; avoid criteria that are really opinions in disguise. Every
acceptance criterion should be traceable to at least one assertion in the test suite
once the issue is closed: if a criterion cannot be tested, that is usually a sign it
needs to be rewritten as something that can be.

## Definition of done

An issue is not complete when the code runs once on your own machine. Before closing an
issue or merging its pull request, all of the following must be true.

- **Unit tests exist and pass.** Every issue that adds or changes code must come with
  matching unit tests, added or updated in the same pull request; not as a follow-up
  issue. See "Test and code granularity" below for what these should look like.
- **Documentation exists or is updated.** See "Documentation for pulled data" below for
  the specific rule that applies to any issue that fetches or produces data.
- **Every source file is runnable from the terminal on its own.** Any file added under
  `src/sar/` or `scripts/` should be directly runnable and produce a visible result,
  either via an `if __name__ == "__main__":` block with a small `argparse` interface
  (see `src/sar/fetch/current.py`), or, for a `scripts/` file, by running
  `python scripts/<name>.py [args]` directly. A module that only exposes functions for
  other code to import, with no way to invoke it and see output, is not finished; add a
  minimal CLI entry point even if the "real" caller will eventually be other code.
- **Nothing forbidden has been staged.** See "What must never be pushed" below.
- **The vault is updated where relevant**, a
  settled question becomes a decision file, a new script gets a line in `code/MAP.md`,
  a discovered limitation goes in the limitations register, and so on. This is separate
  from, and in addition to, the repository-side documentation described here.

## Test and code granularity

Write several small, single-purpose functions rather than one large function that does
everything. Each function shuold do one conceptually distinct thing: open a resource,
convert a unit, look up a single record, pull a batch of records, write a report. This
is not a style preference for its own sake; it is what makes the next rule possible.

Unit tests should mirror that same granularity, not sit one level above it:

- One group of tests per function, not one large end-to-end test standing in for all of
  them. Structuring tests as one class per function under test (as in
  `tests/fetch/test_current.py`) makes it obvious which function a failing test belongs
  to.
- Cover the ordinary case, at least one edge or off-grid case, and every explicit guard
  the function raises (an `assert`, a `raise`, an out-of-range input). A guard with no
  test exercising it is a guard nobody has verified actually fires.
- Tests must not depend on a live network call or an external service by default. Where
  the code under test talks to a real API or data store, build a small synthetic
  substitute shaped like the real thing (matching field names, dimensions and
  conventions) and inject it via monkeypatching, as `tests/fetch/test_current.py` does
  for the HYCOM dataset. This keeps the suite fast, deterministic, and runnable without
  internet access or a live account, including in continuous integration.
- A test's name should say what behaviour it is checking, not just repeat the function
  name (`test_empty_time_range_raises`, not `test_fetch_current_box`).

## What must never be pushed

- **Any raw or derived data file.** `*.nc`, `*.csv`, `*.parquet`, `*.zarr/`, and
  anything else written under `data/` is gitignored and must stay that way. A single
  drifter pull has already reached 108.8 MB; GitHub hard-rejects any file over 100 MB
  outright, and a smaller one that slips through remains in the repository's history
  even after it is deleted, so this has to be caught before the first `git add`, not
  after.
- **The Python virtual environment.** `.venv/` (or equivalent) is large, wholly
  reproducible from `pyproject.toml`, and platform-specific; it must never be committed.
- **Credentials of any kind.** Passwords, API tokens, SSH keys. These belong out of band
  or in an untracked local file, never in a commit, an issue, or a code comment.
- **Anything under `figures/reference/`.** These are reference images the project does
  not own the rights to redistribute; only figures under `figures/report/` (the
  project's own deliverable figures) are tracked.

Before staging a broad `git add`, run `git status` and read the list rather than
trusting it by habit, particularly after a session that pulled new data.

## Documentation for pulled data

Any issue whose result is a script or function that pulls data from an external source
(an API, an OPeNDAP or Zarr endpoint, a file download) must be accompanied by a
documentation file under `docs/` covering two things, modelled on
`docs/current-fetch.md`:

- **A user guide.** What each field returned actually means (units, sign convention,
  valid range), what inputs are accepted and how out-of-range or off-grid inputs are
  handled (silently snapped to the nearest available value, rejected, or something
  else), and the resolution or granularity of what is returned, in real-world units
  (metres, hours) as well as whatever units the source uses internally.
- **Background on where the data comes from.** Whether it is a direct sensor reading, a
  model output, a reanalysis product, or something assimilated from multiple sources,
  and what that implies about its accuracy and limitations. A reader deciding whether to
  trust a number should not have to go and find this out for themselves.

## Additional conventions worth following

- **Name things after what they produce, not how they are called.** A script's
  docstring should open by saying what files it writes, in one or two lines, before
  explaining how to run it.
- **State the inclusive/exclusive convention for any date range explicitly**, and follow
  the existing convention in the codebase (start inclusive, end exclusive) unless there
  is a specific reason not to.
- **Measure claims against the live server or dataset rather than trusting its
  documentation page**, and record what was actually measured, with the date, rather
  than what was assumed. Several real bugs on this project (barotropic vs. surface
  velocity, ERA5 variable names, HYCOM's real coverage window) came from trusting a
  documentation page instead of querying the source directly.
- **Flag a size or cost warning wherever a script's output could plausibly grow large**
  (a wide date range, a fine grid, a request that multiplies with parameters), including
  a real measured example, the way `docs/current-fetch.md` does for the current fetch.
- **Reference the GitHub issue an implementation closes**, in the module or script
  docstring, so the connection between code and its originating decision is easy to
  trace later.
