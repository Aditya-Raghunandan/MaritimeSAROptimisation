# env/ — how to get a working environment

Two files, and the distinction matters:

| File | Use it |
|---|---|
| **`requirements.txt`** | **always.** Direct dependencies, pinned, resolves on Windows *and* Linux |
| `requirements-lock-win.txt` | only on Windows, only when you need byte-identical reproduction |

The lock is a **freeze, not a multi-platform resolve** — some of those versions have no
matching wheel on Ubuntu 20.04. Installing it on the cluster will fail or quietly pull a
different build. Read the header before using it.

## Local (Windows)

```bash
uv venv .venv --python 3.13
uv pip install -r env/requirements.txt
```

Plain `python -m venv .venv && pip install -r env/requirements.txt` works too; `uv` is just
much faster.

## Cluster (Jaguar)

**Python 3.11+ is required and Ubuntu 20.04 ships 3.8**, which is end-of-life and cannot
install these versions at all. So bring your own interpreter — and point it away from
`/home` *before* the first install, because `/home` is **1.7 GB and shared between both of
us**, while this tree is ~1 GB:

```bash
export MAMBA_ROOT_PREFIX=/data1/26p67/mamba
export PIP_CACHE_DIR=/data1/26p67/pipcache
```

Getting that wrong does not produce a clean error. It produces a full filesystem, for
everyone, on every node.

## Where data goes — never here

This repository holds **code, and nothing that a script can regenerate**. Raw data lives on
the cluster (`/data1/26p67/raw/`) and locally outside any synced folder
(`C:\maritime-data`). `.gitignore` enforces it: `*.nc`, `*.csv`, `*.parquet`, `*.zarr/` and
`data/` are all blocked.

That is not tidiness. One drifter pull is **108.8 MB** — GitHub hard-rejects any single file
over 100 MB, and a smaller one that slips through stays in history forever, even after it is
deleted. The rule has to hold before the first `git add`, not after.

## Two things that will cost you an afternoon

Both measured on 2026-09-10, neither discoverable from a documentation page.

**1. ARCO-ERA5 is Zarr format 2**, despite `.zarr-v3` in its URL. With `zarr` 3.x you must
pass `consolidated=True`:

```python
ds = xr.open_zarr(url, storage_options={"token": "anon"}, consolidated=True)
```

Without it, `open_zarr` **never returns** — no error, no output, no traceback. It just hangs.

**2. Its variables use full CF names**, not the CDS/GRIB short codes:

```
10m_u_component_of_wind      not  10u
10m_v_component_of_wind      not  10v
mean_sea_level_pressure      not  msl
```

`ds["10u"]` raises `KeyError`. And watch for `10m_u_component_of_neutral_wind`, which is a
*different quantity* sitting immediately adjacent alphabetically.

## Reproducibility

`D008` is explicit that an HPC result which cannot be rerun is not evidence. So: pinned
versions, seeded runs, hyperparameters logged next to outputs. If a cluster result ever
needs exact reproduction, generate a lock **there** and commit it as
`requirements-lock-linux.txt`.
