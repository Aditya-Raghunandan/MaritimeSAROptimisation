# env/ — how to get a working environment

Dependencies are specified **once**, in `pyproject.toml` at the repository root. There is
no `requirements.txt`: two files drift apart, and the pins below have to stay identical on
a laptop and on the cluster or a result cannot be reproduced.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`-e` installs `sar` in editable mode, so `scripts/` and `tests/` can import it.

## Cluster (Jaguar) — measured 2026-09-14, not inherited from a guide

**Everything below was checked on the machine.** Several widely-repeated facts about this
cluster, including ones in Ken's access email and in earlier versions of this file, are
not true of `jaguar1` as it stands.

| Claim you will hear | What the machine actually says |
|---|---|
| "Ubuntu 20.04 ships Python 3.8, bring your own interpreter" | `python3` is **3.12.3**. A plain `venv` is enough; no micromamba, no conda. |
| "`/home` is 1.7 GB and shared, install elsewhere" | **No quota.** 2.5 GB written at 330 MB/s with no error. `/home` sits on the 1.8 T root filesystem. |
| "Raw data goes in `/data1/26p67/raw/`" | `/data` and `/data1` are **empty, root-owned and not writable**. Nothing is mounted on them. `/archive` is `root:root 700`, also denied. |

So, until someone with root creates and chowns `/data1/26p67`:

```
/home/26p67/data/raw/        raw NetCDF
/home/26p67/data/derived/    normalised Parquet and Zarr
```

Pass `--out /home/26p67/data` to every fetch script. **They default to `data` relative to
the working directory**, which on a laptop quietly writes raw NetCDF into a synced folder.

`341 G` was free at the time of writing, on a root filesystem that is 82 % full and shared
with every user of the cluster. Five years of ERA5 is roughly 3 GB, so this is not tight —
but filling `/` breaks the machine for everyone, so check `df -h /` before anything large.

## Where data goes — never here

This repository holds **code, and nothing a script can regenerate**. `.gitignore` enforces
it: `*.nc`, `*.csv`, `*.parquet`, `*.zarr/` and `data/` are all blocked.

That is not tidiness. One drifter pull is **108.8 MB** — GitHub hard-rejects any single
file over 100 MB, and a smaller one that slips through stays in history forever, even
after it is deleted. The rule has to hold before the first `git add`, not after.

## Two things that will cost you an afternoon

Both measured on 2026-09-10, neither discoverable from a documentation page.

**1. ARCO-ERA5 is Zarr format 2**, despite `.zarr-v3` in its URL. With `zarr` 3.x you must
pass `consolidated=True`:

```python
ds = xr.open_zarr(url, storage_options={"token": "anon"}, consolidated=True)
```

Without it, `open_zarr` **never returns** — no error, no output, no traceback. It hangs.

**2. Its variables use full CF names**, not the CDS/GRIB short codes:

```
10m_u_component_of_wind      not  10u
10m_v_component_of_wind      not  10v
mean_sea_level_pressure      not  msl
```

`ds["10u"]` raises `KeyError`. And watch for `10m_u_component_of_neutral_wind`, a
*different quantity* sitting immediately adjacent alphabetically.

## Access and lockouts

The account is shared and the head node bans an IP for **a week after four failed
logins**, so install an SSH key once and never type the password again:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub jaguar1     # appends. ssh-copy-id or >>, NEVER >
```

`/home` is NFS-shared across every node, so one key covers all of them — and a single `>`
on `authorized_keys` overwrites the other person's key and locks them out. The cluster is
also **unreachable from campus eduroam**, which produces a timeout indistinguishable from
a wrong password. Check reachability before concluding anything about credentials.

## Reproducibility

`D008` is explicit that an HPC result which cannot be rerun is not evidence. Pinned
versions, seeded runs, hyperparameters logged next to outputs. If a cluster result ever
needs exact reproduction, generate a lock **there** with `pip freeze` and commit it as
`env/requirements-lock-linux.txt`, labelled with the platform it came from — a freeze is
not a multi-platform resolve and will not install anywhere else.
