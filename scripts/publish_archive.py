"""
publish_archive.py -- push a published Zarr archive to a Hugging Face dataset.

    python scripts/publish_archive.py --src /home/26p67/published \
        --repo AdityaRugs/MaritimeSARoperations

This is the last step of D021 step 4: `sar.viz.archive` writes the store,
this puts it where a browser can reach it. GitHub Pages carries the code; the
data lives here, which satisfies the no-data-in-the-repo rule by construction
rather than by discipline.

WHY HUGGING FACE AND NOT CLOUDFLARE R2 (decided 2026-09-17)
-----------------------------------------------------------
R2's free tier is real -- 10 GB-month, 1 M Class A, 10 M Class B, zero egress
-- and our operation count is nowhere near the limits. It was rejected because
it **requires a linked payment card**: Cloudflare's billing policy places
preauthorisation holds on that card for usage-based services, and a failed hold
makes the buckets return errors with deletion after 30 days. A site that has to
stay up through assessment should not carry a billing failure mode.

Hugging Face needs no card. The honest caveat is that its free public storage
is documented as "best-effort" with no guaranteed figure; the wind archive
MEASURED 1.60 GB across three tiers and 2,238 files (Slurm 58640, 2026-09-17),
inside the "first few gigabytes" they describe as unremarkable, and well inside
their <100k files per repo and <10k per folder. It also wants a dataset card, which this writes, and
which the report wants anyway.

THERE IS DELIBERATELY NO --token FLAG
-------------------------------------
A token passed on a command line ends up in shell history, in `ps` output, in
CI logs and in any transcript of the session. R3g says credentials never enter
the vault, and the same reasoning applies everywhere else they get written
down by accident. Authenticate out of band, once:

    huggingface-cli login          # stores it under ~/.cache/huggingface
    # or, for one shell only:
    export HF_TOKEN=...            # never `HF_TOKEN=... python ...`, which is logged

This script reads whichever of those is present and never prints it.
"""

import argparse
import json
import sys
from pathlib import Path

CARD = """---
license: mit
tags:
  - maritime
  - search-and-rescue
  - oceanography
  - era5
  - hycom
---

# Maritime SAR operations -- forcing archive

Gridded wind and surface-current fields over the western North Atlantic,
published as multi-resolution [Zarr](https://zarr.dev) so a web map can scrub
through five years without downloading five years.

Supporting data for an undergraduate investigation project at the University of
the Witwatersrand on **stochastic and machine-learning-driven maritime search
patterns for strong-current environments**. The code that produces it is at
<https://github.com/Aditya-Raghunandan/MaritimeSAROptimisation>.

## What is here

| | |
|---|---|
| Domain | 17-36 N, 82-63 W -- the Gulf Stream and the western Bermuda Triangle |
| Period | 2019-01-01 to 2024-01-01 |
| Wind | ERA5 10 m `u10`, `v10`, 0.25 deg, hourly at source |
| Current | HYCOM `GLBy0.08/expt_93.0` `water_u`, `water_v` at depth level 0, 0.08 deg lon x 0.04 deg lat, 3-hourly at source |

Each product is published at several time resolutions, because nobody can
perceive hourly detail while scrubbing across a year and nobody should download
it: the hourly wind tier is 1.27 GB against 0.07 GB for the daily one.

```
<product>_archive.json     the manifest: grid, bbox, tiers, provenance
<product>_hourly.zarr      full cadence
<product>_6-hourly.zarr    every sixth step
<product>_daily.zarr       every twenty-fourth step
```

Coarser tiers are **strided, not averaged**. A 24 h mean of a rotating wind
vector is close to zero, so an averaged daily tier would show five years of
calm. A stride shows a real hour, just fewer of them.

## Conventions

- **Longitude is -180..180** here. The project stores 0-360 internally and
  converts once, at the presentation boundary, which is this export.
- **Latitude is ascending**, which ERA5 is not at source and HYCOM is.
- Arrays are `(time, lat, lon)`, `float32`, `m/s`.
- Coordinates are regular, so the manifest carries `lat0/dlat/nlat` and the
  time axis as `start` + `step_seconds` + `frames`. A tier whose time axis has
  gaps is marked `"regular": false` and its timestamps must be read from the
  store's own `time` array rather than reconstructed.
- Chunks are **48 timesteps x the whole spatial box**, about 1.11 MB, which is
  the 48 h scenario window at hourly cadence.
- Compression is **zstd level 19**. Blosc compresses no better here and
  `zarrita.js` cannot decode it in a browser.

## Reading it

```python
import xarray as xr
url = "https://huggingface.co/datasets/{repo}/resolve/main/wind_daily.zarr"
ds = xr.open_zarr(url, consolidated=False)
```

```js
import * as zarr from "zarrita";
const store = new zarr.FetchStore(
  "https://huggingface.co/datasets/{repo}/resolve/main/wind_daily.zarr");
const u = await zarr.open(zarr.root(store).resolve("u10"), {{ kind: "array" }});
```

## Provenance and licence

Derived from two public sources, reprojected and downsampled but not otherwise
altered:

- **ERA5** (Hersbach et al., Copernicus Climate Change Service) via
  [ARCO-ERA5](https://github.com/google-research/arco-era5) on Google Cloud.
  Contains modified Copernicus Climate Change Service information; neither the
  European Commission nor ECMWF is responsible for any use of it.
- **HYCOM + NCODA Global 1/12 deg Analysis**, `GLBy0.08/expt_93.0`, via the
  HYCOM THREDDS server.

Code and this derived packaging are MIT. The underlying data carry their
originators' terms.
"""


def summarise(src: Path) -> dict:
    """What is about to be uploaded, before anything is."""
    files = [f for f in src.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    stores = sorted(d.name for d in src.iterdir() if d.is_dir() and d.name.endswith(".zarr"))
    manifests = sorted(f.name for f in src.glob("*_archive.json"))
    return {"files": len(files), "bytes": total, "stores": stores, "manifests": manifests}


def check_manifests(src: Path) -> list[str]:
    """Refuse to publish a tier whose time axis cannot be reconstructed.

    `sar.viz.archive` already refuses to WRITE one without --allow-gaps. This
    is the second gate, because the thing that reaches the browser is what
    matters and the two steps can be run days apart.
    """
    problems = []
    for m in sorted(src.glob("*_archive.json")):
        manifest = json.loads(m.read_text(encoding="utf-8"))
        for name, tier in manifest.get("tiers", {}).items():
            if tier.get("regular") is False:
                problems.append(
                    f"{m.name}: tier {name!r} has an irregular time axis "
                    f"({len(tier.get('gaps') or [])} gap(s)) -- the client cannot "
                    f"reconstruct its timestamps and will refuse it"
                )
            verified = tier.get("verified") or {}
            if verified.get("match") is False:
                problems.append(
                    f"{m.name}: tier {name!r} failed its own round-trip check against "
                    f"the source -- do not publish it"
                )
    return problems


def resolve_token():
    """Whatever the environment already holds. Never a command-line argument."""
    from huggingface_hub import get_token
    return get_token()


def publish(src: Path, repo: str, *, private: bool = False, dry_run: bool = False) -> str:
    # huggingface_hub is imported DOWN THERE, past the guards and past the
    # dry-run return, not here. It is an optional [publish] extra, so on a
    # machine that only installed [dev] -- CI, for one -- importing it at the
    # top of this function makes every guard below unreachable and every one of
    # their tests fail on a ModuleNotFoundError. Which is exactly what happened
    # on the first CI run of this branch, 2026-09-18.
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory -- run sar.viz.archive first")

    problems = check_manifests(src)
    if problems:
        raise SystemExit("refusing to publish:\n  " + "\n  ".join(problems))

    info = summarise(src)
    print(f"source    {src}")
    print(f"stores    {', '.join(info['stores']) or '(none)'}")
    print(f"manifests {', '.join(info['manifests']) or '(none)'}")
    print(f"upload    {info['files']} files, {info['bytes'] / 1e6:.1f} MB")
    if not info["stores"]:
        raise SystemExit("no *.zarr store in that directory -- nothing to publish")

    if dry_run:
        print("\n--dry-run: nothing was uploaded")
        return ""

    token = resolve_token()
    if not token:
        raise SystemExit(
            "no Hugging Face credential found.\n"
            "  Run `huggingface-cli login` once, or set HF_TOKEN in this shell.\n"
            "  This script takes no --token flag on purpose: a token on a command "
            "line ends up in shell history and in logs."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"\nauthenticated as {who}")

    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)

    card = src / "README.md"
    if not card.exists():
        card.write_text(CARD.replace("{repo}", repo), encoding="utf-8")
        print(f"wrote     {card} (Hugging Face requires a dataset card)")

    # upload_large_folder is resumable and chunks its own commits, which matters
    # because a Zarr store is thousands of small files rather than a few big
    # ones, and a failed single-commit upload of 1.67 GB would start over.
    print("uploading ...")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(src))

    url = f"https://huggingface.co/datasets/{repo}"
    print(f"\npublished {url}")
    print(f"VITE_DATA_BASE={url}/resolve/main")
    return url


def main() -> None:
    p = argparse.ArgumentParser(
        description="Push a published Zarr archive to a Hugging Face dataset repo.")
    p.add_argument("--src", required=True, help="directory written by sar.viz.archive")
    p.add_argument("--repo", required=True, help="e.g. AdityaRugs/MaritimeSARoperations")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="say what would be uploaded and stop")
    args = p.parse_args()

    try:
        publish(Path(args.src), args.repo, private=args.private, dry_run=args.dry_run)
    except SystemExit as e:
        print(e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
