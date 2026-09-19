"""
drifters.py -- publish the GDP drifter observations for the site.

    python -m sar.viz.drifters --data /home/26p67/data --out /home/26p67/data/published

The third dataset, and the one that is neither a gridded field nor engine
output. It is the **validation** set: R2's hindcast compares simulated drift
against these real trajectories, and the `track` layer is how that comparison
becomes a picture rather than a table of errors.

PARQUET, NOT ZARR. Zarr is for regular gridded arrays and these are nothing of
the kind -- irregular positions, irregular times, one row per observation per
buoy. Parquet is the format for that, it is what Hugging Face's dataset viewer
reads, and `pyarrow` is already pinned. Forcing a trajectory into a grid would
throw away the thing that makes it a trajectory.

WHAT IS PUBLISHED, AND WHY IT IS NOT THE RAW CSV. `load_drifters` applies every
rule `sar.fetch.drifters.report()` established -- axis renames, longitude to
0-360, SST Kelvin to Celsius masked against fill values, `undrogued` per
OBSERVATION rather than per buoy, and `segment_id` split on gaps over 3 h. The
raw CSV is 103.8 MB of ERDDAP's conventions with a units row wedged under the
header; republishing that would hand everyone else the same four traps.

Longitude is converted to -180..180 here, as in `sar.viz.archive`: this is the
presentation boundary and a web map cannot use 0-360.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sar.utils.data_io import load_drifters
from sar.utils.geo import to_display_longitude

# Columns worth shipping. `sst_c` earns its place because D013's survivable
# window is a function of water temperature, so the same file answers "where
# did it drift" and "how long would someone last there".
COLUMNS = ["ID", "time", "lat", "lon", "segment_id", "undrogued", "sst_c", "ve", "vn"]

# A segment shorter than this cannot test a 48 h drift forecast, which is what
# D018 budgets the validation runs against.
#
# DURATION, NOT OBSERVATION COUNT, and the difference is not pedantry. 48 hourly
# observations span 47 hours -- the usual fencepost -- and a 47 h segment cannot
# verify a 48 h forecast. Measured 2026-09-18 on the full five-year pull:
#
#     segments total                     825
#     duration      >= 48 h              640   <- what R2 can actually use
#     observations  >= 48                642   <- what D018 cites
#
# Two segments have 48 observations across 47 hours. D018's 642 counts
# observations; for its own stated purpose the number is 640.
MIN_SEGMENT_HOURS = 48


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to the publishable columns and convert to display longitude."""
    keep = [c for c in COLUMNS if c in df.columns]
    out = df[keep].copy()
    out["lon"] = to_display_longitude(out["lon"].to_numpy())
    # float32 halves the file and is far finer than the data's own precision:
    # a GDP position is good to ~100 m, which is 1e-3 deg, while float32 holds
    # about 1e-5 deg at these magnitudes.
    for c in ("lat", "lon", "sst_c", "ve", "vn"):
        if c in out:
            out[c] = out[c].astype("float32")
    return out.sort_values(["ID", "time"]).reset_index(drop=True)


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous in-box run: the unit R2 validates against.

    The site draws tracks from this rather than scanning a million rows to work
    out what a track even is, and it is the table that answers "how many
    independent validation cases do we actually have".
    """
    g = df.groupby("segment_id", sort=True)
    out = pd.DataFrame({
        "segment_id": g.size().index,
        "buoy": g["ID"].first().to_numpy(),
        "start": g["time"].min().to_numpy(),
        "end": g["time"].max().to_numpy(),
        "n": g.size().to_numpy(),
        "lat0": g["lat"].first().to_numpy(),
        "lon0": g["lon"].first().to_numpy(),
        # Per observation, so a buoy that loses its drogue mid-segment is
        # partly undrogued rather than being forced into one camp.
        "undrogued_frac": g["undrogued"].mean().to_numpy(),
    })
    hours = (out["end"] - out["start"]) / np.timedelta64(1, "h")
    out["hours"] = hours.astype("float32")
    return out.sort_values("start").reset_index(drop=True)


def publish(data: Path, out: Path, source: Path | None = None) -> dict:
    if source is None:
        files = sorted((data / "raw").glob("gdp_hourly_*.csv"))
        if not files:
            raise FileNotFoundError(f"no gdp_hourly_*.csv in {data / 'raw'}")
        if len(files) > 1:
            # Refuse rather than guess. The first version took files[-1], which
            # is "last alphabetically" -- against the three local pulls that
            # silently selected the SMALLEST (a 3-month slice) over the full
            # five-year file, and published it without a word. Picking the
            # largest would work today and break the day a bigger partial pull
            # appears, so the caller says which.
            listing = "\n  ".join(
                f"{f.name}  ({f.stat().st_size / 1e6:.1f} MB)" for f in files)
            raise SystemExit(
                f"{len(files)} drifter CSVs in {data / 'raw'} -- pass --source to "
                f"say which:\n  {listing}"
            )
        source = files[0]

    print(f"reading   {source.name} ({source.stat().st_size / 1e6:.1f} MB)")
    obs = prepare(load_drifters(source))
    segs = segment_summary(obs)
    long_enough = segs[segs["hours"] >= MIN_SEGMENT_HOURS]

    out.mkdir(parents=True, exist_ok=True)
    obs_path = out / "drifters.parquet"
    seg_path = out / "drifter_segments.parquet"
    obs.to_parquet(obs_path, index=False, compression="zstd")
    segs.to_parquet(seg_path, index=False, compression="zstd")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": "drifters",
        "label": "NOAA GDP drifters (hourly, quality-controlled)",
        "longitude_convention": "-180..180 (display; D020 stores 0-360)",
        "files": {
            "observations": {"path": obs_path.name, "rows": int(len(obs)),
                             "bytes": obs_path.stat().st_size},
            "segments": {"path": seg_path.name, "rows": int(len(segs)),
                         "bytes": seg_path.stat().st_size},
        },
        "counts": {
            "observations": int(len(obs)),
            "buoys": int(obs["ID"].nunique()),
            "segments": int(len(segs)),
            f"segments_over_{MIN_SEGMENT_HOURS}h_duration": int(len(long_enough)),
            "undrogued_fraction": float(obs["undrogued"].mean()),
        },
        "period": {"start": str(obs["time"].min()), "end": str(obs["time"].max())},
        "bbox": [float(obs["lat"].min()), float(obs["lon"].min()),
                 float(obs["lat"].max()), float(obs["lon"].max())],
        "provenance": {"source_file": source.name, "script": "sar.viz.drifters"},
    }
    (out / "drifters_archive.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    c = manifest["counts"]
    print(f"wrote     {obs_path.name}  {len(obs):,} obs, "
          f"{obs_path.stat().st_size / 1e6:.1f} MB "
          f"(from {source.stat().st_size / 1e6:.0f} MB of CSV)")
    print(f"wrote     {seg_path.name}  {len(segs):,} segments, "
          f"{seg_path.stat().st_size / 1e3:.0f} kB")
    print(f"buoys     {c['buoys']}")
    print(f"undrogued {c['undrogued_fraction'] * 100:.2f} %")
    print(f"segments  {len(segs):,} total, {len(long_enough):,} lasting at least "
          f"{MIN_SEGMENT_HOURS} h (duration, not observation count)")
    print(f"period    {manifest['period']['start'][:10]} -> {manifest['period']['end'][:10]}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Publish GDP drifters as Parquet for the site.")
    p.add_argument("--data", required=True, help="archive root holding raw/gdp_hourly_*.csv")
    p.add_argument("--out", required=True, help="where the published files go. No default.")
    p.add_argument("--source", help="which gdp_hourly_*.csv, if the archive holds several")
    args = p.parse_args()
    publish(Path(args.data), Path(args.out),
            Path(args.source) if args.source else None)


if __name__ == "__main__":
    main()
