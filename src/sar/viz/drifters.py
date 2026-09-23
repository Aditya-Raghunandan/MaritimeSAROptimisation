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


# ---------------------------------------------------------------------------
# The track export the site's drifter layer reads (issue #50).
#
#   drifter_index.json          one row per buoy: small, loaded once
#   drifter_tracks/<ID>.json    one buoy's every fix, fetched when it is needed
#
# SPLIT BY BUOY, NOT ONE FILE, because the site shows one buoy's path at a time
# and a few buoys' positions at once. All of them together are tens of MB; one
# is tens of kB, so a click costs one small fetch and the path keeps its full
# hourly resolution rather than being thinned to fit one download.
# ---------------------------------------------------------------------------

INDEX_NAME = "drifter_index.json"
TRACK_DIR = "drifter_tracks"

# 4 decimal places of a degree is 11 m, far finer than a GDP position is good for
# (~100 m), and it keeps the files small.
COORD_DP = 4

# A fix further than this from a real GPS fix is the fitting method's guess, and
# is drawn dashed (#55: 1.49 % of hourly points).
FAR_FROM_FIX_H = 3.0


def _iso(ts: pd.Timestamp) -> str:
    """UTC ISO 8601 with a Z, which every browser's Date parses the same way."""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _tier_summary(g: pd.DataFrame) -> str:
    """One word for a buoy's drogue record: drogued, undrogued, mixed or uncertain."""
    if g["tier_uncertain"].any():
        return "uncertain"
    und = g["undrogued"]
    if und.all():
        return "undrogued"
    return "mixed" if und.any() else "drogued"


def track_record(g: pd.DataFrame) -> dict:
    """One buoy's fixes as compact columns, in time order.

    `h` is whole hours since `t0` (every fix is on the hour); `seg` renumbers the
    buoy's in-box segments from 0, so the site never draws a line across a gap;
    `und` and `far` are 0/1 per fix. Longitude is the display convention.
    """
    g = g.sort_values("time")
    t0 = g["time"].iloc[0]
    hours = ((g["time"] - t0).dt.total_seconds() / 3600.0).round().astype(int)
    seg = g["segment_id"].rank(method="dense").astype(int) - 1
    far = (g["fix_gap_h"] > FAR_FROM_FIX_H) if "fix_gap_h" in g.columns else pd.Series(False, index=g.index)
    return {
        "id": str(g["ID"].iloc[0]),
        "t0": _iso(t0),
        "h": hours.tolist(),
        "lat": g["lat"].round(COORD_DP).tolist(),
        "lon": np.round(to_display_longitude(g["lon"].to_numpy()), COORD_DP).tolist(),
        "seg": seg.tolist(),
        "und": g["undrogued"].astype(int).tolist(),
        "far": far.fillna(False).astype(int).tolist(),
    }


def buoy_index(df: pd.DataFrame, units: pd.DataFrame | None = None) -> list[dict]:
    """One entry per buoy, sorted by first fix: what the dots, list and search need.

    `splits` lists the validation splits the buoy's units fall in (#51), empty if
    none of its runs is long enough to be a unit. `sealed` is true if any unit is
    sealed or in the 2023 holdout: the raw track may be looked at, but the engine
    must not be compared against it before the frozen evaluation run (D025).
    """
    splits: dict[str, list[str]] = {}
    if units is not None and len(units):
        for bid, s in units.groupby(units["ID"].astype(str))["split"]:
            splits[bid] = sorted(set(s))
    rows = []
    for bid, g in df.groupby(df["ID"].astype(str), sort=False):
        g = g.sort_values("time")
        first = g.iloc[0]
        lost = g.loc[g["undrogued"], "time"]
        s = splits.get(bid, [])
        rows.append({
            "id": bid,
            "start": _iso(first["time"]),
            "end": _iso(g["time"].iloc[-1]),
            "lat0": round(float(first["lat"]), COORD_DP),
            "lon0": round(float(to_display_longitude(np.array([first["lon"]]))[0]), COORD_DP),
            "fixes": int(len(g)),
            "products": sorted(set(g["product"])) if "product" in g.columns else ["hourly"],
            "tier": _tier_summary(g),
            "drogue_lost": _iso(lost.iloc[0]) if len(lost) and not lost.index.equals(g.index) else None,
            "splits": s,
            "sealed": bool({"sealed", "holdout-2023"} & set(s)),
            "file": f"{TRACK_DIR}/{bid}.json",
        })
    return sorted(rows, key=lambda r: (r["start"], r["id"]))


def publish_tracks(df: pd.DataFrame, units: pd.DataFrame | None, out: Path) -> dict:
    """Write the index and every buoy's track file under `out`; return what was written."""
    tracks = out / TRACK_DIR
    tracks.mkdir(parents=True, exist_ok=True)
    index = buoy_index(df, units)
    sizes = []
    for bid, g in df.groupby(df["ID"].astype(str)):
        path = tracks / f"{bid}.json"
        path.write_text(json.dumps(track_record(g), separators=(",", ":")), encoding="utf-8")
        sizes.append(path.stat().st_size)
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": "drifter_tracks",
        "label": "NOAA GDP drifters, one file per buoy",
        "longitude_convention": "-180..180 (display; D020 stores 0-360)",
        "buoys": index,
        "counts": {"buoys": len(index), "fixes": int(len(df)),
                   "sealed_buoys": sum(r["sealed"] for r in index)},
        "provenance": {"script": "sar.viz.drifters --tracks",
                       "split": "sar.validate.split (vault D025)"},
    }
    index_path = out / INDEX_NAME
    index_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    summary = {
        "buoys": len(index),
        "index_bytes": index_path.stat().st_size,
        "track_bytes_total": int(sum(sizes)),
        "track_bytes_median": int(np.median(sizes)) if sizes else 0,
        "track_bytes_max": int(max(sizes)) if sizes else 0,
    }
    print(f"wrote     {INDEX_NAME}  {summary['buoys']} buoys, {summary['index_bytes'] / 1e3:.0f} kB")
    print(f"wrote     {TRACK_DIR}/  {summary['track_bytes_total'] / 1e6:.1f} MB in total; "
          f"median {summary['track_bytes_median'] / 1e3:.0f} kB, "
          f"largest {summary['track_bytes_max'] / 1e3:.0f} kB per buoy")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Publish GDP drifters for the site.")
    p.add_argument("--data", required=True, help="archive root holding raw/gdp_*.csv")
    p.add_argument("--out", required=True, help="where the published files go. No default.")
    p.add_argument("--source", help="which gdp_hourly_*.csv, if the archive holds several")
    p.add_argument("--tracks", action="store_true",
                   help="write the per-buoy track export (#50) instead of the Parquet: "
                        "both products, with the validation split from derived/validation_split.csv")
    args = p.parse_args()
    if args.tracks:
        # Imported here so the Parquet path keeps working on a checkout without it.
        from sar.validate.split import load_study_drifters

        split_csv = Path(args.data) / "derived" / "validation_split.csv"
        if not split_csv.exists():
            raise SystemExit(f"no {split_csv}; run `python -m sar.validate.split --data {args.data}` first")
        publish_tracks(load_study_drifters(args.data), pd.read_csv(split_csv), Path(args.out))
        return
    publish(Path(args.data), Path(args.out),
            Path(args.source) if args.source else None)


if __name__ == "__main__":
    main()
