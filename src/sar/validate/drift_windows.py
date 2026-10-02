"""drift_windows.py: dev drifter tracks cut into windows a forecast can be scored on (#89).

A WINDOW is a stretch of one dev unit's track. The model starts at the buoy's fix at t0,
and both are followed for `hours`. The sigma calibration (#89) needs this, and so will
the hindcast scorer (#52), so it lives here once.

THE RULES, each for a reason
  * Dev units only. A sealed or holdout-2023 unit raises SealedUnitError, and there is no
    override: D025 keeps those tracks out of anything that tunes the engine.
  * One drogue tier per window. Tier-uncertain units are left out, because D018 compares
    the two tiers and an uncertain unit belongs to neither.
  * Windows start at 00:00 UTC, then every `hours` within the unit, and never overlap.
    The forcing is sampled at one time per call (docs/gridded-forcing.md, Cost), so only
    windows that start together can share a run of the engine. Aligned starts give about
    1,400 runs instead of about 13,000, and lose at most 23 h per unit.
  * The start fix must be within 3 h of a real GPS fix. Otherwise the model would start
    from a position the hourly product interpolated.
  * Truth at each later hour counts only where fix_gap_h <= 3, D025's rule, so the
    calibration and the sealed test agree on what "where the buoy was" means.

    python -m sar.validate.drift_windows --data C:/maritime-data --out <stem>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sar.fetch.drifters import output_paths
from sar.utils.data_io import load_drifters
from sar.validate.split import HOURLY_WINDOW

DEV = "dev"
TIERS = ("drogued", "undrogued")
WINDOW_HOURS = 48
# D025: a position counts as truth only within 3 h of a real GPS fix.
MAX_FIX_GAP_H = 3.0

_HOUR = np.timedelta64(1, "h")


class SealedUnitError(ValueError):
    """A sealed or holdout-2023 unit was handed to code that tunes the engine (D025)."""


@dataclass
class Windows:
    """The windows, one table row each, and the buoy's position at every hour of each."""

    table: pd.DataFrame              # window, unit, ID, group, tier, t0, lat0, lon0, ve0, vn0
    lat: np.ndarray                  # (W, hours + 1) degrees, NaN where the buoy has no fix
    lon: np.ndarray                  # (W, hours + 1) degrees east, 0 to 360
    ok: np.ndarray                   # (W, hours + 1) True where that position counts as truth
    hours: int
    dropped: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table)

    def batches(self) -> list[np.ndarray]:
        """Row indices grouped by start time, in time order: one engine run each."""
        t0 = self.table["t0"].to_numpy("datetime64[ns]")
        return [np.flatnonzero(t0 == t) for t in np.unique(t0)]

    def subset(self, rows) -> Windows:
        rows = np.asarray(rows)
        return Windows(self.table.iloc[rows].reset_index(drop=True), self.lat[rows],
                       self.lon[rows], self.ok[rows], self.hours, dict(self.dropped))

    def save(self, stem) -> None:
        """<stem>.parquet for the table and <stem>.npz for the hourly positions."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(stem.with_suffix(".parquet"), index=False)
        np.savez_compressed(stem.with_suffix(".npz"), lat=self.lat, lon=self.lon, ok=self.ok,
                            window=self.table["window"].to_numpy(str), hours=self.hours,
                            dropped=json.dumps(self.dropped))

    @classmethod
    def load(cls, stem) -> Windows:
        stem = Path(stem)
        table = pd.read_parquet(stem.with_suffix(".parquet"))
        with np.load(stem.with_suffix(".npz")) as z:
            if not np.array_equal(z["window"], table["window"].to_numpy(str)):
                raise ValueError(f"{stem}: the table and the positions are out of step")
            return cls(table, z["lat"], z["lon"], z["ok"], int(z["hours"]),
                       json.loads(str(z["dropped"])))


def read_split(path) -> pd.DataFrame:
    """validation_split.csv with its times as UTC timestamps."""
    split = pd.read_csv(path, dtype={"ID": str})
    for col in ("start", "end"):
        split[col] = pd.to_datetime(split[col], utc=True)
    return split


def check_dev(units: pd.DataFrame) -> None:
    """Refuse anything that is not dev. No flag turns this off (D025)."""
    bad = units.loc[units["split"] != DEV, "unit"]
    if len(bad):
        raise SealedUnitError(f"{len(bad)} unit(s) are not dev, e.g. {bad.iloc[0]}: the "
                              "calibration must never see sealed or holdout-2023 tracks")


def dev_units(split: pd.DataFrame, tiers=TIERS) -> pd.DataFrame:
    """The dev units of the given tiers."""
    return split[(split["split"] == DEV) & split["tier"].isin(tiers)].reset_index(drop=True)


def first_midnight(t: pd.Timestamp) -> pd.Timestamp:
    """The first 00:00 UTC at or after t."""
    day = t.normalize()
    return day if day == t else day + pd.Timedelta(days=1)


def _naive_ns(times) -> np.ndarray:
    return pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")


def make_windows(units: pd.DataFrame, tracks: pd.DataFrame,
                 hours: int = WINDOW_HOURS) -> Windows:
    """Cut each unit into aligned, non-overlapping windows of `hours`."""
    if hours <= 0:
        raise ValueError(f"hours must be positive, got {hours}")
    check_dev(units)
    if tracks["fix_gap_h"].isna().all():
        raise ValueError("the tracks carry no fix_gap_h, so no position can count as truth")

    by_id = {k: g for k, g in tracks.groupby("ID", sort=False)}
    rows, lats, lons, oks = [], [], [], []
    dropped = {"start_fix_missing_or_interpolated": 0}
    offsets = np.arange(hours + 1) * _HOUR

    for u in units.itertuples(index=False):
        fixes = by_id.get(str(u.ID))
        if fixes is None:
            continue
        fixes = fixes[(fixes["time"] >= u.start) & (fixes["time"] <= u.end)]
        index = pd.Index(_naive_ns(fixes["time"]))
        lat, lon = fixes["lat"].to_numpy(float), fixes["lon"].to_numpy(float)
        gap = fixes["fix_gap_h"].to_numpy(float)
        ve, vn = fixes["ve"].to_numpy(float), fixes["vn"].to_numpy(float)

        t0 = first_midnight(u.start)
        k = 0
        while t0 + pd.Timedelta(hours=hours) <= u.end:
            at = index.get_indexer(_naive_ns([t0])[0] + offsets)
            if at[0] < 0 or not gap[at[0]] <= MAX_FIX_GAP_H:
                dropped["start_fix_missing_or_interpolated"] += 1
            else:
                have = at >= 0
                pick = np.where(have, at, 0)
                lats.append(np.where(have, lat[pick], np.nan))
                lons.append(np.where(have, lon[pick], np.nan))
                oks.append(have & (gap[pick] <= MAX_FIX_GAP_H))
                rows.append({"window": f"{u.unit}#{k}", "unit": u.unit, "ID": str(u.ID),
                             "group": int(u.group), "tier": u.tier,
                             "t0": _naive_ns([t0])[0], "month": t0.strftime("%Y-%m"),
                             "lat0": lat[at[0]], "lon0": lon[at[0]],
                             "ve0": ve[at[0]], "vn0": vn[at[0]]})
            t0 += pd.Timedelta(hours=hours)
            k += 1

    table = pd.DataFrame(rows, columns=["window", "unit", "ID", "group", "tier", "t0", "month",
                                        "lat0", "lon0", "ve0", "vn0"])
    shape = (0, hours + 1)
    return Windows(table,
                   np.array(lats) if lats else np.empty(shape),
                   np.array(lons) if lons else np.empty(shape),
                   np.array(oks) if oks else np.empty(shape, bool),
                   hours, dropped)


def load_tracks(data) -> pd.DataFrame:
    """The hourly GDP product with its buoy table: the only product dev units come from."""
    tracks, buoys = output_paths(data, "hourly", *HOURLY_WINDOW)
    return load_drifters(tracks, product="hourly", buoys=buoys)


def build(data, split_path=None, hours: int = WINDOW_HOURS, tiers=TIERS) -> Windows:
    split = read_split(split_path or Path(data) / "derived" / "validation_split.csv")
    return make_windows(dev_units(split, tiers), load_tracks(data), hours)


def summary(w: Windows) -> dict:
    t = w.table
    return {"windows": len(w),
            "by_tier": {tier: {"windows": int((t["tier"] == tier).sum()),
                               "groups": int(t.loc[t["tier"] == tier, "group"].nunique()),
                               "units": int(t.loc[t["tier"] == tier, "unit"].nunique())}
                        for tier in sorted(t["tier"].unique())},
            "start_days": int(t["t0"].nunique()),
            "truth_hours_usable": float(w.ok[:, 1:].mean()) if len(w) else None,
            "dropped": w.dropped, "hours": w.hours}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data", required=True, help="archive root holding raw/ and derived/")
    p.add_argument("--split", help="validation_split.csv, default <data>/derived/")
    p.add_argument("--hours", type=int, default=WINDOW_HOURS)
    p.add_argument("--out", required=True, help="output stem: writes <out>.parquet and .npz")
    args = p.parse_args(argv)
    w = build(args.data, args.split, args.hours)
    w.save(args.out)
    out = summary(w)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
