"""
drifters.py -- pull NOAA Global Drifter Program tracks for the study box and
window, with the quality fields and a per-buoy metadata table (issue #55).

Writes two CSVs under <out>/raw/ per product:

    gdp_<product>_<box>_<start>-<end>.csv         one row per fix
    gdp_<product>_buoys_<box>_<start>-<end>.csv   one row per buoy

    python -m sar.fetch.drifters --product hourly   --start 2019-01-01 --end 2024-01-01 --out C:/maritime-data
    python -m sar.fetch.drifters --product 6-hourly --start 2022-11-01 --end 2024-01-01 --out C:/maritime-data

`--out` is REQUIRED. It used to default to `data` relative to the working
directory, which is how raw CSVs end up inside OneDrive (CLAUDE.md rule 4).
Dates are start inclusive, end exclusive, like every other fetcher here.

TWO PRODUCTS, MEASURED 2026-09-23 against the server, not its docs page:

  drifter_hourly_qc  -> 2022-10-31   Elipot et al. 2022, v2.01, fitted by a model
  drifter_6hour_qc   -> 2025-06-18   Lumpkin & Centurioni 2019, kriged

The hourly product says it is updated quarterly and has not moved past
2022-10-31 since at least August 2023. The 6-hourly one was updated March 2026,
and it is the ONLY drifter truth for Nov 2022 - Dec 2023: 141 segments >= 48 h
from 94 buoys, 60 of them not in the hourly set. So both are pulled.

WHY THE QUALITY FIELDS (they were skipped on 10 Sep; 8 of ~50 columns taken)
------------------------------------------------------------------------------
- `gap`: seconds between the real fixes either side of an hourly estimate. The
  hourly product fills across raw-fix gaps up to 12 h without leaving a hole in
  the series, so 1.49 % of points (in 168 of 640 segments >= 48 h) are the
  model's guess, not an observation. Hourly only; the 6-hourly product has no
  such field.
- Per buoy: `location_type` (265 GPS, 3 Argos in the box; Argos is far less
  accurate), `DrogueCenterDepth` (52 buoys were deployed with NO drogue),
  `DrogueDetectSensor`, `typebuoy`, `typedeath`, deploy and end dates. These are
  one value per buoy, so they go in their own small file rather than being
  repeated on 900,000 rows. `location_type` is not served by the 6-hourly
  product at all.
- `err_lat` / `err_lon` are NOT pulled. They are labelled a 95 % interval and
  measured a median under 1 cm, which no GPS drifter position is. Unusable as
  served.

WHY UNDROGUED SEGMENTS MATTER, AND WHY DROGUED ONES DO TOO (D018)
-----------------------------------------------------------------
A drogued buoy hangs a sail at 15 m to follow the water and ignore the wind. An
undrogued one floats at the surface and feels about 1-1.5 % of the wind (Pazan
& Niiler 2001; Lumpkin et al. 2013), against 1.9-2.7 % for a person (Allen
2000). The leeway term should improve the fit for undrogued buoys and not for
drogued ones; the drogued set is the control that tells modelling wind apart
from absorbing current error.
"""

import argparse
import urllib.parse
from pathlib import Path

import pandas as pd

ERDDAP = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap"

# D014 study box. MUST be identical to the wind and current box -- a track
# outside it has no forcing field, so the comparison is meaningless there.
LAT_S, LAT_N = 17.0, 36.0
LON_W, LON_E = -82.0, -63.0

# Real column names, read off each dataset's info endpoint on 2026-09-23. Do not
# guess these: the two products do not offer the same fields.
PRODUCTS = {
    "hourly": {
        "dataset": "drifter_hourly_qc",
        "tag": "hourly",
        "sst_units": "Kelvin",
        "tracks": ["ID", "time", "latitude", "longitude", "ve", "vn", "sst",
                   "drogue_lost_date", "gap"],
        "buoys": ["ID", "location_type", "typebuoy", "DrogueCenterDepth",
                  "DrogueDetectSensor", "typedeath", "deploy_date", "end_date",
                  "drogue_lost_date"],
    },
    "6-hourly": {
        "dataset": "drifter_6hour_qc",
        "tag": "6hour",
        # NOT Kelvin, unlike the hourly product. Both read off the server's `sst`
        # attributes on 2026-09-23; assuming Kelvin masked every 2023 value.
        "sst_units": "degree_C",
        "tracks": ["ID", "time", "latitude", "longitude", "ve", "vn", "sst",
                   "drogue_lost_date"],
        "buoys": ["ID", "typebuoy", "DrogueCenterDepth", "DrogueDetectSensor",
                  "typedeath", "deploy_date", "end_date", "drogue_lost_date"],
    },
}

TRACK_DATES = ["time", "drogue_lost_date"]
BUOY_DATES = ["deploy_date", "end_date", "drogue_lost_date"]


def coverage_url(dataset: str, start: str, end: str, cols: list[str],
                 distinct: bool = False) -> str:
    """ERDDAP tabledap query for the study box, `start` inclusive, `end` exclusive.

    `distinct` asks the server for unique rows only, which is how the per-buoy
    table comes back as one row per buoy instead of one per fix.
    """
    parts = [
        f"time>={start}T00:00:00Z", f"time<{end}T00:00:00Z",
        f"latitude>={LAT_S}", f"latitude<={LAT_N}",
        f"longitude>={LON_W}", f"longitude<={LON_E}",
    ]
    if distinct:
        parts.append("distinct()")
    q = ",".join(cols) + "&" + "&".join(parts)
    return f"{ERDDAP}/{dataset}.csv?{urllib.parse.quote(q, safe=',&=:.-()')}"


def output_paths(out: str | Path, product: str, start: str, end: str) -> tuple[Path, Path]:
    """Where a product's track and buoy CSVs go, named after what they hold."""
    spec = product_spec(product)
    tag = f"{start.replace('-', '')}-{end.replace('-', '')}"
    box = f"{int(LAT_S)}-{int(LAT_N)}N_{abs(int(LON_W))}-{abs(int(LON_E))}W"
    raw = Path(out) / "raw"
    return (raw / f"gdp_{spec['tag']}_{box}_{tag}.csv",
            raw / f"gdp_{spec['tag']}_buoys_{box}_{tag}.csv")


def product_spec(product: str) -> dict:
    """The dataset and columns for a product name, or a clear error."""
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; expected one of {sorted(PRODUCTS)}")
    return PRODUCTS[product]


def read_erddap_csv(url: str, dates: list[str]) -> pd.DataFrame:
    """Read an ERDDAP CSV, dropping its units row, which would poison the dtypes."""
    df = pd.read_csv(url, skiprows=[1])
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def fetch(product: str, start: str, end: str, out: str | Path) -> tuple[Path, Path]:
    """Pull one product's tracks and buoy table, write both, and return the paths."""
    spec = product_spec(product)
    tracks_path, buoys_path = output_paths(out, product, start, end)
    tracks_path.parent.mkdir(parents=True, exist_ok=True)

    url = coverage_url(spec["dataset"], start, end, spec["tracks"])
    print("GET", url)
    tracks = read_erddap_csv(url, TRACK_DATES)
    tracks.to_csv(tracks_path, index=False)
    print(f"wrote {tracks_path} ({tracks_path.stat().st_size / 1e6:.1f} MB)")

    url = coverage_url(spec["dataset"], start, end, spec["buoys"], distinct=True)
    print("GET", url)
    buoys = read_erddap_csv(url, BUOY_DATES)
    buoys.to_csv(buoys_path, index=False)
    print(f"wrote {buoys_path} ({len(buoys)} buoys)")

    report(tracks, buoys, sst_units=spec["sst_units"])
    return tracks_path, buoys_path


def report(df: pd.DataFrame, buoys: pd.DataFrame | None = None, sst_units: str = "Kelvin") -> None:
    """The numbers that decide whether R2 is achievable, printed from what arrived."""
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    print(f"\nrows (fixes)      : {len(df):,}")
    print(f"distinct buoys    : {df.ID.nunique()}")
    print(f"actual time span  : {df.time.min()} -> {df.time.max()}")
    print("  ^ compare with what you asked for. The hourly product ends 2022-10-31.")

    if "sst" in df.columns:
        # With unmasked fill values (hourly: observed -76.9 to 1000 C). The loader
        # converts and masks; this only says how much of it is poison.
        sst_c = df["sst"] - 273.15 if sst_units == "Kelvin" else df["sst"]
        plausible = sst_c.between(-2, 40)
        print(f"sst plausible     : {plausible.mean():.1%} in [-2, 40] C; the rest are fill values")

    if "gap" in df.columns:
        gap_h = df["gap"] / 3600.0
        print(f"fix gap > 3 h     : {(gap_h > 3).mean():.3%} of points are interpolated "
              f"across more than 3 h between real fixes")

    if buoys is not None and len(buoys):
        if "location_type" in buoys.columns:
            print(f"tracking          : {buoys.location_type.value_counts(dropna=False).to_dict()}")
        depth = buoys["DrogueCenterDepth"].astype(str).str.strip()
        print(f"drogue depth      : {depth.value_counts(dropna=False).to_dict()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--product", choices=sorted(PRODUCTS), default="hourly")
    p.add_argument("--start", required=True, help="UTC date, YYYY-MM-DD, inclusive")
    p.add_argument("--end", required=True, help="UTC date, YYYY-MM-DD, exclusive")
    p.add_argument("--out", required=True,
                   help="archive root, e.g. C:/maritime-data; writes <out>/raw/. No default.")
    args = p.parse_args()
    fetch(args.product, args.start, args.end, args.out)


if __name__ == "__main__":
    main()
