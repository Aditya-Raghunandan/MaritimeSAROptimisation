"""
fetch_current_range.py : pulls every HYCOM surface-current record (D014 box, depth level 0) between two dates and save it to data/current/.

Uses sar.fetch.current, so it stays correct if the fetch logic changes.

Run:
    python scripts/fetch_current_range.py --start 2019-01-01 --end 2019-01-08

    --start is inclusive, --end is exclusive, same convention as code/fetch_wind_arco.py in the vault.

Writes data/current/current_<start>-<end>.<ext>
    one row per (time, lat, lon) record, columns time, lat, lon, water_u, water_v.
Defaults to .txt (whitespace-separated) if --format is not given.
"""

import argparse
from pathlib import Path

from sar.fetch.current import LAT_N, LAT_S, LON_E, LON_W, fetch_current_box

FORMATS = {
    "txt": "txt",
    "csv": "csv",
    "parquet": "parquet",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Pull HYCOM surface currents over a date range")
    p.add_argument("--start", required=True, help="UTC date, YYYY-MM-DD, inclusive")
    p.add_argument("--end", required=True, help="UTC date, YYYY-MM-DD, exclusive")
    p.add_argument("--out", default="data/current", help="output directory")
    p.add_argument("--format", choices=FORMATS, default="txt", help="output file type")
    args = p.parse_args()

    print(f"opening HYCOM and selecting {args.start}..{args.end} over the D014 box ...")
    sub = fetch_current_box(args.start, args.end, (LAT_S, LAT_N), (LON_W, LON_E))
    print("  grid:", dict(sub.sizes))

    df = sub.to_dataframe().reset_index()[["time", "lat", "lon", "water_u", "water_v"]]
    print(f"  {len(df)} records")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    period = f"{args.start}_{args.end}"
    ext = FORMATS[args.format]
    out_path = out_dir / f"current_{period}.{ext}"

    if ext == "csv":
        df.to_csv(out_path, index=False)
    elif ext == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        # Timestamps must not contain a raw space: to_string is whitespace-delimited, and "2019-01-01 00:00:00" would silently split into two fields on read-back, swallowing the date into an implicit index.
        out_df = df.copy()
        out_df["time"] = out_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        out_df.to_string(out_path, index=False)

    print("wrote", out_path)


if __name__ == "__main__":
    main()
