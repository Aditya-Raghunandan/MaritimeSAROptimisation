"""
find_current_gaps.py : pull every NaN record out of a current_<period> file produced by fetch_current_range.py, and write them (plus a summary) to a separate text file for gap analysis and mitigation decisions.

HYCOM land cells (and any other masked value) decode to NaN in water_u and water_v. This does not distinguish "land" from any other kind of gap; that judgement, and what to do about it (drop, interpolate, flag, ...), belongs to whoever reads the output, not this script.

Run:
    python scripts/find_current_gaps.py data/current/current_2019-01-01_2019-01-03.txt

Writes <out>/gaps_<input stem>.txt (default <out> = same directory as the input file) containing a summary (record counts, affected locations and timesteps) followed by every NaN row.
"""

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["time", "lat", "lon", "water_u", "water_v"]


def read_records(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, parse_dates=["time"])
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".txt":
        return pd.read_csv(path, sep=r"\s+", parse_dates=["time"])
    raise ValueError(f"unrecognised input format: {suffix}")


def find_gaps(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["water_u"].isna() | df["water_v"].isna()]


def write_report(df: pd.DataFrame, gaps: pd.DataFrame, input_path: Path, out_path: Path) -> None:
    total = len(df)
    n_gaps = len(gaps)
    frac = (n_gaps / total * 100) if total else 0.0
    locations = gaps[["lat", "lon"]].drop_duplicates()
    per_timestep = gaps.groupby("time").size()

    with open(out_path, "w") as f:
        f.write(f"gap report for {input_path}\n")
        f.write(f"total records      : {total}\n")
        f.write(f"NaN records        : {n_gaps} ({frac:.2f}%)\n")
        f.write(f"distinct locations : {len(locations)}\n")
        f.write(f"distinct timesteps : {gaps['time'].nunique()} of {df['time'].nunique()}\n")
        f.write("\n--- NaN records per timestep ---\n")
        f.write(per_timestep.to_string() if len(per_timestep) else "(none)")
        f.write("\n\n--- every NaN record ---\n")
        f.write(gaps.to_string(index=False) if n_gaps else "(none)")
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Pull NaN records out of a fetched current file")
    p.add_argument("input", type=Path, help="a current_<period>.txt/.csv/.parquet file")
    p.add_argument("--out", type=Path, default=None,
                    help="output directory (default: same directory as input)")
    args = p.parse_args()

    df = read_records(args.input)
    gaps = find_gaps(df)
    print(f"{len(gaps)} of {len(df)} records are NaN ({len(gaps) / len(df) * 100:.2f}%)")

    out_dir = args.out or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gaps_{args.input.stem}.txt"

    write_report(df, gaps, args.input, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
