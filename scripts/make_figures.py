"""
make_figures.py -- render the project's figures, and print the numbers they argue.

    python scripts/make_figures.py --data C:/maritime-data --figures figures/report
    python scripts/make_figures.py --data /home/26p67/data --figures figures/report

Writes, and for each one prints the measured numbers that `figures/FIGURES.md`
rows are written from:

    wind_rose_study_period.png          the box is an easterly regime, and how steady it is
    speed_histogram_boxmean.png         the bulk state is light -- and see the next one
    speed_histogram_percell.png         what the box mean hides
    seasonal_cycle_study_period.png     whether season alone can stratify the split

Every figure gets a row in `figures/FIGURES.md` saying what it ARGUES, per that
file's own rule: "a figure that cannot be given an argument in one sentence
probably does not belong in the paper."

THE TWO HISTOGRAMS ARE ONE FIGURE MAKING ONE POINT, and it is why this script
draws both. A box mean over 19 deg x 19 deg is not a wind speed -- measured
2026-09-16, the box-mean speed never once reaches 10 m/s across the whole five
years, while over a single ordinary week 15.5 % of individual cells exceed it
and the per-cell maximum is 3.13x the box-mean maximum. Reporting only the first
would understate the wind that actually drives leeway by a factor of three.

Closes nothing on its own. Feeds L1 (the ERA5 high-wind bias exposure), L13, and
the "what is a weather scenario" question that blocks the held-out split.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from sar.utils.data_io import open_box_means, open_forcing
from sar.viz.fields import seasonal_cycle, speed_histogram, wind_rose

# Yearly box-mean files are ~385 KB; the 10 Sep test slices are under 15 KB and
# overlap 2021, which is 312 duplicate hours. Select on size rather than making
# someone tidy the directory first.
YEARLY_MIN_BYTES = 300_000


def yearly_box_means(data: Path) -> list[Path]:
    found = sorted((data / "derived").glob("wind_box_mean_*.parquet"))
    yearly = [f for f in found if f.stat().st_size >= YEARLY_MIN_BYTES]
    if not yearly:
        raise FileNotFoundError(
            f"no yearly box-mean files in {data / 'derived'} "
            f"(found {len(found)} file(s), all below {YEARLY_MIN_BYTES:,} bytes -- "
            "those are the 10 Sep test slices, not the yearly archive)"
        )
    return yearly


def per_cell_speeds(data: Path, limit: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-cell speeds AND the matching box-mean series, over identical hours.

    Both come from the same gridded files on purpose. Comparing the per-cell
    distribution against the box-mean Parquet would compare thirteen days of one
    with five years of the other, and the ratio that falls out of that is not a
    measurement of anything -- it mixes the smoothing effect with the difference
    between two sampling periods. Computing the box mean here, from the same
    array, makes the comparison controlled: identical hours, identical cells,
    the only difference being whether the spatial average is taken first.

    The box mean is the magnitude of the mean VECTOR, matching how
    `sar.fetch.wind` builds `speed` in the Parquet -- not the mean of the
    magnitudes, which is a different and larger quantity.

    One year is 595 MB and ~208 million cell-hours, so files are read one at a
    time. `limit` caps them for a quick local run.
    """
    files = sorted((data / "raw").glob("era5_*.nc"))
    if not files:
        raise FileNotFoundError(f"no era5_*.nc in {data / 'raw'}")
    files = files[:limit] if limit else files

    cell_chunks, box_chunks, used = [], [], []
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # pre-D020 files warn; expected here
            ds = open_forcing(f)
        u, v = ds.u10.values, ds.v10.values
        cell_chunks.append(np.hypot(u, v).ravel())
        box_chunks.append(np.hypot(u.mean(axis=(1, 2)), v.mean(axis=(1, 2))))
        used.append(f.name)
        ds.close()
    return np.concatenate(cell_chunks), np.concatenate(box_chunks), used


def main() -> None:
    p = argparse.ArgumentParser(description="Render the project's figures and print their numbers.")
    p.add_argument("--data", required=True, help="C:/maritime-data or /home/26p67/data")
    p.add_argument("--figures", default="figures/report")
    p.add_argument("--max-years", type=int, default=None,
                   help="cap gridded files read for the per-cell histogram")
    p.add_argument("--json", help="also write every measured number here")
    args = p.parse_args()

    data, fig_dir = Path(args.data), Path(args.figures)
    numbers: dict[str, dict] = {}

    df = open_box_means(yearly_box_means(data))
    span = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"

    numbers["wind_rose"] = wind_rose(
        df.dir_from_deg, df.speed,
        path=fig_dir / "wind_rose_study_period.png",
        title=f"ERA5 10 m wind direction (from), box mean\n{span}")

    numbers["speed_histogram_boxmean"] = speed_histogram(
        df.speed, path=fig_dir / "speed_histogram_boxmean.png", ylabel="hours",
        title=f"Wind speed: BOX MEAN over 17-36 N, 82-63 W\n{span}")

    numbers["seasonal_cycle"] = seasonal_cycle(
        df.index, df.speed,
        path=fig_dir / "seasonal_cycle_study_period.png",
        title=f"Seasonal cycle of box-mean wind speed\n{span}")

    cell_speeds, box_speeds, used = per_cell_speeds(data, args.max_years)
    hours = len(box_speeds)
    numbers["speed_histogram_percell"] = speed_histogram(
        cell_speeds, path=fig_dir / "speed_histogram_percell.png", ylabel="cell-hours",
        title=(f"Wind speed: PER CELL, every 0.25 deg cell\n"
               f"{hours} hours, {len(cell_speeds):,} cell-hours"))
    numbers["speed_histogram_percell"]["files"] = used
    numbers["speed_histogram_percell"]["hours"] = hours

    # The comparison IS the argument, so it is computed rather than left to the
    # reader to do in their head from two separate figures -- and it is computed
    # on the SAME hours, from `box_speeds`, not against the five-year Parquet.
    cell = numbers["speed_histogram_percell"]
    numbers["boxmean_vs_percell"] = {
        "hours_compared": hours,
        "files": used,
        "boxmean_max_ms": float(box_speeds.max()),
        "percell_max_ms": cell["max_ms"],
        "max_ratio": cell["max_ms"] / float(box_speeds.max()),
        "boxmean_mean_ms": float(box_speeds.mean()),
        "percell_mean_ms": cell["mean_ms"],
        "mean_ratio": cell["mean_ms"] / float(box_speeds.mean()),
        "boxmean_fraction_above_10ms": float((box_speeds > 10.0).mean()),
        "percell_fraction_above_10ms": cell["fraction_above_threshold"],
        "claim": ("over identical hours, a box mean across 19 deg x 19 deg understates both "
                  "the peak wind and the fraction of the field in the leeway-relevant band, "
                  "because opposing flows cancel in the vector average"),
    }

    for name, nums in numbers.items():
        print(f"\n{name}")
        for k, v in nums.items():
            print(f"  {k:28s} {v:.4g}" if isinstance(v, float) else f"  {k:28s} {v}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(numbers, indent=2), encoding="utf-8")
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
