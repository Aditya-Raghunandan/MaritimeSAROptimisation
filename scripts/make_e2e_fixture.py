"""make_e2e_fixture.py: builds the tiny published archive the browser suite runs against.

Writes `frontend/tests/e2e/fixtures/` -- two manifests and their Zarr stores, a few
hundred kilobytes in total. The Playwright suite intercepts every request to the Hugging
Face origin and serves these instead, so the browser tests are offline, deterministic and
independent of whether a public server is up. That is the same rule the pytest and vitest
suites already follow; nothing in this repository's test suites touches the network.

NOTHING HERE IS NEW MATHS, AND THAT IS THE POINT. It runs the real pipeline over fake
input:

    scripts/make_demo_forcing.py   invented but correctly shaped NetCDF, land patch and all
      -> sar.viz.archive.export_archive    the SAME function that published the real archive
        -> frontend/tests/e2e/fixtures/    manifests and Zarr stores in the production shape

A fixture written by hand would be a second, unverified opinion about what the published
archive looks like, and the first time the real exporter changed shape the suite would go
on passing against a store the site no longer reads. Generating it through the exporter
means the fixture cannot drift from the thing it stands in for without this script
failing, which is the same argument `export_resultant_golden.py` makes for the resultant.

The values are meaningless -- a smooth invented field over a 1 degree box, plus a NaN
wedge where land would be. The suite asserts that the page WIRES UP and PAINTS, never
what the numbers are. The numbers are the unit suites' job.

Run:
    python scripts/make_e2e_fixture.py
    python scripts/make_e2e_fixture.py --out some/other/dir
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "frontend" / "tests" / "e2e" / "fixtures"

# EVERY tier the real archive publishes, over DAYS days. This used to be the finest tier
# of each over one day, on the reasoning that the suite never changes span. But the page
# changes it itself: past 14 days it opens on a coarser tier and switches to hourly for the
# default one-day view. That switch is what left the drift arrows reading a stale store and
# freezing at hour 49 on the live site (23 Sep), and a one-day, one-tier fixture could not
# reach it. Sixteen days is the shortest span that makes the page open coarse.
TIERS = {"wind": ["hourly", "6-hourly", "daily"], "current": ["3-hourly", "daily"]}
DAYS = 16


def run(args: list[str], what: str) -> None:
    """Run a step, and fail loudly with its own output rather than a return code."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{what} failed with exit code {proc.returncode}")


def demo_drifters() -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Three invented buoys in the demo box, shaped as `load_drifters` returns them.

    Each exercises one thing the drifter layer must get right: A loses its drogue
    part-way, so its trail changes colour; B is sealed and has a stretch far from a
    real fix, so it is badged and partly dashed; C starts on the 15th, outside the
    default one-day window, so only the search can find it; D runs ten days, so
    picking it changes the wind tier as well as the span. The exporter is the real
    one (`sar.viz.drifters.publish_tracks`), for the reason the forcing goes through
    `sar.viz.archive`: a hand-written fixture would be a second opinion of the format.
    """
    import numpy as np
    import pandas as pd

    def buoy(bid, start, hours, lat, lon, seg, lost=None, far=()):
        t = pd.date_range(start, periods=hours + 1, freq="h", tz="UTC")
        gap = np.ones(len(t))
        gap[list(far)] = 5.0
        return pd.DataFrame({
            "ID": bid, "time": t,
            "lat": lat + np.linspace(0.0, 0.3, len(t)),
            "lon": lon + np.linspace(0.0, 0.3, len(t)),      # 0-360, as loaded
            "segment_id": seg,
            "undrogued": (t >= pd.Timestamp(lost, tz="UTC")) if lost else False,
            "tier_uncertain": False, "fix_gap_h": gap, "product": "hourly",
        })

    df = pd.concat([
        buoy("E2E-A", "2021-01-05 06:00", 60, 26.3, 281.2, 0, lost="2021-01-06 06:00"),
        buoy("E2E-B", "2021-01-08 00:00", 48, 26.5, 281.5, 1, far=(20, 21, 22)),
        buoy("E2E-C", "2021-01-15 00:00", 30, 26.6, 281.3, 2),
        # Ten days: long enough that picking it switches the wind to a coarser tier,
        # which is the path that used to drag the clock back to the old window.
        buoy("E2E-D", "2021-01-10 12:00", 240, 26.2, 281.6, 3),
    ], ignore_index=True)
    units = pd.DataFrame({"ID": ["E2E-A", "E2E-B"], "split": ["dev", "sealed"]})
    return df, units


def build(out: Path) -> None:
    if out.exists():
        # Rebuilt from scratch every time. Leaving stale stores behind is how a fixture
        # ends up with files no manifest names and a suite that passes on the wrong data.
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        print(f"1/4  demo forcing -> {raw}")
        run([sys.executable, str(REPO / "scripts" / "make_demo_forcing.py"),
             "--out", str(raw), "--days", str(DAYS)], "make_demo_forcing.py")

        for i, (product, tiers) in enumerate(TIERS.items(), start=2):
            print(f"{i}/4  export {product} ({', '.join(tiers)}) -> {out}")
            tier_args = [arg for tier in tiers for arg in ("--tier", tier)]
            run([sys.executable, "-m", "sar.viz.archive",
                 "--data", str(raw), "--out", str(out),
                 "--product", product, *tier_args], f"sar.viz.archive --product {product}")

    print(f"4/4  drifter tracks -> {out}")
    from sar.viz.drifters import publish_tracks

    drifters, units = demo_drifters()
    publish_tracks(drifters, units, out)

    report(out)


def report(out: Path) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} files, {total / 1024:.0f} kB in {out}")
    if total > 2_000_000:
        # A fixture is committed, so its size is permanent. Two megabytes is already
        # generous for sixteen days over a 1 degree box; past that something has gone
        # wrong -- the run grew, or the demo grid did.
        raise SystemExit(f"fixture is {total / 1e6:.1f} MB, which is too big to commit")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"where the fixture goes (default: {DEFAULT_OUT})")
    args = p.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
