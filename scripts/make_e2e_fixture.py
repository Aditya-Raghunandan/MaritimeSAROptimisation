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

# Only the finest tier of each product. The frontend picks a tier by visible span and the
# suite never changes span, so publishing the coarse tiers would add files nothing reads.
TIERS = {"wind": "hourly", "current": "3-hourly"}


def run(args: list[str], what: str) -> None:
    """Run a step, and fail loudly with its own output rather than a return code."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{what} failed with exit code {proc.returncode}")


def build(out: Path) -> None:
    if out.exists():
        # Rebuilt from scratch every time. Leaving stale stores behind is how a fixture
        # ends up with files no manifest names and a suite that passes on the wrong data.
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        print(f"1/3  demo forcing -> {raw}")
        run([sys.executable, str(REPO / "scripts" / "make_demo_forcing.py"),
             "--out", str(raw)], "make_demo_forcing.py")

        for i, (product, tier) in enumerate(TIERS.items(), start=2):
            print(f"{i}/3  export {product} ({tier} tier) -> {out}")
            run([sys.executable, "-m", "sar.viz.archive",
                 "--data", str(raw), "--out", str(out),
                 "--product", product, "--tier", tier], f"sar.viz.archive --product {product}")

    report(out)


def report(out: Path) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} files, {total / 1024:.0f} kB in {out}")
    if total > 2_000_000:
        # A fixture is committed, so its size is permanent. Two megabytes is already
        # generous for a handful of frames over a 1 degree box; past that something has
        # gone wrong -- a coarse tier crept in, or the demo grid grew.
        raise SystemExit(f"fixture is {total / 1e6:.1f} MB, which is too big to commit")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"where the fixture goes (default: {DEFAULT_OUT})")
    args = p.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
