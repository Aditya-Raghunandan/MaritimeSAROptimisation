"""produce_maps.py (v3): scripts/produce_maps.py's maps for v3 models, into figures/produced_maps_v3.

A v3 model sees exactly v1's inputs, so it flies through v1's code unchanged; this only adds the
v3 lines to the MODEL list and points --out at figures/produced_maps_v3. Same flags, folders,
names and layout as scripts/produce_maps.py, whose opening comment lists them all.

    PYTHONPATH=src python scripts/v3/produce_maps.py --runs-dir /home/26p67/data/derived/rl/runs_v3 \
        --windows /home/26p67/data/derived/rl/windows_arr30-60-120-180
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The v1 script by its own path: this file shares its name, so a plain import would load itself.
_spec = importlib.util.spec_from_file_location("produce_maps_v1", Path(__file__).resolve().parents[1] / "produce_maps.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OUT = "figures/produced_maps_v3"
_rows_v1 = base.model_rows


def model_rows(run, model):
    return [("version", "v3"), ("reward", "found / square's POS"), *_rows_v1(run, model)]


base.model_rows = model_rows


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--out" not in argv:
        argv += ["--out", OUT]
    base._cli(argv)


if __name__ == "__main__":
    _cli()
