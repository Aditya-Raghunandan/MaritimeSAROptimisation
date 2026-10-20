"""produce_maps.py (v2): scripts/produce_maps.py's maps, flown by v2 models, into figures/produced_maps_v2.

Same flags, folders, names and layout as scripts/produce_maps.py, whose opening comment lists
them all; only two things differ. The model is flown through sar.rl_v2.fly (the 2D observation),
and --out defaults to figures/produced_maps_v2. The MODEL list gains the v2 inputs and reward.

    PYTHONPATH=src python scripts/v2/produce_maps.py --runs-dir /home/26p67/data/derived/rl/runs_v2 \
        --windows /home/26p67/data/derived/rl/windows_arr30-60-120-180

One map:
    PYTHONPATH=src python scripts/v2/produce_maps.py --runs-dir /home/26p67/data/derived/rl/runs_v2 \
        --windows /home/26p67/data/derived/rl/windows_arr30-60-120-180 --models run_baseline_v2 \
        --locations gulf-stream-off-jacksonville --dates 2021-07-27 --arrivals 180 --out /tmp/check
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from sar.rl_v2 import fly as fly_v2

# The v1 script by its own path: this file shares its name, so a plain import would load itself.
_spec = importlib.util.spec_from_file_location("produce_maps_v1", Path(__file__).resolve().parents[1] / "produce_maps.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OUT = "figures/produced_maps_v2"
_rows_v1 = base.model_rows


def model_rows(run, model):
    return [("version", "v2"), ("maps", "2D, CNN"), ("reward", "final POS only"), *_rows_v1(run, model)]


base.fly_window, base.load_policy, base.model_rows = fly_v2.fly_window, fly_v2.load_policy, model_rows


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--out" not in argv:
        argv += ["--out", OUT]
    base._cli(argv)


if __name__ == "__main__":
    _cli()
