"""make_e2e_scenarios.py: a one-scenario bundle for the browser tests of the scenario page and game.

The e2e suite runs offline against committed fixtures (frontend/tests/e2e/site.spec.js). This
writes frontend/tests/e2e/fixtures/scenarios/v1/: one scenario, both noise models, the 2 h
arrival only, 200 particles, under constant forcing, by the same exporter that writes the
published bundles (scripts/export_scenario_bundles.py), plus a benchmark.json. About 200 kB.
The numbers are not under test here; the wiring and painting are.

    python scripts/make_e2e_scenarios.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd

from sar.pipeline.forcing import ConstantForcing
from sar.search.benchmark import GREEDY, NOISE, SEARCHERS

OUT = Path("frontend/tests/e2e/fixtures/scenarios/v1")
SCRIPTS = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row() -> pd.Series:
    base = {"scenario": "S01", "start": "2019-06-01T06:00", "lat": 26.5, "lon": -79.0,
            "seed": 5, "set": "train", "stratum": "jet", "group": 1, "speed_4h_ms": 1.1}
    for h in range(1, 7):
        base[f"lat_{h}h"], base[f"lon_{h}h"] = 26.5 + 0.0045 * h, -79.0 + 0.0100 * h
    return pd.Series(base)


def main() -> None:
    bundles = load("export_scenario_bundles")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    forcing = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))
    entry = bundles.export_row(row(), forcing, ["rv", "rw"], [2.0], 200, OUT, GREEDY)
    bundles.write_index(OUT, [entry], {"particles": 200, "noise": NOISE, "greedy": GREEDY,
                                       "arrival_h": [2.0], "searchers": list(SEARCHERS),
                                       "commit": "e2e"})
    for p in OUT.glob(".index.*.json"):
        p.unlink()
    rows = []
    for noise in ("rv", "rw"):
        meta = json.loads((OUT / "S01" / f"{noise}_2h.json").read_text())
        for name, f in meta["flights"].items():
            pos = f["python"]["pos"]
            rows.append({"noise": noise, "searcher": name, "arrival_h": 2.0, "pos": pos,
                         "pos_lo": pos, "pos_hi": pos, "found": float(bool(f["python"]["found"])),
                         "found_lo": 0.0, "found_hi": 1.0, "n_rows": 1, "n_groups": 1})
    (OUT / "benchmark.json").write_text(json.dumps({"format": 1, "scenarios55": rows}, indent=1))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT}: {size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
