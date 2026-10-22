"""square.py: the Expanding Square's POS on every window, flown once and kept, to scale v3's reward.

One row per windows file and arrival time, flown through the same referee and pattern as the test
scoring (sar.rl.fly.fly_window), so "over 1.0" in training means what it means on the test weeks.
Written to a CSV the first time and read back after, so it is flown once per set of windows.
"""

from __future__ import annotations

import csv
from multiprocessing import Pool
from pathlib import Path

from sar.rl.features import Context
from sar.rl.fly import fly_window
from sar.rl.windows import load
from sar.search.benchmark import BASELINE


def key(name: str, minutes: int) -> str:
    return f"{name}|{minutes}"


def _fly_file(args) -> list[tuple[str, float]]:
    path, arrivals_min = args
    windows = load(path)
    out = []
    for k, arrival in enumerate(windows.arrivals_s):
        minutes = round(float(arrival) / 60.0)
        if minutes not in arrivals_min:
            continue
        setup = windows.setup(k)
        context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
        row = {"lat": windows.lkp[0], "lon": windows.lkp[1], "seed": windows.seed}
        pos = fly_window(setup, context, row, [BASELINE])[BASELINE]["python"]["pos"]
        out.append((key(windows.name, minutes), float(pos)))
    return out


def square_pos(paths, arrivals_min, cache: Path, workers: int = 1) -> dict[str, float]:
    """{"<windows name>|<arrival minutes>": the square's POS}, read from `cache`, flying what is missing."""
    cache = Path(cache)
    table = {}
    if cache.exists():
        with open(cache, newline="") as f:
            table = {r["key"]: float(r["square_pos"]) for r in csv.DictReader(f)}
    todo = [str(p) for p in paths
            if any(key(Path(p).stem, m) not in table for m in arrivals_min)]
    if todo:
        print(f"flying the Expanding Square over {len(todo)} windows files, {workers} at a time", flush=True)
        with Pool(workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_fly_file, [(p, list(arrivals_min)) for p in todo]), 1):
                table.update(rows)
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i} of {len(todo)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "square_pos"])
            w.writerows(sorted(table.items()))
    return table
