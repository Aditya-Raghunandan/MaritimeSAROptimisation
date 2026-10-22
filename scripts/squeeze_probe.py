"""squeeze_probe.py: does believing the buoy is nearer the cloud's middle find more REAL buoys? (13 Oct)

Dev buoys (bench_dev_48h.csv) with their real tracks; rvc clouds rebuilt at each row's seed.
Greedy and Telemachus 2 are flown reading a cloud pulled toward its weighted centre by k
(k = 1 is the cloud as it is); the referee still scores the true cloud (POS) and the real buoy
(found). The patterns are the reference. Nothing is trained: k is a hand-set belief.
    python scripts/squeeze_probe.py PART PARTS OUT   (probe_rows.csv beside OUT; 290 dev rows, 3 a group)

Run on the laptop 13 Oct in 12 parts; results in C:/maritime-data/derived/results/squeeze_probe_dev290.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sar.pipeline.gridded import GriddedForcing
from sar.search.benchmark import GREEDY, NOISE, searcher
from sar.search.episode import STEPS, run
from sar.search.greedy import greedy_policy
from sar.search.platform import STEP_S
from sar.search.scenario import scenario_search, search_episode
from sar.rl.features import Context
from sar.rl.fly import load_halves, load_policy, ppo_policy

D = "C:/maritime-data"
TELE = f"{D}/derived/rl/chosen_tele/best_model.zip"
KS = (1.0, 0.7, 0.5)
ARRIVALS = (0.5, 1.0, 2.0, 3.0)


class Shrunk:
    """The episode as a searcher sees it, every particle pulled toward the cloud's centre by k."""
    def __init__(self, ep, k):
        self._ep, self._k = ep, k
    def __getattr__(self, name):
        return getattr(self._ep, name)
    def particles(self):
        lat, lon = self._ep.particles()
        w = np.asarray(self._ep.window.weight, dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        clat, clon = np.average(lat[ok], weights=w[ok]), np.average(lon[ok], weights=w[ok])
        return clat + self._k * (lat - clat), clon + self._k * (lon - clon)


def shrunk(policy, k):
    return policy if k == 1.0 else (lambda ep: policy(Shrunk(ep, k)))


def main(part, parts, out):
    rows = pd.read_csv(Path(out).parent / "probe_rows.csv")
    rows = rows.iloc[part::parts]
    model, headings = load_policy(TELE)
    halves = load_halves(TELE)
    lines = []
    for _, row in rows.iterrows():
        start = np.datetime64(str(row["start"]), "us")
        try:
            forcing = GriddedForcing.from_dir(D, start, start + np.timedelta64(int(3 * 3600 + STEPS * STEP_S), "s"))
        except Exception as e:
            lines.append({"scenario": row["scenario"], "skipped": str(e)[:200]}); continue
        for h in ARRIVALS:
            try:
                setup = scenario_search(row, forcing, round(h * 3600), 10_000, **NOISE["rvc"])
            except Exception as e:
                lines.append({"scenario": row["scenario"], "arrival_h": h, "skipped": str(e)[:200]}); continue
            ctx = Context((float(row["lat"]), float(row["lon"])), float(setup.arrival_s), setup.drift_bearing_deg)
            flights = {"parallel": searcher("parallel", setup, row, h)[0],
                       "expanding-square": searcher("expanding-square", setup, row, h)[0]}
            for k in KS:
                flights[f"greedy_k{k}"] = shrunk(greedy_policy(**GREEDY), k)
                flights[f"telemachus_k{k}"] = shrunk(ppo_policy(model, headings, ctx, halves), k)
            for name, policy in flights.items():
                m = run(policy, search_episode(setup))
                t = m.get("target") or {}
                lines.append({"scenario": row["scenario"], "group": int(row["group"]), "arrival_h": h,
                              "searcher": name, "pos": m["pos"], "found": t.get("found"),
                              "closest_m": t.get("closest_m")})
        Path(out).write_text("".join(json.dumps(x) + "\n" for x in lines))
    Path(out).write_text("".join(json.dumps(x) + "\n" for x in lines))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
