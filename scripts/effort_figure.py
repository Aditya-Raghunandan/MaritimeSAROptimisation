"""effort_figure.py: is the search limited by how it is flown, or by how much sea one helicopter can sweep?

Every searcher's POS falls as the cloud grows (r = -0.99, 12 Oct), so a plot of the agent's POS
against cloud size cannot tell a weak policy from an impossible task. The ceiling can
(`scripts/ceiling.py`): the teleport ceiling is the most any searcher could clear on that cloud
with that helicopter's sweep, so what a searcher leaves unfound splits in two:

  * 1 - ceiling: out of reach of one 45-minute sortie. No searcher, learned or doctrinal, gets it.
  * ceiling - POS: lost to how this searcher flew. The only part a better policy could win.

THE X AXIS is the coverage factor C = Z / A (IAMSAR's effort over area): Z = 23.15 km², one
window's sweep, over A, the cloud's 90 % area at arrival. C > 1: the helicopter can sweep more
than the cloud; C < 1: it cannot, whatever it does. One axis for every arrival time and water.

THE CEILING READ GENEROUSLY. The headline uses the in-sample teleport ceiling, the higher of the
two (cells chosen and counted on the same particles), which makes "how it was flown" as large as
it can be: a conservative test of "effort, not the policy". The split ceiling is the band's lower
edge. Neither knows that a helicopter flies a connected path and turns, so both overstate what
was possible.

PANEL B, MORE EFFORT: the same ceiling with twice and four times the sweep (`teleport_x2`,
`teleport_x4`): a longer endurance, a second aircraft, or drones, counted as swept area only.

    python scripts/effort_figure.py --windows CEILING/windows.csv --out figures/report/NAME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

Z_KM2 = 23.15
BINS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0, 40.0])
MIN_WINDOWS = 15
SEED, RESAMPLES = 20261022, 1000
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = {"ppo": ("Telemachus S", "#2a78d6"), "parallel": ("Parallel Track", "#eb6834")}


def binned(df: pd.DataFrame, column: str, rng) -> pd.DataFrame:
    """Mean per coverage bin, its 95 % interval resampling shared-water groups within the bin."""
    out = []
    for k, d in df.groupby("bin", observed=True):
        if len(d) < MIN_WINDOWS:
            continue
        by = d.groupby("group")[column].mean()
        boot = [by.sample(len(by), replace=True, random_state=rng).mean() for _ in range(RESAMPLES)]
        out.append({"c": float(np.exp(np.log(d["coverage"]).mean())), "mean": float(d[column].mean()),
                    "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                    "windows": len(d)})
    return pd.DataFrame(out)


def numbers(df: pd.DataFrame) -> dict:
    """The decomposition by arrival time, means over windows: what the paper quotes."""
    out = {}
    for h, d in df.groupby("arrival_h"):
        line = {"windows": len(d), "coverage_median": float(d["coverage"].median()),
                "ceiling": float(d["teleport"].mean()), "ceiling_split": float(d["teleport_split"].mean()),
                "out_of_reach": float(1 - d["teleport"].mean())}
        for s in [c for c in ("ppo", "telemachus", "parallel", "expanding-square", "greedy") if c in d]:
            line[f"{s}_pos"] = float(d[s].mean())
            line[f"{s}_flown_loss"] = float((d["teleport"] - d[s]).mean())
            line[f"{s}_of_ceiling"] = float((d[s] / d["teleport"]).mean())
        for k in ("x2", "x4"):
            if f"teleport_{k}" in d:
                line[f"ceiling_{k}"] = float(d[f"teleport_{k}"].mean())
        out[str(h)] = line
    return out


def draw(df: pd.DataFrame, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.patches import Patch

    rng = np.random.default_rng(SEED)
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.8,
                         "legend.frameon": False, "legend.fontsize": 8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True, constrained_layout=True)
    ticks = [0.25, 0.5, 1, 2, 4, 10]
    arrivals = df.groupby("arrival_h")["coverage"].median()
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Where each arrival time sits on the axis: the median window's coverage factor.
        top = ax.secondary_xaxis("top")
        top.set_xticks(list(arrivals.values),
                       [f"{h * 60:g} min" if h < 1 else f"{h:g} h" for h in arrivals.index])
        top.minorticks_off()
        top.tick_params(length=3, labelsize=7, colors=INK_2)
        top.spines["top"].set_visible(False)
    fig.supxlabel("coverage factor: one sortie's sweep (23.15 km²) ÷ the cloud's 90 % area\n"
                  "← bigger clouds than one helicopter can sweep                    "
                  "smaller clouds →", fontsize=8, color=INK_2)
    a.set_ylabel("probability cleared (POS)")
    a.set_ylim(0, 1.02)

    ceil = binned(df, "teleport", rng)
    split = binned(df, "teleport_split", rng)
    a.fill_between(ceil["c"], ceil["mean"], 1.0, color=GRID, alpha=0.8, linewidth=0)
    a.fill_between(ceil["c"], split["mean"], ceil["mean"], color=MUTED, alpha=0.3, linewidth=0)
    a.plot(ceil["c"], ceil["mean"], color=INK, linewidth=2)
    drawn = {"ceiling": ceil.to_dict("records"), "ceiling_split": split.to_dict("records")}
    handles = [Patch(color=GRID, label="out of reach of one 45-min sortie"),
               plt.Line2D([], [], color=INK, linewidth=2, label="ceiling: the most any searcher could clear")]
    for key, (name, colour) in SERIES.items():
        if key not in df:
            continue
        s = binned(df, key, rng)
        a.fill_between(s["c"], s["lo"], s["hi"], color=colour, alpha=0.15, linewidth=0)
        line, = a.plot(s["c"], s["mean"], color=colour, linewidth=2, marker="o", markersize=3.5,
                       markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        handles.append(line)
        drawn[key] = s.to_dict("records")
    a.legend(handles=handles, loc="lower right", handlelength=1.6)
    a.set_title("A  One helicopter, 45 minutes", loc="left", fontsize=9, color=INK, pad=16)

    labels = {"teleport_x4": "ceiling, 4× the sweep", "teleport_x2": "ceiling, 2× the sweep",
              "teleport": "ceiling, one sortie"}
    shades = {"teleport": INK, "teleport_x2": INK_2, "teleport_x4": MUTED}
    for key, name in labels.items():
        if key not in df:
            continue
        c = binned(df, key, rng)
        b.plot(c["c"], c["mean"], color=shades[key], linewidth=2, label=name)
        drawn[key] = c.to_dict("records")
    if "ppo" in drawn:
        s = pd.DataFrame(drawn["ppo"])
        b.plot(s["c"], s["mean"], color=SERIES["ppo"][1], linewidth=1.4, linestyle=(0, (4, 2)),
               label="Telemachus S, one sortie")
    b.legend(loc="lower right", handlelength=1.6)
    b.set_title("B  More sweep on the same clouds", loc="left", fontsize=9, color=INK, pad=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    return drawn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--windows", type=Path, required=True, help="ceiling.py summary's windows.csv")
    p.add_argument("--out", type=Path, required=True, help="figure path without extension")
    args = p.parse_args(argv)
    df = pd.read_csv(args.windows)
    df["coverage"] = Z_KM2 / df["area90_km2"]
    df["bin"] = pd.cut(df["coverage"], BINS)
    drawn = draw(df, args.out)
    record = {"windows": len(df), "groups": int(df["group"].nunique()), "by_arrival": numbers(df),
              "binned": drawn}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps(record["by_arrival"], indent=1))


if __name__ == "__main__":
    main()
