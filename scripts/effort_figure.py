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

SIX PANELS, one figure for a two-column paper (7.2 in wide):
  A  POS against the coverage factor: the ceiling, what is out of reach, five searchers.
  B  For each arrival, where the probability goes: cleared by Telemachus S, lost to how it flew,
     out of reach; the Parallel Track's two numbers beside it.
  C  More effort on the same clouds, one panel per arrival: the ceiling at 1-4 x the sweep, one
     helicopter with k x the sweep width, and k ordinary helicopters (`scripts/fleet.py`), flown.

Means are by shared-water group first, then across groups (D025); bands are 95 % intervals
resampling groups.

    python scripts/effort_figure.py --windows CEILING/windows.csv --fleet FLEET/fleet.csv --out PATH
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
KS = (1, 2, 3, 4)
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
REACH = "#d9d8d3"                                   # the out-of-reach fill
# Categorical slots 1-5 of the validated palette, in fixed order (dataviz palette.md), each with a
# marker and a dash pattern, so identity never rests on colour: the paper prints in greyscale.
SEARCHERS = {"ppo": ("Telemachus S (ML)", "#2a78d6", "o", "-"),
             "parallel": ("Parallel Track", "#eb6834", "s", (0, (5, 2))),
             "expanding-square": ("Expanding Square", "#1baf7a", "^", (0, (1, 1.4))),
             "greedy": ("Greedy (reads the map)", "#eda100", "D", (0, (5, 1.5, 1, 1.5))),
             "telemachus": ("Telemachus 2 (ML)", "#e87ba4", "v", (0, (2.5, 1.5)))}
WIDE = ("one helicopter, k× the sweep width", "#6250d6", "o", "-")
FLEET = ("k helicopters", "#008300", "s", (0, (1, 1.2)))


def group_mean(d: pd.DataFrame, column: str) -> float:
    return float(d.groupby("group")[column].mean().mean())


def interval(d: pd.DataFrame, column: str, rng) -> tuple[float, float]:
    by = d.groupby("group")[column].mean()
    boot = [by.sample(len(by), replace=True, random_state=rng).mean() for _ in range(RESAMPLES)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def binned(df: pd.DataFrame, column: str, rng) -> pd.DataFrame:
    """Group-first mean per coverage bin, and its 95 % interval resampling groups in the bin."""
    out = []
    for _, d in df.groupby("bin", observed=True):
        if len(d) < MIN_WINDOWS or column not in d:
            continue
        lo, hi = interval(d, column, rng)
        out.append({"c": float(np.exp(np.log(d["coverage"]).mean())), "mean": group_mean(d, column),
                    "lo": lo, "hi": hi, "windows": len(d)})
    return pd.DataFrame(out)


def numbers(df: pd.DataFrame, fleet: pd.DataFrame | None) -> dict:
    """What the paper quotes, by arrival, group-first: the decomposition and the effort options."""
    out = {}
    for h, d in df.groupby("arrival_h"):
        line = {"windows": len(d), "groups": int(d["group"].nunique()),
                "coverage_median": float(d["coverage"].median()),
                "area90_median_km2": float(d["area90_km2"].median()),
                "ceiling": group_mean(d, "teleport"), "ceiling_split": group_mean(d, "teleport_split"),
                "out_of_reach": 1 - group_mean(d, "teleport")}
        for s in [s for s in SEARCHERS if s in d]:
            line[f"{s}_pos"] = group_mean(d, s)
            line[f"{s}_flown_loss"] = group_mean(d.assign(x=d["teleport"] - d[s]), "x")
        for k in KS[1:]:
            if f"teleport_x{k}" in d:
                line[f"ceiling_x{k}"] = group_mean(d, f"teleport_x{k}")
        if fleet is not None:
            f = fleet[fleet["arrival_h"] == h]
            for k in KS:
                for kind in (("k1",) if k == 1 else (f"wide{k}", f"fleet{k}")):
                    line[f"{kind}_pos"] = group_mean(f, f"{kind}_pos")
                    line[f"{kind}_found"] = group_mean(f.assign(x=f[f"{kind}_found"].astype(float)), "x")
        out[str(h)] = line
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_a(ax, df, rng, drawn):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _style(ax)
    ax.set_xscale("log")
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    arrivals = df.groupby("arrival_h")["coverage"].median()
    top = ax.secondary_xaxis("top")
    top.set_xticks(list(arrivals.values), [f"{h * 60:g} min" if h < 1 else f"{h:g} h" for h in arrivals.index])
    top.minorticks_off()
    top.tick_params(length=3, labelsize=6.5, colors=INK_2)
    top.spines["top"].set_visible(False)

    ceil, split = binned(df, "teleport", rng), binned(df, "teleport_split", rng)
    ax.fill_between(ceil["c"], ceil["mean"], 1.0, color=REACH, linewidth=0)
    ax.fill_between(ceil["c"], split["mean"], ceil["mean"], color=MUTED, alpha=0.35, linewidth=0)
    ax.plot(ceil["c"], ceil["mean"], color=INK, linewidth=2.2, zorder=4)
    drawn["A_ceiling"], drawn["A_ceiling_split"] = ceil.to_dict("records"), split.to_dict("records")
    handles = [Patch(color=REACH, label="out of reach of one sortie"),
               Line2D([], [], color=INK, linewidth=2.2, label="ceiling: the most any searcher could clear")]
    for key, (name, colour, marker, dash) in SEARCHERS.items():
        if key not in df:
            continue
        s = binned(df, key, rng)
        if key in ("ppo", "parallel"):
            ax.fill_between(s["c"], s["lo"], s["hi"], color=colour, alpha=0.14, linewidth=0)
        line, = ax.plot(s["c"], s["mean"], color=colour, linewidth=1.5 if key in ("ppo", "parallel") else 1.1,
                        linestyle=dash, marker=marker, markersize=3.2, markeredgecolor=SURFACE,
                        markeredgewidth=0.7, label=name, zorder=5 if key == "ppo" else 3)
        handles.append(line)
        drawn[f"A_{key}"] = s.to_dict("records")
    # The legend sits in the out-of-reach corner, which no line crosses.
    ax.legend(handles=handles, loc="upper left", fontsize=6.3, frameon=False, handlelength=2.4,
              borderaxespad=0.3, labelspacing=0.3)
    ax.annotate("C = 1: the cloud is as big\nas one sortie can sweep", (1.0, 0.03), xytext=(4, 0),
                textcoords="offset points", fontsize=6.2, color=INK_2, ha="left", va="bottom")
    ax.set_xlabel("coverage factor C = one sortie's sweep ÷ the cloud's 90 % area", fontsize=7, color=INK_2)
    ax.set_ylabel("probability cleared (POS)", fontsize=7.5)
    ax.set_title("A  One helicopter, 45 min: the ceiling, and five searchers under it",
                 loc="left", fontsize=8, color=INK, pad=13)


def panel_b(ax, df, numbers_by_h):
    _style(ax)
    ax.grid(False)
    rows = [h for h in sorted(df["arrival_h"].unique())]
    labels = []
    y = 0.0
    for h in rows:
        n = numbers_by_h[str(h)]
        for key, name in (("ppo", "S"), ("parallel", "PT")):
            if f"{key}_pos" not in n:
                continue
            got, lost = n[f"{key}_pos"], n["ceiling"] - n[f"{key}_pos"]
            reach = n["out_of_reach"]
            colour = SEARCHERS[key][1]
            ax.barh(y, got, color=colour, height=0.72, edgecolor=SURFACE, linewidth=1)
            ax.barh(y, lost, left=got, color=colour, alpha=0.35, height=0.72, edgecolor=SURFACE,
                    linewidth=1, hatch="////")
            ax.barh(y, reach, left=got + lost, color=REACH, height=0.72, edgecolor=SURFACE, linewidth=1)
            ax.text(min(got, 0.98) - 0.015, y, f"{got * 100:.0f}", ha="right", va="center", fontsize=6,
                    color="white", fontweight="bold")
            if lost >= 0.025:
                ax.text(got + lost / 2, y, f"{lost * 100:.0f}", ha="center", va="center", fontsize=6, color=INK)
            if reach >= 0.04:
                ax.text(got + lost + reach / 2, y, f"{reach * 100:.0f}", ha="center", va="center",
                        fontsize=6, color=INK_2)
            labels.append((y, f"{h * 60:g} min, {name}" if h < 1 else f"{h:g} h, {name}"))
            y += 0.8
        y += 0.45
    ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_title("B  Where the probability goes (%)", loc="left", fontsize=8, color=INK, pad=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label="cleared"),
                       Patch(facecolor=MUTED, alpha=0.35, hatch="////", edgecolor=SURFACE,
                             label="lost to how it flew"),
                       Patch(color=REACH, label="out of reach")],
              loc="upper center", bbox_to_anchor=(0.42, -0.07), ncol=3, fontsize=6, frameon=False,
              handlelength=1.2, columnspacing=0.8)


def panel_c(axes, df, fleet, numbers_by_h, rng, drawn):
    hours = sorted(df["arrival_h"].unique())
    for ax, h in zip(axes, hours):
        _style(ax)
        d = df[df["arrival_h"] == h]
        ks = [k for k in KS if (k == 1 and "teleport" in d) or f"teleport_x{k}" in d]
        ceil = [group_mean(d, "teleport" if k == 1 else f"teleport_x{k}") for k in ks]
        ax.plot(ks, ceil, color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)), zorder=2,
                label="ceiling at k× the sweep")
        drawn[f"C_{h}_ceiling"] = dict(zip(ks, ceil))
        if "ppo" in d:
            s = group_mean(d, "ppo")
            ax.plot([1], [s], marker=SEARCHERS["ppo"][2], color=SEARCHERS["ppo"][1], markersize=4.5,
                    linestyle="none", zorder=6, label="Telemachus S, one helicopter")
        if fleet is not None:
            f = fleet[fleet["arrival_h"] == h]
            for name, (label, colour, marker, dash) in (("wide", WIDE), ("fleet", FLEET)):
                cols = ["k1_pos"] + [f"{name}{k}_pos" for k in KS[1:]]
                means = [group_mean(f, c) for c in cols]
                cis = [interval(f, c, rng) for c in cols]
                ax.fill_between(KS, [c[0] for c in cis], [c[1] for c in cis], color=colour, alpha=0.12,
                                linewidth=0)
                ax.plot(KS, means, color=colour, linewidth=1.6, linestyle=dash, marker=marker, markersize=3.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.7, label=label, zorder=4)
                drawn[f"C_{h}_{name}"] = dict(zip(KS, means))
            gap = (group_mean(f, "wide4_pos") - group_mean(f, "fleet4_pos")) * 100
            ax.annotate(f"{gap:+.0f}", (4, (group_mean(f, "wide4_pos") + group_mean(f, "fleet4_pos")) / 2),
                        xytext=(5, 0), textcoords="offset points", fontsize=6.3, color=INK, va="center",
                        fontweight="bold")
            ax.plot([4.08, 4.08], [group_mean(f, "fleet4_pos"), group_mean(f, "wide4_pos")], color=INK_2,
                    linewidth=0.8)
        n = numbers_by_h[str(h)]
        ax.set_title(f"{h * 60:g} min" if h < 1 else f"{h:g} h", loc="left", fontsize=7.5, color=INK, pad=3)
        ax.text(0.97, 0.04, f"cloud {n['area90_median_km2']:.0f} km²\nC = {n['coverage_median']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK_2)
        ax.set_xticks(KS, [str(k) for k in KS], fontsize=6.5)
        ax.set_xlim(0.7, 4.75)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("POS", fontsize=7.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    return axes[0].get_legend_handles_labels()


def draw(df: pd.DataFrame, fleet: pd.DataFrame | None, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "hatch.color": SURFACE,
                         "hatch.linewidth": 0.8})
    fig = plt.figure(figsize=(7.2, 6.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.6, left=0.08, right=0.985,
                             top=0.915, bottom=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.32)
    a, b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    bottom = outer[1].subgridspec(1, 4, wspace=0.08)
    c = [fig.add_subplot(bottom[i]) for i in range(4)]

    nums = numbers(df, fleet)
    drawn = {}
    panel_a(a, df, rng, drawn)
    panel_b(b, df, nums)
    handles, labels = panel_c(c, df, fleet, nums, rng, drawn)
    fig.text(0.08, 0.455, "C  More effort: one helicopter that sees k× as wide, or k helicopters "
             "(bold: the gap at k = 4, POS points)", fontsize=8, color=INK)
    fig.text(0.5, 0.072, "k: sweep, as a multiple of one MH-60T sortie (45 min, 90 kt, W = 185 m)",
             ha="center", fontsize=7, color=INK_2)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=6.5,
               frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.text(0.985, 0.002, "Not modelled: cost, fuel, crews, aircraft availability, keeping k aircraft apart; "
             "a k-fold sweep width stands for a better sensor, not better eyes.",
             ha="right", va="bottom", fontsize=5.6, color=MUTED, style="italic")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    return {"numbers": nums, "drawn": drawn}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--windows", type=Path, required=True, help="ceiling.py summary's windows.csv")
    p.add_argument("--fleet", type=Path, help="fleet.py summary's fleet.csv (panel C's flights)")
    p.add_argument("--out", type=Path, required=True, help="figure path without extension")
    args = p.parse_args(argv)
    df = pd.read_csv(args.windows)
    df["coverage"] = Z_KM2 / df["area90_km2"]
    df["bin"] = pd.cut(df["coverage"], BINS)
    fleet = None
    if args.fleet is not None:
        fleet = pd.read_csv(args.fleet)
        fleet = fleet[fleet["arrival_h"].isin(df["arrival_h"].unique())]
    result = draw(df, fleet, args.out)
    record = {"windows": len(df), "groups": int(df["group"].nunique()),
              "fleet_windows": None if fleet is None else len(fleet), **result}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps(result["numbers"], indent=1, default=float))


if __name__ == "__main__":
    main()
