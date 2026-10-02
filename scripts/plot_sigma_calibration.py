"""plot_sigma_calibration.py: the figures for the sigma calibration (#89).

Reads what sar.validate.calibrate_sigma wrote and draws the report figures, each with a
JSON of the numbers beside it, so a figure can be checked without re-running anything:

  sigma_separation_by_tier   Table A as a picture: how far the model, and two naive
                             forecasts, end up from the buoy, drogued and undrogued
  sigma_msd_growth           mean squared gap against time on log axes, with t and t^2
  sigma_by_horizon           sigma by estimator at 6, 12, 24 and 48 h, with CIs
  sigma_wind_frame           the 24 h gaps, downwind against crosswind
  sigma_coverage_ladder      coverage of the 90 % region at 24 h against sigma
  sigma_coverage_by_lead     coverage at the final sigma, by lead time and by tier
  sigma_twin_recovery        planted sigma against what the method gave back

    python scripts/plot_sigma_calibration.py --windows <stem> --stage1 <dir> \
        --fit <fit.json> [--calibration <calibration.json>] --out figures/report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sar.validate.calibrate_sigma import CALIBRATION_HOUR, LEVEL, load_stage1  # noqa: E402
from sar.validate.drift_windows import Windows  # noqa: E402

# The dataviz reference palette: categorical slots in fixed order, chrome in ink tokens.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "font.size": 10,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
})


def save(fig, out: Path, name: str, data: dict) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, default=float))
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=9, color=INK2)


def separation_by_tier(rows, out):
    """Median separation at every hour, both tiers, model with and without leeway."""
    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, tier in zip(axes, ("drogued", "undrogued")):
        sub = rows[(rows["tier"] == tier) & rows["valid"]]
        series = {
            "model, leeway 2 %": (sub[np.isclose(sub["alpha"], 0.02)], "sep", BLUE),
            "model, no leeway": (sub[np.isclose(sub["alpha"], 0.0)], "sep", ORANGE),
            "buoy stays put": (sub[np.isclose(sub["alpha"], 0.02)], "sep_still", MUTED),
            "buoy keeps its first velocity": (sub[np.isclose(sub["alpha"], 0.02)],
                                              "sep_persist", INK2),
        }
        data[tier] = {}
        for name, (q, col, colour) in series.items():
            med = q.groupby("hour")[col].median() / 1e3
            data[tier][name] = med.round(3).to_dict()
            ax.plot(med.index, med.to_numpy(), color=colour, label=name)
        n = int(sub[(sub["hour"] == 24) & np.isclose(sub["alpha"], 0.02)].shape[0])
        ax.set_title(f"{tier.capitalize()} buoys ({n:,} windows at 24 h)")
        ax.set_xlabel("hours after the start")
        ax.set_xticks([0, 6, 12, 24, 36, 48])
        ax.set_xlim(0, 48)
    axes[0].set_ylabel("median distance from the buoy (km)")
    axes[1].legend(loc="upper left")
    fig.suptitle("How far each forecast ends up from the real buoy (dev windows, sigma = 0)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, out, "sigma_separation_by_tier", data)


def msd_growth(fit, out):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = {}
    for key, colour in ((f"undrogued, alpha=0.02", BLUE), (f"drogued, alpha=0.02", ORANGE)):
        b = fit["beta"][key]
        t = np.array([int(h) for h in b["msd_km2_by_hour"]])
        msd = np.array(list(b["msd_km2_by_hour"].values()))
        data[key] = {"beta": b["value"], "ci95": b["ci95"], "msd_km2_by_hour": b["msd_km2_by_hour"]}
        ax.loglog(t, msd, color=colour, label=f"{key.split(',')[0]}: beta = {b['value']:.2f}")
    t = np.array([6.0, 48.0])
    ref = data["undrogued, alpha=0.02"]["msd_km2_by_hour"]
    m6 = float(list(ref.values())[0])
    ax.loglog(t, m6 * t / 6, color=MUTED, lw=1)
    ax.loglog(t, m6 * (t / 6) ** 2, color=MUTED, lw=1)
    end_label(ax, 48, m6 * 8, "grows as t (random walk)")
    end_label(ax, 48, m6 * 64, "grows as t^2 (persistent error)")
    ax.set_xlabel("hours after the start")
    ax.set_ylabel("mean squared gap (km^2)")
    ax.set_xticks([6, 12, 24, 48], ["6", "12", "24", "48"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left")
    ax.set_title("How the gap grows: the exponent beta", loc="left")
    save(fig, out, "sigma_msd_growth", data)


def by_horizon(fit, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    data = {}
    hours = [6, 12, 24, 48]
    for (tier, est), colour, dx in (
            (("undrogued", "sigma_quantile"), BLUE, -0.6),
            (("undrogued", "sigma_debiased"), AQUA, -0.2),
            (("drogued", "sigma_quantile"), ORANGE, 0.2),
            (("drogued", "sigma_debiased"), YELLOW, 0.6)):
        rows = fit["table_a"][f"{tier}, alpha=0.02"]
        v = [rows[str(h)][est]["value"] for h in hours]
        lo = [rows[str(h)][est]["ci95"][0] for h in hours]
        hi = [rows[str(h)][est]["ci95"][1] for h in hours]
        label = f"{tier}, {est.split('_')[1]}"
        data[label] = {"hours": hours, "value": v, "ci95_low": lo, "ci95_high": hi}
        x = np.array(hours) + dx
        ax.errorbar(x, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], color=colour,
                    marker="o", ms=5, lw=2, capsize=0, label=label)
    ax.axvline(CALIBRATION_HOUR, color=AXIS, lw=1)
    ax.set_xticks(hours)
    ax.set_xlabel("horizon T (hours)")
    ax.set_ylabel("sigma (m s^-1/2)")
    ax.legend(loc="best")
    ax.set_title("sigma depends on the horizon it is matched at", loc="left")
    save(fig, out, "sigma_by_horizon", data)


def wind_frame(rows, out):
    p = rows[(rows["tier"] == "undrogued") & np.isclose(rows["alpha"], 0.02)
             & (rows["hour"] == CALIBRATION_HOUR) & rows["valid"]]
    p = p[np.isfinite(p["gap_dw"])]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(p["gap_cw"] / 1e3, p["gap_dw"] / 1e3, s=3, color=BLUE, alpha=0.25, lw=0)
    m = (p["gap_cw"].mean() / 1e3, p["gap_dw"].mean() / 1e3)
    ax.scatter([m[0]], [m[1]], s=64, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    end_label(ax, m[0], m[1], f"mean ({m[0]:.1f}, {m[1]:.1f}) km")
    lim = float(np.nanpercentile(np.abs(p[["gap_cw", "gap_dw"]].to_numpy()), 99) / 1e3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("crosswind gap, buoy minus model (km, + to the right of the wind)")
    ax.set_ylabel("downwind gap (km, - means the model ran ahead)")
    ax.set_title(f"Undrogued buoys at 24 h ({len(p):,} windows)", loc="left")
    save(fig, out, "sigma_wind_frame", {
        "windows": int(len(p)), "mean_km": m,
        "sd_km": [float(p["gap_cw"].std() / 1e3), float(p["gap_dw"].std() / 1e3)]})


def coverage_ladder(cal, out):
    lad = cal["ladder"]
    s, c = np.array(lad["sigmas"]), np.array(lad["coverage"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogx(s, c, color=BLUE, marker="o", ms=6)
    ax.axhline(LEVEL, color=MUTED, lw=1)
    star = lad["sigma_star"]
    ax.axvspan(*star["ci95"], color=BLUE, alpha=0.1, lw=0)
    ax.axvline(star["value"], color=BLUE, lw=1)
    end_label(ax, star["value"], 0.6, f"sigma* = {star['value']:.1f}")
    end_label(ax, s[-1], LEVEL, "90 % target", dy=-8)
    ax.set_xticks(s, [f"{v:.0f}" for v in s])
    ax.set_xlabel("sigma (m s^-1/2)")
    ax.set_ylabel("buoys inside the 90 % region at 24 h")
    ax.set_ylim(0, 1)
    ax.set_title(f"The ladder: coverage against sigma ({lad['windows']:,} undrogued windows)",
                 loc="left")
    save(fig, out, "sigma_coverage_ladder", lad)


def coverage_by_lead(cal, out):
    rel = cal["confirm"]["reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = {}
    for tier, colour in (("undrogued", BLUE), ("drogued", ORANGE)):
        if tier not in rel:
            continue
        leads = sorted(int(k) for k in rel[tier])
        c90 = [rel[tier][str(h)]["coverage"][str(LEVEL)] if str(h) in rel[tier]
               else rel[tier][h]["coverage"][str(LEVEL)] for h in leads]
        data[tier] = dict(zip(leads, c90))
        axes[0].plot(leads, c90, color=colour, marker="o", ms=5, label=tier)
        r24 = rel[tier].get(str(CALIBRATION_HOUR), rel[tier].get(CALIBRATION_HOUR))
        nominal = [float(k) for k in r24["coverage"]]
        observed = list(r24["coverage"].values())
        axes[1].plot(nominal, observed, color=colour, marker="o", ms=5, label=tier)
    axes[0].axhline(0.9, color=MUTED, lw=1)
    axes[0].axhline(0.8, color=AXIS, lw=1)
    end_label(axes[0], 48, 0.9, "90 %", dy=6)
    end_label(axes[0], 48, 0.8, "80 % (R2d)", dy=-6)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("hours after the start")
    axes[0].set_ylabel("buoys inside the 90 % region")
    axes[0].set_title(f"At sigma = {cal['confirm']['sigma']:.1f}, by lead time", loc="left")
    axes[0].legend(loc="lower left")
    axes[1].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[1].set_xlim(0.4, 1)
    axes[1].set_ylim(0.4, 1)
    axes[1].set_xlabel("stated probability of the region")
    axes[1].set_ylabel("fraction of buoys inside it")
    axes[1].set_title("Reliability at 24 h: on the diagonal is honest", loc="left")
    save(fig, out, "sigma_coverage_by_lead", {"coverage90_by_lead": data, "reliability": rel})


def twin_recovery(cal, out):
    tw = cal["twin"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    data = {}
    for case, v in tw.items():
        planted = float(case.replace("sigma", "").split("+")[0])
        s1 = v["stage1_sigma_quantile"]["value"]
        s2 = v["stage2"]["sigma_star"]
        data[case] = {"planted": planted, "stage1_formula": s1, "stage2_ladder": s2}
        slide = "+slide" in case
        ax.scatter([planted], [s1], s=48, color=ORANGE, marker="s" if slide else "o",
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.errorbar([planted], [s2["value"]],
                    yerr=[[s2["value"] - s2["ci95"][0]], [s2["ci95"][1] - s2["value"]]],
                    color=BLUE, marker="D" if slide else "o", ms=7, capsize=0, zorder=4)
    lim = [10, 130]
    ax.plot(lim, lim, color=AXIS, lw=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.scatter([], [], color=ORANGE, s=48, label="stage 1 formula")
    ax.scatter([], [], color=BLUE, s=48, label="stage 2 ladder (sigma*)")
    ax.legend(loc="upper left")
    ax.set_xlabel("sigma planted in the fake buoys")
    ax.set_ylabel("sigma the method gave back")
    ax.set_title("Twin experiment: does the method recover a known sigma?", loc="left")
    save(fig, out, "sigma_twin_recovery", data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--windows", required=True)
    p.add_argument("--stage1", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--calibration")
    p.add_argument("--out", default="figures/report")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, _ = load_stage1(Windows.load(args.windows), args.stage1)
    fit = json.loads(Path(args.fit).read_text())
    separation_by_tier(rows, out)
    msd_growth(fit, out)
    by_horizon(fit, out)
    wind_frame(rows, out)
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        if "ladder" in cal:
            coverage_ladder(cal, out)
        if "confirm" in cal:
            coverage_by_lead(cal, out)
        if "twin" in cal:
            twin_recovery(cal, out)


if __name__ == "__main__":
    main()
